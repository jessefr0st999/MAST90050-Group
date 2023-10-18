import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, max_, min_, or_, and_

import pickle

BIG_M = 1e10
ROOM_OPEN_TIME = 8*60
ROOM_CLOSE_TIME = 17*60

def exact_solve(jobs_df, n_electives, n_rooms, obj_weights=[1, 500, 1500, 3, 10], soft_time=1*60, hard_time=1*60, soft_gap=0.05):
    
    # constants
    n_jobs = len(jobs_df)

    clean_t_same = 20
    clean_t_diff = 90
    family_list = np.array(jobs_df.family)
    diff_family = (family_list[:, None] != family_list[:, None].T).astype(int)
    clean_t = clean_t_same + diff_family * (clean_t_diff - clean_t_same)

    J = range(n_jobs)
    E = range(n_jobs - n_electives)
    R = range(n_rooms)

    model = gp.Model()
    model.setParam('IntegralityFocus', 1)
    model.setParam('Heuristics', 0.33)


    # variables
    first = model.addVars(R, J, name='first', vtype=GRB.BINARY)
    adj = model.addVars(J, J, name='adj', vtype=GRB.BINARY)

    s = model.addVars(J, name='s', vtype=GRB.CONTINUOUS)
    c = model.addVars(J, name='c', vtype=GRB.CONTINUOUS)

    emerg_wait = model.addVars(E, name='emerg_wait', vtype=GRB.CONTINUOUS)
    emerg_wait_ = model.addVars(E, name='emerg_wait_', vtype=GRB.CONTINUOUS)

    emerg_wait_exponent = model.addVars(E, name='emerg_wait_exponent', vtype=GRB.CONTINUOUS)
    b = model.addVars(J, name='b', vtype=GRB.CONTINUOUS)
    in_gap = model.addVars(J, name='in_gap', vtype=GRB.BINARY)
    wait = model.addVars(J, J, name='wait', vtype=GRB.CONTINUOUS)
    min_wait = model.addVars(J, name='min_wait', vtype=GRB.CONTINUOUS)

    finish_k_geq_start_j = model.addVars(J, J, name='finish_k_geq_start_j', vtype=GRB.BINARY)
    start_j_gt_start_k = model.addVars(J, J, name='start_j_gt_start_k', vtype=GRB.BINARY)
    start_j_geq_finish_k = model.addVars(J, J, name='start_j_geq_finish_k', vtype=GRB.BINARY)

    before_first = model.addVars(J, J, name='before_first', vtype=GRB.BINARY)
    after_last = model.addVars(J, J, name='after_last', vtype=GRB.BINARY)
    between_adj = model.addVars(J, J, J, name='between_adj', vtype=GRB.BINARY)

    job_first = model.addVars(J, name='job_first', vtype=GRB.BINARY)
    job_not_last = model.addVars(J, name='job_not_last', vtype=GRB.BINARY)
    job_last = model.addVars(J, name='job_last', vtype=GRB.BINARY)


    max_run = model.addVar(vtype=GRB.CONTINUOUS, name="max_run")
    max_run_ = model.addVar(vtype=GRB.CONTINUOUS, name="max_run_")
    rooms_open = model.addVar(vtype=GRB.INTEGER, name="rooms_open")
    weighted_emerg_wait = model.addVar(vtype=GRB.CONTINUOUS, name="weighted_emerg_wait")
    tardiness = model.addVar(vtype=GRB.CONTINUOUS, name="tardiness")
    lateness = model.addVar(vtype=GRB.CONTINUOUS, name="lateness")
    bim = model.addVar(vtype=GRB.CONTINUOUS, name="bim")

    model.setObjective(obj_weights[0]*max_run + obj_weights[1]*rooms_open + obj_weights[2]*weighted_emerg_wait + obj_weights[3]*tardiness + obj_weights[4]*bim, GRB.MINIMIZE)


    # job does not come after itself
    model.addConstrs(adj[j,j] == 0 for j in J)
    # at most one job follows another
    model.addConstrs((sum(adj[j,k] for k in J) <= 1) for j in J)
    # at most one job is the first in a room
    model.addConstrs((sum(first[r,j] for j in J) <= 1) for r in R)
    # a job is either first in its room, or comes after another
    model.addConstrs(((sum(first[r,j] for r in R) + sum(adj[k,j] for k in J)) == 1) for j in J)

    # try to cut down on some permutations
    for r in range(n_rooms - 1):
        model.addConstr(sum(first[r,j] for j in J) >= sum(first[r+1,j] for j in J))



    # determine latest finishing job. Final changeover time included in objective
    model.addConstr(max_run_ == max_((c[j])  for j in J) )
    model.addConstr(max_run == max_run_ + clean_t_same )


    # rooms open
    model.addConstr(rooms_open == sum(first[r,j]  for r in R for j in J))

    # emergency wait
    for e in E:
        i = n_electives + e
        arrival = jobs_df.iloc[i].arrival
        prio = jobs_df.iloc[i].priority
        model.addConstr(emerg_wait_exponent[e] == (s[i] - arrival)/60/(11 - prio))

        model.addGenConstrExpA(emerg_wait_exponent[e], emerg_wait_[e], 2)
        model.addConstr(emerg_wait[e] == emerg_wait_[e] - 1)

    model.addConstr(weighted_emerg_wait == sum(emerg_wait[e] for e in E))


    # TODO: consider modifying this to start at max of arrival time and a new variable delay
    #       Will help in stochastic case
    # job starts after arrival time
    model.addConstrs(s[j] >= jobs_df.iloc[j].arrival for j in J)
    # job ends after processing
    model.addConstrs(c[j] == s[j] + jobs_df.iloc[j].length for j in J)
    # next job cannot start until after changeover
    for j in J:
        for k in J:
            model.addGenConstrIndicator(adj[j,k], True, s[k] - c[j] >= clean_t[j,k])

    # determine tardiness
    model.addConstr(lateness == max_run - ROOM_CLOSE_TIME)
    model.addGenConstrMax(tardiness, [lateness], 0)

    # determine BIM
    model.addConstr(bim == max_((b[j])  for j in J) )

    # binary job time relations
    eps = 0.001
    model.addConstrs( (c[k] >= s[j] - 24*60*(1 - finish_k_geq_start_j[j,k])) for j in J for k in J)
    model.addConstrs( (c[k] <= s[j] - eps + 24*60*finish_k_geq_start_j[j,k]) for j in J for k in J)
    model.addConstrs( (s[j] >= s[k] + eps - 24*60*(1 - start_j_gt_start_k[j,k])) for j in J for k in J)
    model.addConstrs( (s[j] <= s[k] + 24*60*start_j_gt_start_k[j,k]) for j in J for k in J)
    model.addConstrs( (s[j] >= c[k] - 24*60*(1 - start_j_geq_finish_k[j,k])) for j in J for k in J)
    model.addConstrs( (s[j] <= c[k] - eps + 24*60*start_j_geq_finish_k[j,k]) for j in J for k in J)

    # determine wait times until a job finishes after every job start
    for j in J:
        for k in J:
            model.addGenConstrIndicator(finish_k_geq_start_j[j,k], True, c[k] - s[j] == wait[j,k])
            model.addGenConstrIndicator(finish_k_geq_start_j[j,k], False, wait[j,k] == 24*60)

    model.addConstrs(min_wait[j] == min_((wait[j,k]) for k in J) for j in J)

    # determine whether a job starts in a gap
    for j in J:
        model.addGenConstrIndicator(in_gap[j], True, b[j] == 0)
        model.addGenConstrIndicator(in_gap[j], False, b[j] == min_wait[j])

    model.addConstrs((in_gap[j] == or_( *(before_first[j,k] for k in J) , *(after_last[j,k] for k in J), *(between_adj[j,k,l] for k in J for l in J))) for j in J)

    model.addConstrs((job_first[j] == sum(first[r,j] for r in R)) for j in J)
    model.addConstrs((job_not_last[j] == sum(adj[j,k] for k in J)) for j in J)
    model.addConstrs((job_last[j] == -1*job_not_last[j] + 1) for j in J)


    model.addConstrs((before_first[j,k] == and_(job_first[k], start_j_gt_start_k[k,j])) for j in J for k in J)
    model.addConstrs((after_last[j,k] == and_(job_last[k], start_j_geq_finish_k[j,k])) for j in J for k in J)
    model.addConstrs((between_adj[j,k,l] == and_(adj[k,l], start_j_geq_finish_k[j,k], start_j_gt_start_k[l,j])) for j in J for k in J for l in J)




    def callback(model, where):
        if where == GRB.Callback.MIP and soft_time and soft_gap:
            run_time = model.cbGet(GRB.Callback.RUNTIME)
            obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
            obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            gap = abs((obj_best - obj_bound) / obj_best)
            if run_time > soft_time and gap < soft_gap:
                model.terminate()
    model.setParam('TimeLimit', hard_time)
    model.optimize(callback)

    return model