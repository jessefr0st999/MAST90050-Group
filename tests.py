import argparse
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from schedules import OracleSchedule, StartDayScheduleDetElectivesDetEmerg, \
    StartDayScheduleStochElectives
from jobs import DetElectivesDetEmergencies, StochElectivesStochEmergencies, \
    default_elective_jobs, default_emergency_jobs, det_electives
from heuristics import heuristic_optimise, heuristic_optimise_alt1, heuristic_optimise_alt2, heuristic_optimise_alt3, heuristic_optimise_alt4
from gurobi import exact_solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--rooms', type=int, default=5)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--n_emerg', type=int, default=None)
    parser.add_argument('--n_emerg_lambda', type=float, default=10)
    parser.add_argument('--elective_family_nums', type=int, nargs=4, default=[4, 3, 2, 1])
    parser.add_argument('--obj_weights', type=int, nargs=5, default=[1, 500, 1500, 3, 10])
    parser.add_argument('--bim', action='store_true', default=False)
    parser.add_argument('--heur_it', type=int, default=3)
    parser.add_argument('--det_el', action='store_true', default=False)
    parser.add_argument('--oracle', action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--heur_log', action='store_true', default=False)
    parser.add_argument('--results_file', default='test_results')
    parser.add_argument('--read_results', action='store_true', default=False)
    parser.add_argument('--jobs_file', default='test_jobs')
    parser.add_argument('--read_jobs', action='store_true', default=False)
    parser.add_argument('--alt_heur', action='store_true', default=False)
    parser.add_argument('--alt_heur_id', type=int, default=1)
    parser.add_argument('--det_counterpart', action='store_true', default=False)
    parser.add_argument('--gurobi', action='store_true', default=False)

    args = parser.parse_args()

    # This seeds the jobs but not the solutions
    if args.seed:
        np.random.seed(args.seed)

    # Determine what heuristic scheme to use
    selected_heuristic_optimise = heuristic_optimise
    if args.alt_heur:
        selected_heuristic_optimise = [heuristic_optimise, heuristic_optimise_alt1, heuristic_optimise_alt2, heuristic_optimise_alt3, heuristic_optimise_alt4][args.alt_heur_id]

    if args.gurobi:
        with open(f'jobs/{args.jobs_file}.pkl', 'rb') as f:
            n_electives, elective_dfs, emerg_dfs = pickle.load(f)
            elective_df = elective_dfs[0]

        model = exact_solve(jobs_df=elective_df, n_electives=n_electives, n_rooms=args.rooms, obj_weights=args.obj_weights)
        print(f"\nBest gurobi solution found for first sample of jobs file {args.job_file}: {model.ObjVal}\n")
        return

    # do the deterministic counterpart separately because this is getting messy :')
    if args.det_counterpart:
        # only run this with previously run stochastic, so jobs file exists
        with open(f'jobs/{args.jobs_file}.pkl', 'rb') as f:
                n_electives, elective_dfs, emerg_dfs = pickle.load(f)

        families = list(elective_dfs[0].family)
        elective_df = det_electives(families) 
        det_schedule = StartDayScheduleDetElectivesDetEmerg(
                n_electives=n_electives,
                n_rooms=args.rooms,
                electives_df=elective_df,
                emerg_df=pd.DataFrame(),
                obj_weights=args.obj_weights,
                bim=args.bim,
            )
        stoch_schedule = StartDayScheduleStochElectives(
            n_electives=n_electives,
            n_rooms=args.rooms,
            electives_dfs=elective_dfs,
            emerg_dfs=emerg_dfs,
            obj_weights=args.obj_weights,
            bim=args.bim,
        )

        # optimise deterministic case
        selected_heuristic_optimise(det_schedule, n_parallel=args.heur_it, log=args.heur_log)
        start_obj_det, start_obj_detailed_det = det_schedule.eval_schedule(args.log, detailed=True)
        start_result, start_delays = det_schedule.get_schedule()
        print(f'deterministic electives, start of day:')
        print(round(start_obj_det))
        print(start_obj_detailed_det)

        # set stochastic schedule to be result of deterministic 
        stoch_schedule.set_schedule(start_result, start_delays)

        start_obj, start_obj_detailed = stoch_schedule.eval_schedule(args.log, detailed=True)
        start_result, start_delays = stoch_schedule.get_schedule()
        print(f'stochastic electives, start of day, using deterministic counterpart schedule:')
        print(round(start_obj))
        print(start_obj_detailed)

        jobs_dfs = [pd.concat([elective_dfs[i], emerg_dfs[i]], ignore_index=True)
            for i in range(args.samples)]
        output = {
            'start_det': (start_obj, start_obj_detailed, start_result, start_delays),
            'start': (start_obj, start_obj_detailed, start_result, start_delays),
            'end': [],
            'jobs': jobs_dfs,
        }

        stoch_schedule.produce_end_day_schedule()
        (end_obj, end_obj_detailed), end_results = \
            stoch_schedule.eval_end_day_schedule(args.log, True)
        end_obj_detailed_av = [0 for _ in range(5)]
        for details in end_obj_detailed:
            for i in range(5):
                end_obj_detailed_av[i] += details[i] / len(end_obj_detailed)
        print(f'stochastic electives, end of day, deterministic counterpart start:')
        print(round(sum(end_obj) / len(end_obj)))
        print(end_obj_detailed_av)
        for i, (end_result, end_delays) in enumerate(end_results):
            output['end'].append((end_obj[i], end_obj_detailed[i], end_result, end_delays))
            if args.log:
                print(f'stochastic electives (sample {i + 1}), end of day:')
                print(end_obj[i])
                print(end_obj_detailed[i])
                print(end_result)
                print(end_delays)
                print()
                stoch_schedule.plot(end_result, end_delays, jobs_dfs[i])

        with open(f'results/{args.results_file}.pkl', 'wb') as f:
            pickle.dump(output, f)

        return

    # First, generate the jobs and instance the schedule objects
    if args.det_el:
        if args.read_jobs:
            with open(f'jobs/{args.jobs_file}.pkl', 'rb') as f:
                n_electives, elective_dfs, emerg_dfs = pickle.load(f)
            elective_df = elective_dfs[0]
            emerg_df = emerg_dfs[0]

        else:
            jobs = DetElectivesDetEmergencies(default_elective_jobs, default_emergency_jobs)
            n_electives, elective_df, emerg_df = jobs.get_jobs()

        jobs_df = pd.concat([elective_df, emerg_df], ignore_index=True)
        if args.oracle:
            schedule = OracleSchedule(
                n_electives=n_electives,
                n_rooms=args.rooms,
                electives_df=elective_df,
                emerg_df=emerg_df,
                obj_weights=args.obj_weights,
            )
        else:
            schedule = StartDayScheduleDetElectivesDetEmerg(
                n_electives=n_electives,
                n_rooms=args.rooms,
                electives_df=elective_df,
                emerg_df=emerg_df,
                obj_weights=args.obj_weights,
                bim=args.bim,
            )
    else:
        num_p, num_o, num_h, num_c = args.elective_family_nums
        elective_families = [
            *['Plastic'] * num_p,
            *['Orthopaedic'] * num_o,
            *['Hepatobilary'] * num_h,
            *['Cardiothoracic'] * num_c,
        ]
        if args.read_jobs:
            with open(f'jobs/{args.jobs_file}.pkl', 'rb') as f:
                n_electives, elective_dfs, emerg_dfs = pickle.load(f)
        else:
            job_generator = StochElectivesStochEmergencies(
                elective_families=elective_families,
                n_emerg=args.n_emerg,
                n_emerg_lambda=args.n_emerg_lambda,
            )
            job_generator.generate_samples(args.samples)
            n_electives, elective_dfs, emerg_dfs = job_generator.get_jobs()
            with open(f'jobs/{args.jobs_file}.pkl', 'wb') as f:
                pickle.dump((n_electives, elective_dfs, emerg_dfs), f)
        jobs_dfs = [pd.concat([elective_dfs[i], emerg_dfs[i]], ignore_index=True)
            for i in range(args.samples)]
        if args.log:
            for i in range(args.samples):
                print(f'Sample {i + 1}')
                print(jobs_dfs[i])
                print()
        schedule = StartDayScheduleStochElectives(
            n_electives=n_electives,
            n_rooms=args.rooms,
            electives_dfs=elective_dfs,
            emerg_dfs=emerg_dfs,
            obj_weights=args.obj_weights,
            bim=args.bim,
        )
    
    # If schedules have already been calculated, plot them
    if args.read_results:
        with open(f'results/{args.results_file}.pkl', 'rb') as f:
            output = pickle.load(f)
        start_obj, start_obj_detailed, start_result, start_delays = output['start']
        print(f'start objective: {start_obj}')
        end_objectives = []
        end_obj_detailed_av = [0 for _ in range(5)]
        for i in range(args.samples):
            if i % 2 == 0:
                figure, axes = plt.subplots(2, 2, layout='compressed')
                axes = iter(axes.T.flatten())
            axis = next(axes)
            schedule.plot(start_result, start_delays, output['jobs'][i],
                title=f'sample {i + 1} start', axis=axis)
            if args.oracle:
                return
            end_obj, end_obj_detailed, end_result, end_delays = output['end'][i]
            end_objectives.append(end_obj)
            for j in range(5):
                end_obj_detailed_av[j] += end_obj_detailed[j] / args.samples
            print(f'sample {i + 1} end objective: {end_obj}')
            print(f'sample {i + 1} end objective details: {end_obj_detailed}')
            axis = next(axes)
            schedule.plot(end_result, end_delays, output['jobs'][i],
                title=f'sample {i + 1} end', axis=axis)
            if args.det_el:
                break
        plt.show()
        print(f'end objective: {round(sum(end_objectives) / len(end_objectives))}')
        print(f'end objective details: {[x for x in end_obj_detailed_av]}')
        return
    
    # Otherwise, calculate the optimal start of day then end of day schedules
    if args.det_el:
        selected_heuristic_optimise(schedule, n_parallel=args.heur_it, log=args.heur_log)
        start_obj, start_obj_detailed = schedule.eval_schedule(args.log, detailed=True)
        start_result, start_delays = schedule.get_schedule()
        print(f'deterministic electives, start of day:')
        print(round(start_obj))
        print(start_obj_detailed)
        if args.log:
            print(start_result)
            print(start_delays)
            print('\n')
            schedule.plot(start_result, start_delays, jobs_df)
        output = {
            'start': (start_obj, start_obj_detailed, start_result, start_delays),
            'end': [],
            'jobs': [jobs_df],
        }
        if not args.oracle:
            schedule.produce_end_day_schedule()
            (end_obj, end_obj_detailed), (end_result, end_delays) = \
                schedule.eval_end_day_schedule(args.log, detailed=True)
            print(f'deterministic electives, end of day:')
            print(round(end_obj))
            print(end_obj_detailed)
            if args.log:
                print(end_result)
                print(end_delays)
                print('\n')
                schedule.plot(end_result, end_delays, jobs_df)
            output['end'] = [(end_obj, end_obj_detailed, end_result, end_delays)]
        with open(f'results/{args.results_file}.pkl', 'wb') as f:
            pickle.dump(output, f)
    else:
        # Log the first guess for the start of day schedule
        if args.log:
            objective, obj_detailed = schedule.eval_schedule(detailed=True)
            result, delays = schedule.get_schedule()
            print(f'stochastic electives, start of day, initial:')
            print(round(objective))
            print(obj_detailed)
            print(result)
            print(delays)
            print()

        selected_heuristic_optimise(schedule, n_parallel=args.heur_it, log=args.heur_log)
        # Log the optimal start of day schedule
        start_obj, start_obj_detailed = schedule.eval_schedule(args.log, detailed=True)
        start_result, start_delays = schedule.get_schedule()
        print(f'stochastic electives, start of day, after optimisation:')
        print(round(start_obj))
        print(start_obj_detailed)
        if args.log:
            print(start_result)
            print(start_delays)
            print('\n')
            for i in range(args.samples):
                schedule.plot(start_result, start_delays, jobs_dfs[i])
        output = {
            'start': (start_obj, start_obj_detailed, start_result, start_delays),
            'end': [],
            'jobs': jobs_dfs,
        }

        schedule.produce_end_day_schedule()
        (end_obj, end_obj_detailed), end_results = \
            schedule.eval_end_day_schedule(args.log, True)
        end_obj_detailed_av = [0 for _ in range(5)]
        for details in end_obj_detailed:
            for i in range(5):
                end_obj_detailed_av[i] += details[i] / len(end_obj_detailed)
        print(f'stochastic electives, end of day:')
        print(round(sum(end_obj) / len(end_obj)))
        print(end_obj_detailed_av)
        for i, (end_result, end_delays) in enumerate(end_results):
            output['end'].append((end_obj[i], end_obj_detailed[i], end_result, end_delays))
            if args.log:
                print(f'stochastic electives (sample {i + 1}), end of day:')
                print(end_obj[i])
                print(end_obj_detailed[i])
                print(end_result)
                print(end_delays)
                print()
                schedule.plot(end_result, end_delays, jobs_dfs[i])

        with open(f'results/{args.results_file}.pkl', 'wb') as f:
            pickle.dump(output, f)

if __name__ == '__main__':
    main()