import argparse

import pandas as pd
import numpy as np

from schedules import OracleSchedule, StartDayScheduleDetElectivesDetEmerg, \
    StartDayScheduleStochElectives
from jobs import DetElectivesDetEmergencies, StochElectivesStochEmergencies, \
    default_elective_jobs, default_emergency_jobs
from heuristics import SimulatedAnnealing, LocalSearch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--rooms', type=int, default=5)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--n_emerg', type=int, default=None)
    parser.add_argument('--n_emerg_lambda', type=float, default=12)
    parser.add_argument('--elective_family_nums', type=int, nargs=4, default=[4, 4, 2, 2])
    parser.add_argument('--obj_weights', type=int, nargs=5, default=[1, 500, 1500, 3, 10])
    parser.add_argument('--bim', action='store_true', default=False)
    args = parser.parse_args()

    # jobs = DetElectivesDetEmergencies(default_elective_jobs, default_emergency_jobs)
    # n, elect_df, emerg_df = jobs.get_jobs()
    # schedule = OracleSchedule(electives_df=elect_df, emerg_df=emerg_df, n_electives=n)

    # This seeds the jobs but not the solutions
    # TODO: Find out a way to seed the solutions
    if args.seed:
        np.random.seed(args.seed)

    num_p, num_o, num_h, num_c = args.elective_family_nums
    elective_families = [
        *['Plastic'] * num_p,
        *['Orthopaedic'] * num_o,
        *['Hepatobilary'] * num_h,
        *['Cardiothoracic'] * num_c,
    ]
    job_generator = StochElectivesStochEmergencies(
        elective_families=elective_families,
        n_emerg=args.n_emerg,
        n_emerg_lambda=args.n_emerg_lambda,
    )
    job_generator.generate_samples(args.samples)
    n_electives, elective_dfs, emerg_dfs = job_generator.get_jobs()
    for sample in range(args.samples):
        print(f'Sample {sample + 1}')
        print(pd.concat([elective_dfs[sample], emerg_dfs[sample]], ignore_index=True))
        print()

    schedule = StartDayScheduleStochElectives(
        n_electives=n_electives,
        electives_dfs=elective_dfs,
        emerg_dfs=emerg_dfs,
        n_rooms=args.rooms,
        obj_weights=args.obj_weights,
        bim=args.bim,
    )

    objective = schedule.eval_schedule()
    result, delays = schedule.get_schedule()
    print('stochastic electives, start of day, initial:')
    print(objective, result)
    print()

    sa = SimulatedAnnealing(schedule)
    ls = LocalSearch(schedule)
    sa.sa()
    objective = schedule.eval_schedule()
    result, delays = schedule.get_schedule()
    print('stochastic electives, start of day, after SA:')
    print(objective, result, delays)
    print('\n')

    ls.local_search()
    objective = schedule.eval_schedule(True)
    result, delays = schedule.get_schedule()
    print('stochastic electives, start of day, after SA and LS:')
    print(objective, result, delays)
    print('\n')

    for sample in range(args.samples):
        jobs_df = pd.concat([elective_dfs[sample], emerg_dfs[sample]], ignore_index=True)
        schedule.plot(result, delays, jobs_df)

    schedule.produce_end_day_schedule()
    objective, _, results = schedule.eval_end_day_schedule(True)
    print('stochastic electives, end of day:')
    print(objective)
    for result, delays in results:
        print(result, delays)
        jobs_df = pd.concat([elective_dfs[sample], emerg_dfs[sample]], ignore_index=True)
        schedule.plot(result, delays, jobs_df)

if __name__ == '__main__':
    main()