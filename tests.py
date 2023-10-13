import argparse
import pickle

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
    parser.add_argument('--stoch_el', action='store_true', default=False)
    parser.add_argument('--oracle', action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--results_file', default='test_results')
    parser.add_argument('--read', action='store_true', default=False)
    args = parser.parse_args()

    # TODO: implement deterministic electives and oracle
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
    jobs_dfs = [pd.concat([elective_dfs[i], emerg_dfs[i]], ignore_index=True)
        for i in range(args.samples)]
    if args.log:
        for i in range(args.samples):
            print(f'Sample {i + 1}')
            print(jobs_dfs[i])
            print()

    schedule = StartDayScheduleStochElectives(
        n_electives=n_electives,
        electives_dfs=elective_dfs,
        emerg_dfs=emerg_dfs,
        n_rooms=args.rooms,
        obj_weights=args.obj_weights,
        bim=args.bim,
    )
    
    if args.read:
        with open(f'results/{args.results_file}.pkl', 'rb') as f:
            output = pickle.load(f)
        start_obj, start_result, start_delays = output['start']
        end_objectives = []
        for i in range(args.samples):
            end_obj, end_result, end_delays = output['end'][i]
            print(f'sample {i + 1} end objective: {end_obj}')
            schedule.plot(start_result, start_delays, output['jobs'][i],
                title=f'sample {i + 1} start')
            schedule.plot(end_result, end_delays, output['jobs'][i],
                title=f'sample {i + 1} end')
            end_objectives.append(end_obj)
        print()
        print(f'start objective: {start_obj}')
        print(f'end objective: {sum(end_objectives) / len(end_objectives)}')
        return

    if args.log:
        objective = schedule.eval_schedule()
        result, delays = schedule.get_schedule()
        print(f'stochastic electives, start of day, initial:')
        print(objective)
        print(result)
        print(delays)
        print()

    sa = SimulatedAnnealing(schedule)
    ls = LocalSearch(schedule)
    sa.sa()
    if args.log:
        print('stochastic electives, start of day, after SA:')
        objective = schedule.eval_schedule()
        result, delays = schedule.get_schedule()
        print(objective)
        print(result)
        print(delays)
        print('\n')

    ls.local_search()
    print(f'stochastic electives, start of day, after SA and LS:')
    start_obj = schedule.eval_schedule(args.log)
    start_result, start_delays = schedule.get_schedule()
    print(start_obj)
    if args.log:
        print(start_result)
        print(start_delays)
        print('\n')
        for i in range(args.samples):
            schedule.plot(start_result, start_delays, jobs_dfs[i])
    output = {
        'start': (start_obj, start_result, start_delays),
        'end': [],
        'jobs': jobs_dfs,
    }

    schedule.produce_end_day_schedule()
    print(f'stochastic electives, end of day:')
    end_av_obj, end_obj, end_results = schedule.eval_end_day_schedule(args.log)
    print(end_av_obj)
    for i, (end_result, end_delays) in enumerate(end_results):
        output['end'].append((end_obj[i], end_result, end_delays))
        if args.log:
            print(f'stochastic electives (sample {i + 1}), end of day:')
            print(end_obj[i])
            print(end_result)
            print(end_delays)
            print()
            schedule.plot(end_result, end_delays, jobs_dfs[i])

    with open(f'results/{args.results_file}.pkl', 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    main()