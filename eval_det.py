# Script for evaluating a deterministic schedule in a stochastic situation

import pickle
import pandas as pd
from schedules import StartDayScheduleStochElectives

jobs_file = 'default_jobs'
samples = 10
rooms = 5
obj_weights = [1, 500, 1500, 3, 10]
results_file = 'det_el_bim'
bim = True

with open(f'jobs/{jobs_file}.pkl', 'rb') as f:
    n_electives, elective_dfs, emerg_dfs = pickle.load(f)
with open(f'results/{results_file}.pkl', 'rb') as f:
    output = pickle.load(f)
_, _, start_result, start_delays = output['start']

jobs_dfs = [pd.concat([elective_dfs[i], emerg_dfs[i]], ignore_index=True)
    for i in range(samples)]
schedule = StartDayScheduleStochElectives(
    n_electives=n_electives,
    n_rooms=rooms,
    electives_dfs=elective_dfs,
    emerg_dfs=emerg_dfs,
    obj_weights=obj_weights,
    bim=bim,
    schedule=start_result,
    delays=start_delays,
)
start_obj, start_obj_detailed = schedule.eval_schedule(detailed=True)
print('start:')
print(start_obj)
print(start_obj_detailed)
print()

schedule.produce_end_day_schedule()
(end_obj, end_obj_detailed), end_results = \
    schedule.eval_end_day_schedule(detailed=True)
end_obj_detailed_av = [0 for _ in range(5)]
for details in end_obj_detailed:
    for i in range(5):
        end_obj_detailed_av[i] += details[i] / len(end_obj_detailed)
print('end:')
print(round(sum(end_obj) / len(end_obj)))
print(end_obj_detailed_av)