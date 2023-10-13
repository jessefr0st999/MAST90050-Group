from itertools import product
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

BIG_M = 1e10
ROOM_OPEN_TIME = 8*60
ROOM_CLOSE_TIME = 17*60

MOVE = 0
SWAP = 1
DELAY_CHANGE = 2

PLOT_COLOURS = {
    'Cardiothoracic': 'red',
    'Orthopaedic': 'blue',
    'Plastic': 'green',
    'Hepatobilary': 'black',
}

class Schedule():
    '''
    A base schedule class that has functions (possibly abstract) for use in simulated annealing, 
    and functionality for evaluating a fixed schedule
    '''
    def __init__(self, schedule=[[]], delays = [], clean_t_same=20, clean_t_diff=90,
            n_rooms=6, obj_weights=[1,1,1,1,0]):
        self.schedule = schedule
        self.delays = delays
        self.obj_weights = deepcopy(obj_weights)
        self.clean_t_same = clean_t_same
        self.clean_t_diff = clean_t_diff
        self.n_rooms = n_rooms
        
        self.alt_schedule = [[]]
        self.alt_delays = []
    
    def eval_schedule(self, log=False):
        '''
        Abstract function
        Return an evaluation for the current schedule
        '''
        raise NotImplementedError
    
    def perturb_options(self):
        '''
        Give a list of all possible move/swap perturbations of the schedule
        '''
        options = []
        
        # add all moves
        job_indices = [(i, room.index(job)) for i, room in enumerate(self.schedule) \
            for job in room]
        destinations = [(i, x) for i, room in enumerate(self.schedule) \
            for x in range(len(room) + 1)]
        for orig_room, orig_index in job_indices:
            for dest_room, dest_index in destinations:
                if (orig_room, orig_index) != (dest_room, dest_index):
                    options.append((MOVE, orig_room, orig_index, dest_room, dest_index))

        # add all swaps
        for (j1_room, j1_index), (j2_room, j2_index) in product(job_indices, job_indices):
            if (j1_room, j1_index) != (j2_room, j2_index):
                options.append((SWAP, j1_room, j1_index, j2_room, j2_index))

        return options

    def perturb_schedule(self, perturbation=None):
        '''
        Make a specified perturbation to the schedule (by swapping two jobs, or moving a job), 
        storing the result as an alternative schedule (without modifying the original).
        Make a random perturbation if none is specified
        '''
        # pick a perturbation randomly if none given
        if perturbation is None:
            options = self.perturb_options()
            perturbation = options[np.random.choice(len(options))]
    
        # move one job
        if perturbation[0] == MOVE:
            _, orig_room, orig_index, dest_room, dest_index = perturbation
            new_schedule = deepcopy(self.schedule)
            new_schedule[dest_room].insert(dest_index, new_schedule[orig_room].pop(orig_index))

            self.alt_schedule = new_schedule
            self.alt_delays = self.delays

        # swap a pair of jobs
        if perturbation[0] == SWAP:
            _, j1_room, j1_index, j2_room, j2_index = perturbation
            new_schedule = deepcopy(self.schedule)
            
            new_schedule[j1_room].insert(j1_index + 1, new_schedule[j2_room].pop(j2_index))
            new_schedule[j2_room].insert(j2_index, new_schedule[j1_room].pop(j1_index))
                    
            self.alt_schedule = new_schedule
            self.alt_delays = self.delays

        # adjust the delay of a job (for BIM)
        if perturbation[0] == DELAY_CHANGE:
            _, index, amount = perturbation
            new_delays = deepcopy(self.delays)
            new_delays[index] += amount
            assert(new_delays[index] >= 0)            

            self.alt_schedule = self.schedule
            self.alt_delays = new_delays

    def eval_alt_schedule(self, log=False):
        schedule, delays = self.schedule, self.delays
        self.schedule = self.alt_schedule
        self.delays = self.alt_delays
        obj = self.eval_schedule(log=log)
        self.schedule = schedule
        self.delays = delays
        return obj

    def accept_alt_schedule(self):
        '''
        Update the current schedule to be the previously perturbed schedule
        '''
        self.schedule = self.alt_schedule
        self.delays = self.alt_delays


    def get_schedule(self):
        '''
        Return the schedule 
        '''
        return self.schedule, self.delays
    
    def set_schedule(self, schedule, delays):
        '''
        Set the schedule
        '''
        self.schedule = deepcopy(schedule)
        self.delays = deepcopy(delays)

    def set_alt_schedule(self, schedule, delays):
        '''
        Set the alternative schedule
        '''
        self.alt_schedule = deepcopy(schedule)
        self.alt_delays = deepcopy(delays)
    
    def set_obj_weights(self, obj_weights):
        '''
        Modify the objective weights
        '''
        assert len(obj_weights) == 5
        self.obj_weights = obj_weights

    def _job_statistics(self, job, running_time, prev_job_family, job_info):
        '''
        Determine the relevant statistics for a job for use in room statistic calculations
        '''
        gap_start = running_time

        # add changeover time
        if prev_job_family:
            if prev_job_family != job_info['family']:
                running_time += self.clean_t_diff
            else:
                running_time += self.clean_t_same

        # can't start job until it's ready
        if running_time < job_info['arrival']:
            running_time = job_info['arrival']

        # we may have deliberately delayed an elective job to improve break in moments
        if not job_info['emergency']:
            running_time += self.delays[job]

        gap_end = running_time

        # determine weighted wait for an emergency job
        weighted_emergency_time = 0
        if job_info['emergency']:
            wait = running_time - job_info['arrival']
            weighted_emergency_time = 2 ** (wait / 60 / (11 - job_info['priority'])) - 1
        
        running_time += job_info['length']
        return running_time, weighted_emergency_time, job_info['family'], \
            (gap_start, gap_end)
    
    def _room_statistics(self, room, jobs_df: pd.DataFrame):
        '''
        Determine the statistics of a room given a fixed sequence of jobs to be run in that room
        returns makespan, weighted waiting time of emergency after release
        '''
        running_time = 0
        weighted_emergency_time = 0
        prev_job_family = None
        gaps = set()
        for job in room:
            running_time, _weighted_emergency_time, prev_job_family, gap = \
                self._job_statistics(job, running_time, prev_job_family, jobs_df.iloc[job])
            weighted_emergency_time +=_weighted_emergency_time
            gaps.add(gap)
        last_end = running_time

        # final changeover time
        running_time += self.clean_t_same
        return running_time, weighted_emergency_time, gaps, last_end

    def _eval_single_schedule(self, jobs_df, log=False):
        '''
        Evaluate a single schedule, given fixed job information
        '''
        alpha, beta, gamma, delta, epsilon = self.obj_weights
        max_run = 0
        rooms_open = len(self.schedule)
        weighted_emergency_wait = 0
        tardiness = 0
        
        bim = 0 
        gaps = set()
        earliest_finish = None

        # calculate statistics for each room
        for room in self.schedule:
            if not room:
                rooms_open -= 1
                continue

            room_time, room_emergency_time, room_gaps, room_end = \
                self._room_statistics(room, jobs_df)

            # max run, tardiness stats
            if room_time > max_run:
                max_run = room_time
            weighted_emergency_wait += room_emergency_time
            if room_time > ROOM_CLOSE_TIME:
                tardiness += room_time - ROOM_CLOSE_TIME

            # bim stats
            if earliest_finish is None:
                earliest_finish = room_end
            elif room_end < earliest_finish:
                earliest_finish = room_end
            gaps = gaps | room_gaps


        # calculation of BIM
        job_starts = {gap_end for _, gap_end in gaps}
        for start in job_starts:
            in_gap = False
            for gap_start, gap_end in gaps:
                if gap_start <= start < gap_end:
                    in_gap = True
                    break
            break_in = min([gap_start for gap_start, _ in gaps if gap_start > start] + [earliest_finish])
            # if log:
            #     print(round(start), in_gap, round(break_in))
            if in_gap:
                continue
            if break_in - start > bim:
                bim = break_in - start
        if log:
            # print('bim gaps:')
            # for gap_start, gap_end in gaps:
            #     print(round(gap_start), round(gap_end))
            # print('job_starts:')
            # print([round(t) for t in job_starts])
            print('alpha:', alpha, max_run, alpha * max_run)
            print('beta:', beta, rooms_open, beta * rooms_open)
            print('gamma:', gamma, weighted_emergency_wait, gamma * weighted_emergency_wait)
            print('delta:', delta, tardiness, delta * tardiness)
            print('epsilon:', epsilon, bim, epsilon * bim)
            print()

        return alpha*max_run + beta*rooms_open + gamma*weighted_emergency_wait + \
            delta*tardiness + epsilon*bim

    def _initial_schedule(self, jobs_df, n_rooms, n_electives):
        initial_schedule = [[] for _ in range(n_rooms)]
        families = list(set(jobs_df['family']))
        
        for j, row in jobs_df.iterrows():
            room = families.index(row['family']) % n_rooms
            initial_schedule[room].append(j)
                    
        initial_delays = [0] * n_electives
        return initial_schedule, initial_delays
    
    def get_job_start_times(self, schedule, jobs_df, delays=None, elective_arrivals=None):
        elective_only = elective_arrivals is None
        job_start_times = {}
        for room_jobs in schedule:
            prev_job_info = None
            prev_job_index = None
            for job_index in room_jobs:
                job_info = jobs_df.iloc[job_index]
                if elective_only and job_info['emergency']:
                    continue
                if prev_job_info is not None:
                    cleaning_time = self.clean_t_same if prev_job_info['family'] == job_info['family'] \
                        else self.clean_t_diff
                    job_start_times[job_index] = max([job_info['arrival'],
                        job_start_times[prev_job_index] + prev_job_info['length'] + cleaning_time])
                else:
                    job_start_times[job_index] = job_info['arrival']
                # If only looking at electives, increment start time by delay to get arrival time
                if elective_only:
                    job_start_times[job_index] += delays[job_index]
                # If looking at both, bump elective start time to arrival time if not there already
                elif not job_info['emergency'] and \
                        job_start_times[job_index] < elective_arrivals[job_index]:
                    job_start_times[job_index] = elective_arrivals[job_index]
                prev_job_info = job_info
                prev_job_index = job_index
        return job_start_times
    
    def plot(self, schedule, delays, jobs_df):
        # Run first to determine the effective arrival times of the elective jobs
        elective_job_start_times = self.get_job_start_times(schedule, jobs_df,
            delays=delays)
        # Run again using these arrivals to find the start times of all jobs
        job_start_times = self.get_job_start_times(schedule, jobs_df,
            elective_arrivals=elective_job_start_times)
        # for k, v in job_start_times.items():
        #     print(k, round(v), round(v + jobs_df.iloc[k]['length']))

        figure = plt.figure()
        for room_index, room_jobs in enumerate(schedule):
            # Reverse so that both the legend and room numbers are top-to-bottom
            room_index = len(schedule) - room_index - 1
            for job_index in room_jobs:
                job_info = jobs_df.iloc[job_index]
                start = job_start_times[job_index]
                plt.hlines(room_index + 1, start, start + job_info['length'],
                    color=PLOT_COLOURS[job_info['family']], linewidth=3,
                    label=f'room {room_index + 1}, job {job_index}',
                    linestyle='solid' if job_info['emergency'] else 'dotted')
                if job_info['emergency']:
                    plt.plot(job_info['arrival'], room_index + 1,
                        color=PLOT_COLOURS[job_info['family']], marker='x', markersize=12)
                    plt.annotate(f'j{job_index} (p{job_info["priority"]})',
                        (job_info['arrival'], room_index + 1 - 0.25),
                        color=PLOT_COLOURS[job_info['family']])
                plt.annotate(f'j{job_index}',
                    (start, room_index + 1 + 0.15),
                    color=PLOT_COLOURS[job_info['family']])
        plt.vlines([ROOM_OPEN_TIME, ROOM_CLOSE_TIME], 0, len(schedule) + 1,
            color='gray', linestyle='dotted')
        plt.xlim([300, 1440])
        plt.ylim([0.1, len(schedule) + 0.9])
        axis = plt.gca()
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('minutes into day')
        plt.ylabel('room number')
        # plt.legend()
        plt.show()

class OracleSchedule(Schedule):
    def __init__(self, schedule=[[]], delays=[], clean_t_same=20, clean_t_diff=90,
            n_rooms=5, electives_df=pd.DataFrame(), emerg_df=pd.DataFrame(),
            n_electives=0,obj_weights=[1,1,1,1,0]):
        self.jobs_df = pd.concat([electives_df, emerg_df], ignore_index=True)
        self.n_electives = n_electives

        initial_schedule, initial_delays = deepcopy(schedule), deepcopy(delays)
        if not schedule or len(schedule) == 1 and not schedule[0]:
            initial_schedule, initial_delays = self._initial_schedule(
                self.jobs_df,n_rooms, n_electives)

        super().__init__(initial_schedule, initial_delays, clean_t_same,
            clean_t_diff, n_rooms, obj_weights)
    
    def eval_schedule(self, log=False):
        return self._eval_single_schedule(self.jobs_df, log)
    

class StartDaySchedule(Schedule):
    def __init__(self, schedule=[[]], delays=[], clean_t_same=20, clean_t_diff=90,
            n_rooms=5, bim=False, obj_weights=[1,1,1,1,0]):
        self.bim = bim
        self.end_day_schedule = None
        self.end_day_delays = None
        super().__init__(schedule, delays, clean_t_same, clean_t_diff, n_rooms, obj_weights)

    def perturb_options(self):
        '''
        Give a list of all possible move/swap/delay change perturbations of the schedule
        '''
        bim_options = []

        # add delay shift options
        if self.bim:
            for index, delay in enumerate(self.delays):
                bim_options.append((DELAY_CHANGE, index, 20))
                bim_options.append((DELAY_CHANGE, index, 120))
                if delay >= 10:
                    bim_options.append((DELAY_CHANGE, index, -10))
                if delay >= 30:
                    bim_options.append((DELAY_CHANGE, index, -30))

        return super().perturb_options() + bim_options

    def produce_end_day_schedule(self):
        raise NotImplementedError

    def _insert_emerg(self, jobs_df: pd.DataFrame, emerg, emerg_index, room_sequence: list):
        '''
        Modify a room sequence in place to put an emergency in the earliest possible slot
        '''
        arrival_time = emerg['arrival']
        priority = emerg['priority']
        room_time = ROOM_OPEN_TIME
        prev_job_family = None
        for job in room_sequence:
            cur_job_emerg = jobs_df.iloc[job]['emergency']
            cur_job_priority = jobs_df.iloc[job]['priority']
            # insert job before the current one if we know about the emergency 
            # before the current job starts, and current isn't an already scheduled 
            # emergency with a higher priority
            if arrival_time < room_time and ((not cur_job_emerg) or priority > cur_job_priority):
                room_sequence.insert(room_sequence.index(job), emerg_index)
                return

            room_time, _, prev_job_family, _ = self._job_statistics(job, room_time,
                prev_job_family, jobs_df.iloc[job])

        # schedule emergency at the end of the list if no earlier slot is viable
        room_sequence.append(emerg_index)


    def _produce_single_end_day_schedule(self, electives_df, emerg_df: pd.DataFrame):
        '''
        Produce an end of day schedule by placing emergences into a start of day schedule
        as they arrive. 
        '''
        emerg_df = emerg_df.sort_values(by=['arrival', 'length'], ignore_index=True)
        jobs_df = pd.concat([electives_df, emerg_df], ignore_index=True)

        new_sched = deepcopy(self.schedule)
        n_elect = len(electives_df)
        n_rooms = len(self.schedule)
        
        # set BIM weight to 0 for rescheduling
        obj_weights = deepcopy(self.obj_weights)
        self.obj_weights[-1] = 0

        # simulate the day, scheduling emergencies as they arrive
        for i, (_, emerg) in enumerate(emerg_df.iterrows()):
            # determine best room to place next emergency in to 
            best_room = 0
            best_obj = None
            for room in range(n_rooms):
                self._insert_emerg(jobs_df, emerg, n_elect + i, new_sched[room])
                self.set_alt_schedule(new_sched, self.delays)
                obj = self.eval_alt_schedule()
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_room = room
                
                # take the emergency out to try the next room
                new_sched[room].remove(n_elect + i)

            # put emergency into best room
            self._insert_emerg(jobs_df, emerg, n_elect + i, new_sched[best_room])

        # restore the objective now that rescheduling is complete
        self.obj_weights = obj_weights

        return new_sched, self.delays

    def eval_end_day_schedule(self):
        '''
        A function to evaluate the performance of the end of day schedule/s
        '''
        raise NotImplementedError


class StartDayScheduleDetElectivesDetEmerg(StartDaySchedule):
    def __init__(self, schedule=[[]], delays=[], clean_t_same=20, clean_t_diff=90,
            n_rooms=5, bim=False, electives_df=pd.DataFrame(), emerg_df=pd.DataFrame(),
            n_electives=0, obj_weights=[1,1,1,1,0]):
        self.electives_df = electives_df
        self.emerg_df = emerg_df
        self.jobs_df = pd.concat([electives_df, emerg_df], ignore_index=True)

        initial_schedule, initial_delays = deepcopy(schedule), deepcopy(delays)
        if not schedule or len(schedule) == 1 and not schedule[0]:
            initial_schedule, initial_delays = self._initial_schedule(
                self.electives_df, n_rooms, n_electives)

        super().__init__(initial_schedule, initial_delays, clean_t_same,
            clean_t_diff, n_rooms, bim, obj_weights)
    
    def eval_schedule(self, log=False):
        return self._eval_single_schedule(self.jobs_df,log)
    
    def eval_end_day_schedule(self, log=False):
        if self.end_day_schedule is None or self.end_day_delays is None:
            raise Exception('No end of day schedule has been determined')
        
        # set BIM weight to 0 for evaluation
        obj_weights = deepcopy(self.obj_weights)
        self.obj_weights[-1] = 0

        self.set_alt_schedule(self.end_day_schedule, self.end_day_delays)
        evaluation = self.eval_alt_schedule(log), (self.end_day_schedule, self.end_day_delays)
        
        # restore the objective now that evaluation is complete
        self.obj_weights = obj_weights
        
        return evaluation 

    def produce_end_day_schedule(self):
        schedule, delays = self._produce_single_end_day_schedule(
            self.electives_df,self.emerg_df)
        self.end_day_schedule = schedule
        self.end_day_delays = delays


class StartDayScheduleStochElectives(StartDaySchedule):
    def __init__(self, schedule=[[]], delays=[], clean_t_same=20, clean_t_diff=90, n_rooms=5,
            bim=False, electives_dfs=[pd.DataFrame()], emerg_dfs=[pd.DataFrame()],
            n_electives=0, obj_weights=[1,1,1,1,0]):
        self.electives_dfs = electives_dfs
        self.emerg_dfs = emerg_dfs

        # make sure we have the same number of elective samples as emergency samples
        assert(len(electives_dfs) == len(emerg_dfs))
        self.n_samples = len(electives_dfs)

        initial_schedule, initial_delays = deepcopy(schedule), deepcopy(delays)
        if not schedule or len(schedule) == 1 and not schedule[0]:
            initial_schedule, initial_delays = self._initial_schedule(
                self.electives_dfs[0], n_rooms, n_electives)

        self.end_day_schedule_objects = None

        super().__init__(initial_schedule, initial_delays, clean_t_same, clean_t_diff,
            n_rooms, bim, obj_weights)
    
    def eval_schedule(self, log=False):
        obj = 0
        # if log:
        #     print('Logging one evaluation only')
        for elective_df, emerg_df in zip(self.electives_dfs, self.emerg_dfs):
            obj += self._eval_single_schedule(
                pd.concat([elective_df, emerg_df], ignore_index=True), log)
            # log = False
        return obj/len(self.electives_dfs)

    def produce_end_day_schedule(self):
        schedule_objects = []
        for i in range(self.n_samples):
            single_instance = StartDayScheduleDetElectivesDetEmerg(
                schedule=self.schedule,
                delays=self.delays,
                clean_t_same=self.clean_t_same,
                clean_t_diff=self.clean_t_diff,
                n_rooms=self.n_rooms,
                bim=self.bim,
                electives_df=self.electives_dfs[i],
                emerg_df=self.emerg_dfs[i],
                n_electives=len(self.electives_dfs[i]),
                obj_weights=self.obj_weights,
            )
            single_instance.produce_end_day_schedule()
            schedule_objects.append(single_instance)
        
        self.end_day_schedule_objects = schedule_objects
        
    def eval_end_day_schedule(self, log=False):
        '''
        A function to evaluate the performance of the end of day schedules
        '''
        if self.end_day_schedule_objects is None:
            raise Exception('End of day schedules have not been determined')

        # set BIM weight to 0 for end of day evaluation
        obj_weights = deepcopy(self.obj_weights)
        self.obj_weights[-1] = 0

        # evaluate each of the realisations of the day
        evaluations = []
        schedules = []
        for single_instance in self.end_day_schedule_objects:
            obj, (schedule, delays) = single_instance.eval_end_day_schedule(log)
            evaluations.append(obj)
            schedules.append((schedule, delays))

        # restore the objective now that evaluation is complete
        self.obj_weights = obj_weights
        return sum(evaluations) / len(evaluations), evaluations, schedules
