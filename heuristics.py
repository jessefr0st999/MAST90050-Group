import numpy as np
from tqdm import tqdm
from datetime import datetime

from schedules import Schedule

class SimulatedAnnealing():
    def __init__(self, schedule: Schedule):
        self.schedule = schedule

    def sa(self, t_start=800, t_min=10, t_factor=0.9995):
        best_obj = self.schedule.eval_schedule()
        best_schedule = self.schedule.get_schedule()
        t = t_start

        n_iter = np.ceil(np.emath.logn(t_factor, t_min/t_start))
        with tqdm(total=n_iter) as p_bar:
            obj = self.schedule.eval_schedule()
            while t > t_min:                
                self.schedule.perturb_schedule()
                new_obj = self.schedule.eval_alt_schedule()
                if new_obj < best_obj:
                    best_obj = new_obj
                    self.schedule.accept_alt_schedule()
                    best_schedule = self.schedule.get_schedule()
                    obj = new_obj
                elif new_obj < obj or np.random.uniform() < np.exp((obj - new_obj)/t):
                    self.schedule.accept_alt_schedule()
                    obj = new_obj
                t *= t_factor
                p_bar.update(1)
        
        self.schedule.set_schedule(*best_schedule)

        return best_schedule, best_obj


class LocalSearch():
    def __init__(self, schedule: Schedule) -> None:
        self.schedule = schedule

    def local_search(self):
        improvement = True
        best_obj = self.schedule.eval_schedule()
        while improvement:
            best_improvement = 0
            best_option = None
            best_new_obj = 0

            options = self.schedule.perturb_options()
            for option in options:
                self.schedule.perturb_schedule(option)
                obj = self.schedule.eval_alt_schedule()
                if best_obj - obj > best_improvement:
                    best_option = option
                    best_new_obj = obj
            
            if best_option is None:
                break

            self.schedule.perturb_schedule(best_option)
            self.schedule.accept_alt_schedule()
            best_obj = best_new_obj
        
        return self.schedule.get_schedule(), best_obj


def heuristic_optimise(schedule: Schedule, n_parallel=3, log=False):
    '''
    Run a mix of simulated annealing and local search from the initial solution 
    n_parallel times, taking the best solution eventually found
    '''
    initial_schedule, initial_delays = schedule.get_schedule()

    best_obj = None
    best_schedule, best_delays = None, None

    sa = SimulatedAnnealing(schedule)
    ls = LocalSearch(schedule)

    start_total = datetime.now()
    for i in range(n_parallel):
        start = datetime.now()
        if log:
            print(f'Iteration {i + 1}')
            print(f'Initial: {schedule.eval_schedule()}')

        sa.sa(t_start=800, t_min=400)
        sa.sa(t_start=800, t_min=400)
        if log:
            print(f'First SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'First LS: {schedule.eval_schedule()}')
        sa.sa(t_start=20, t_min=1)
        sa.sa(t_start=10, t_min=1)
        if log:
            print(f'Second SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'Second LS: {schedule.eval_schedule()}')
        sa.sa(t_start=5, t_min=1)
        sa.sa(t_start=5, t_min=1)
        sa.sa(t_start=5, t_min=1)
        if log:
            print(f'Third SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'Third LS: {schedule.eval_schedule()}')

        print(f'Iteration {i + 1} seconds elapsed: {(datetime.now() - start).seconds}')

        obj = schedule.eval_schedule()
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_schedule, best_delays = schedule.get_schedule()

        schedule.set_schedule(initial_schedule, initial_delays) 

    print(f'Total seconds elapsed: {(datetime.now() - start_total).seconds}')
    schedule.set_schedule(best_schedule, best_delays)


def heuristic_optimise_alt1(schedule: Schedule, n_parallel=3, log=False):
    '''
    Alternative heuristic optimise. One SA and one LS
    '''
    initial_schedule, initial_delays = schedule.get_schedule()

    best_obj = None
    best_schedule, best_delays = None, None

    sa = SimulatedAnnealing(schedule)
    ls = LocalSearch(schedule)

    start_total = datetime.now()
    for i in range(n_parallel):
        start = datetime.now()
        if log:
            print(f'Iteration {i + 1}')
            print(f'Initial: {schedule.eval_schedule()}')

        sa.sa(t_start=800, t_min=1)
        if log:
            print(f'First SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'First LS: {schedule.eval_schedule()}')

        print(f'Iteration {i + 1} seconds elapsed: {(datetime.now() - start).seconds}')

        obj = schedule.eval_schedule()
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_schedule, best_delays = schedule.get_schedule()

        schedule.set_schedule(initial_schedule, initial_delays) 

    print(f'Total seconds elapsed: {(datetime.now() - start_total).seconds}')
    schedule.set_schedule(best_schedule, best_delays)

def heuristic_optimise_alt2(schedule: Schedule, n_parallel=3, log=False):
    '''
    Alternative heuristic optimise. High temp SA
    '''
    initial_schedule, initial_delays = schedule.get_schedule()

    best_obj = None
    best_schedule, best_delays = None, None

    sa = SimulatedAnnealing(schedule)
    ls = LocalSearch(schedule)

    start_total = datetime.now()
    for i in range(n_parallel):
        start = datetime.now()
        if log:
            print(f'Iteration {i + 1}')
            print(f'Initial: {schedule.eval_schedule()}')

        sa.sa(t_start=800, t_min=400)
        sa.sa(t_start=800, t_min=400)
        if log:
            print(f'First SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'First LS: {schedule.eval_schedule()}')
        sa.sa(t_start=800, t_min=200)
        sa.sa(t_start=800, t_min=200)
        if log:
            print(f'Second SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'Second LS: {schedule.eval_schedule()}')
        sa.sa(t_start=800, t_min=1)
        sa.sa(t_start=800, t_min=1)
        if log:
            print(f'Third SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'Third LS: {schedule.eval_schedule()}')

        print(f'Iteration {i + 1} seconds elapsed: {(datetime.now() - start).seconds}')

        obj = schedule.eval_schedule()
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_schedule, best_delays = schedule.get_schedule()

        schedule.set_schedule(initial_schedule, initial_delays) 

    print(f'Total seconds elapsed: {(datetime.now() - start_total).seconds}')
    schedule.set_schedule(best_schedule, best_delays)

def heuristic_optimise_alt3(schedule: Schedule, n_parallel=3, log=False):
    '''
    Alternative heuristic optimise. Low temp SA
    '''
    initial_schedule, initial_delays = schedule.get_schedule()

    best_obj = None
    best_schedule, best_delays = None, None

    sa = SimulatedAnnealing(schedule)
    ls = LocalSearch(schedule)

    start_total = datetime.now()
    for i in range(n_parallel):
        start = datetime.now()
        if log:
            print(f'Iteration {i + 1}')
            print(f'Initial: {schedule.eval_schedule()}')

        sa.sa(t_start=50, t_min=1)
        sa.sa(t_start=50, t_min=1)
        if log:
            print(f'First SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'First LS: {schedule.eval_schedule()}')
        sa.sa(t_start=20, t_min=1)
        sa.sa(t_start=20, t_min=1)
        if log:
            print(f'Second SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'Second LS: {schedule.eval_schedule()}')
        sa.sa(t_start=5, t_min=1)
        sa.sa(t_start=5, t_min=1)
        if log:
            print(f'Third SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'Third LS: {schedule.eval_schedule()}')

        print(f'Iteration {i + 1} seconds elapsed: {(datetime.now() - start).seconds}')

        obj = schedule.eval_schedule()
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_schedule, best_delays = schedule.get_schedule()

        schedule.set_schedule(initial_schedule, initial_delays) 

    print(f'Total seconds elapsed: {(datetime.now() - start_total).seconds}')
    schedule.set_schedule(best_schedule, best_delays)

def heuristic_optimise_alt4(schedule: Schedule, n_parallel=3, log=False):
    '''
    Alternative heuristic optimise. Only final LS
    '''
    initial_schedule, initial_delays = schedule.get_schedule()

    best_obj = None
    best_schedule, best_delays = None, None

    sa = SimulatedAnnealing(schedule)
    ls = LocalSearch(schedule)

    start_total = datetime.now()
    for i in range(n_parallel):
        start = datetime.now()
        if log:
            print(f'Iteration {i + 1}')
            print(f'Initial: {schedule.eval_schedule()}')

        sa.sa(t_start=800, t_min=400)
        sa.sa(t_start=800, t_min=400)
        if log:
            print(f'First SA: {schedule.eval_schedule()}')

        sa.sa(t_start=20, t_min=1)
        sa.sa(t_start=10, t_min=1)
        if log:
            print(f'Second SA: {schedule.eval_schedule()}')
        sa.sa(t_start=5, t_min=1)
        sa.sa(t_start=5, t_min=1)
        sa.sa(t_start=5, t_min=1)
        if log:
            print(f'Third SA: {schedule.eval_schedule()}')
        ls.local_search()
        if log:
            print(f'Final/only LS: {schedule.eval_schedule()}')

        print(f'Iteration {i + 1} seconds elapsed: {(datetime.now() - start).seconds}')

        obj = schedule.eval_schedule()
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_schedule, best_delays = schedule.get_schedule()

        schedule.set_schedule(initial_schedule, initial_delays) 

    print(f'Total seconds elapsed: {(datetime.now() - start_total).seconds}')
    schedule.set_schedule(best_schedule, best_delays)