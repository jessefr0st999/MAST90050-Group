import numpy as np
from tqdm import tqdm

from schedules import Schedule

class SimulatedAnnealing():
    def __init__(self, schedule: Schedule, t_start=800, t_min=10, t_factor=0.9995):
        self.schedule = schedule
        self.t_start = t_start
        self.t_min = t_min
        self.t_factor = t_factor

    def sa(self):
        best_obj = self.schedule.eval_schedule()
        best_schedule = self.schedule.get_schedule()
        t = self.t_start

        n_iter = np.ceil(np.emath.logn(self.t_factor, self.t_min/self.t_start))
        with tqdm(total=n_iter) as p_bar:
            obj = self.schedule.eval_schedule()
            while t > self.t_min:                
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
                t *= self.t_factor
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
