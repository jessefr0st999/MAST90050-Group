import numpy as np
import pandas as pd

from scipy.stats import expon

BIG_M = 1e10
ROOM_OPEN_TIME = 8*60
ROOM_CLOSE_TIME = 17*60


JOB_FAMILIES = ['Cardiothoracic', 'Orthopaedic', 'Plastic', 'Hepatobilary']
EMERG_JOB_FAMILIES = ['Cardiothoracic', 'Orthopaedic']

family_samples = {
    'Cardiothoracic': [80, 91, 107],
    'Orthopaedic': [151, 39, 150, 91, 112, 58, 40, 58],
    'Plastic': [44, 44, 94, 52, 55, 88, 84, 71],
    'Hepatobilary': [213, 32, 244],
}

family_exp_locs = {f: expon.fit(family_samples[f])[0] for f in JOB_FAMILIES}
family_exp_scales = {f: expon.fit(family_samples[f])[1] for f in JOB_FAMILIES}
family_expected_lengths = {f: family_exp_locs[f] + family_exp_scales[f] for f in JOB_FAMILIES}

default_elective_jobs = [[length, ROOM_OPEN_TIME, 0, family, False]
        for length, family in [
    [44, 'Plastic'],
    [40, 'Plastic'],
    [94, 'Plastic'],
    [52, 'Plastic'],
    [151, 'Orthopaedic'],
    [80, 'Cardiothoracic'],
    [39, 'Orthopaedic'],
    [213, 'Hepatobilary'],
    [32, 'Hepatobilary'],
    [150, 'Orthopaedic'],
]]
default_emergency_jobs = [
    [400, 7*60 + 50, 1, 'Cardiothoracic', True],
    [30, 8*60 + 0, 4, 'Cardiothoracic', True],
    [30, 8*60 + 11, 5, 'Cardiothoracic', True],
    [60, 8*60 + 40, 5, 'Cardiothoracic', True],
    [60, 6*60 + 5, 5, 'Cardiothoracic', True],
    [40, 13*60 + 20, 5, 'Cardiothoracic', True],
    [40, 12*60 + 12, 5, 'Cardiothoracic', True],
    [30, 9*60 + 25, 5, 'Cardiothoracic', True],
    [15, 8*60 + 12, 5, 'Cardiothoracic', True],
    [120, 9*60 + 50, 8, 'Cardiothoracic', True],
]

# def min_to_time(total_min):
#     hour = int(total_min // 60)
#     minute = int(total_min % 60)
#     minute_str = str(minute) if minute >= 10 else f'0{minute}'
#     return f'{hour}:{minute_str}'

def list_to_df(job_list, sort=False):
    '''
    Format job list as a DataFrame
    '''
    df = pd.DataFrame(job_list, columns=['length', 'arrival', 'priority',
        'family', 'emergency'])
    if sort:
        df = df.sort_values(by=['arrival', 'length']).reset_index()
        df = df.rename(columns={'index': 'orig_index'})
    return df

def det_electives(families):
    '''
    Return sample mean length jobs given input families 
    '''
    return list_to_df([[family_expected_lengths[family], ROOM_OPEN_TIME, 0, family, False] for family in families])

class DetElectivesDetEmergencies():
    def __init__(self, electives, emergencies):
        self.electives = electives
        self.emergencies = emergencies
        self.electives_df = list_to_df(electives)
        self.emerg_df = list_to_df(emergencies, sort=True)
        self.n_electives = len(electives)

    def get_jobs(self):
        return self.n_electives, self.electives_df, self.emerg_df


class StochElectivesStochEmergencies():
    def __init__(self, elective_families, n_emerg=None, n_emerg_lambda=5, priority_sd=2):
        self.elective_dfs = None
        self.emerg_dfs = None
        self.elective_families = elective_families
        self.n_electives = len(elective_families)
        self.n_emerg = n_emerg
        self.n_emerg_lambda = n_emerg_lambda
        self.priority_sd = priority_sd
        self.deterministic_electives = det_electives(elective_families)

    def generate_samples(self, n_samples):
        elective_dfs = []
        emerg_dfs = []

        for _ in range(n_samples):
            # sample electives
            electives_sample = [[
                expon.rvs(loc=family_exp_locs[family], scale=family_exp_scales[family]), 
                        ROOM_OPEN_TIME, 0, family, False] 
                    for family in self.elective_families]

            elective_dfs.append(list_to_df(electives_sample))

            # generate emergencies 
            emergs_sample = []
            n_emerg = self.n_emerg
            if n_emerg is None:
                n_emerg = np.random.poisson(self.n_emerg_lambda)
            for _ in range(n_emerg):
                family = np.random.choice(EMERG_JOB_FAMILIES)
                length = expon.rvs(loc=family_exp_locs[family],
                    scale=family_exp_scales[family])
                arrival = int(6*60 + 12*60 * np.random.random())
                priority = int(min(max(np.random.normal(5, self.priority_sd), 0), 10))
                emergs_sample.append([length, arrival, priority, family, True])

            emerg_dfs.append(list_to_df(emergs_sample, sort=True))
        self.elective_dfs = elective_dfs
        self.emerg_dfs = emerg_dfs


    def get_jobs(self):
        '''
        Get job samples
        '''
        return self.n_electives, self.elective_dfs, self.emerg_dfs

    def get_deterministic_electives(self) -> pd.DataFrame:
        '''
        Return the electives as jobs with expectation as duration
        '''
        return self.deterministic_electives
