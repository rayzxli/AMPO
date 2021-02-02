import numpy as np
from .solution import Solution

class Individual:

    def __init__(self, problem_dim, problem_bound, w, r):
        self.problem_bound = problem_bound
        self.problem_dim = problem_dim
        self.source_ind = False
        self.w = w
        self.r = r
        self.solution = Solution(self.problem_dim, self.problem_bound, self.w, self.r)
        self.fitness = None
        self.type = 'random_search'


    def recover(self):
        trans_type = self.type
        self.__init__(self.problem_dim, self.problem_bound,self.w,self.r)
        if trans_type == 'local_search':
           self.type = 'global_search'
           self.source_ind = True
        elif trans_type == 'global_search':
           self.source_ind = False
           self.type = 'random_search'


    def transform(self,source_individual,trans_type):
        if not self.source_ind:
            self.source_ind = True
            self.type = trans_type
            if trans_type == 'local_search':
                self.solution.solution = source_individual.solution.solution
            else:
                cross_points = np.random.randint(0, 2, size = self.problem_dim).astype(np.bool)
                for idx, point in enumerate(cross_points):
                    if point:
                        self.solution.solution[idx] = source_individual.solution.solution[idx]


    def update(self,gbest,iteration):
        if self.source_ind:
           self.solution.update_control_factors(self.type,gbest)
           self.solution.update(self.type)
        else:
           self.solution.random_update()



