import numpy as np

class Solution:

    def __init__(self, dim, bound, w, r):
        self.dim = dim
        self.bound = bound
        self.sigma = np.random.uniform(low=0.1,high=1.0)
        self.global_search_step_size = np.random.uniform(self.bound[0],self.bound[1]) / 10
        self.local_search_step_size = float('nan')
        self.solution = np.array([self.bound[0] + np.random.rand() * (self.bound[1] - self.bound[0]) for k in range(self.dim)])
        self.w = w
        self.r = r

    def update(self,trans_type):
        if trans_type == 'leader':
            self.solution = self.solution
        elif trans_type == 'local_search':
            self.solution = self.solution + self.local_search_step_size
        elif trans_type == 'global_search':
            self.solution = self.solution + self.global_search_step_size

        # Handle the bound constraints
        self.solution = np.clip(self.solution, *self.bound)

    def update_control_factors(self,trans_type,gbest):
        if trans_type == 'local_search':
            self.sigma = self.sigma * self.r
            self.local_search_step_size = np.random.normal(0,self.sigma) * self.solution
        elif trans_type == 'global_search':
            self.global_search_step_size =  self.w * self.global_search_step_size  +  np.random.rand() * (gbest - self.solution)

    def random_update(self):
        self.solution = np.random.uniform(low=self.bound[0], high=self.bound[1], size=(self.dim,))




