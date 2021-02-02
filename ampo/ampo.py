import numpy as np
from .individual import Individual
from .solution import Solution

class AMPO:

    def __init__(self, func, dim, bound, max_iters, pop=50,
            p_ld_ls=0.8, p_ls_ls=0.8, pr=0.6, w=0.1, r=0.9, show_info=False):

        # Problem information
        self.func = func
        self.dim = dim
        self.bound = bound

        # User-controlled parameters of AMPO
        self.pop = pop
        self.iterations = max_iters
        self.p_ld_ls = p_ld_ls
        self.p_ls_ls = p_ls_ls
        self.pr = pr
        self.w = w
        self.r = r

        self.show_info = show_info

        self.history = []
        self.gbest = {'individual': None, 'fitness': float('inf'), 'solution': None}
        self.migration_gbest = {'individual': None, 'fitness': float('inf'), 'solution': None}

    def run(self):
        individuals = []
        migrating_individuals = []
        trans_probs = {
                'leader': {'local_search': self.p_ld_ls, 'global_search': 1.0 - self.p_ld_ls},
                'local_search': {'local_search': self.p_ls_ls,'global_search': 1.0 - self.p_ls_ls},
                'global_search': {'local_search': 0.0, 'global_search':1.0}
        }
        self.main_pop_size = int(self.pop * self.pr)
        self.migrating_group_pop_size = self.pop - self.main_pop_size

        ################# Initialization Operation #################
        # Initialize individuals
        for _ in range(self.main_pop_size):
            individuals.append(Individual(self.dim, self.bound, self.w, self.r))

        # This is just to initialize the solutions for the migration group
        for _ in range(self.migrating_group_pop_size):
            _individual = Individual(self.dim, self.bound, self.w, self.r)
            migrating_individuals.append(_individual)
        self.migration_solutions = np.array([_individual.solution.solution for _individual in migrating_individuals])

        for iteration in range(self.iterations):
            pbest = {'individual':None, 'fitness': float('inf'), 'solution': None}

            ################# Function Evaluation #################
            for idx,individual in enumerate(individuals):
                solution = individual.solution.solution
                individual.fitness = self.func(solution)
                if individual.fitness < pbest['fitness']:
                    pbest['fitness'] = individual.fitness
                    pbest['solution'] = individual.solution.solution
                    pbest['individual'] = individual

            ################# Selection Operation #################
            if pbest['individual'].fitness < self.gbest['fitness']:
                existing_leader_individual = [h for h in individuals if h.type == 'leader']
                if len(existing_leader_individual) > 0:
                        existing_leader_individual[0].type = 'local_search'
                self.gbest['fitness'] = pbest['fitness']
                self.gbest['solution'] = pbest['solution']
                self.gbest['individual'] = pbest['individual']
                self.gbest['individual'].source_ind = True
                self.gbest['individual'].type = 'leader'

            ################# Transformation Operation #################
            # Sorting: big fitness-> small fitness
            individuals = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)

            source_individuals = [individual for individual in individuals if individual.source_ind]
            random_individuals = [individual for individual in individuals if not individual.source_ind]

            for source_individual in source_individuals[::-1]:
                 trans_type = source_individual.type
                 if len(random_individuals) >= 1:
                     contacted_individuals = random_individuals[0:1]
                     for healthy_individual in contacted_individuals:
                            type_prob = np.random.rand()
                            if type_prob <= trans_probs[trans_type]['global_search']:
                                healthy_individual.transform(source_individual, 'global_search')
                            else:
                                healthy_individual.transform(source_individual, 'local_search')
                            random_individuals.remove(healthy_individual)

            ################# Migration Operation #################
            self.migrating_mutation(self.migrating_group_pop_size)
            self.migrating_crossover(self.migrating_group_pop_size)
            self.migrating_selection()
            if np.random.rand() <= 0.5 * iteration/self.iterations:
                if self.migration_gbest['fitness'] < self.gbest['fitness']:
                   self.gbest['individual'].solution.solution =  self.migration_gbest['solution']
                   self.gbest['individual'].fitness = self.migration_gbest['fitness']
                   self.gbest['solution'] =  self.migration_gbest['solution']
                   self.gbest['fitness'] =  self.gbest['individual'].fitness

            ################# Update Operation #################
            for idx,individual in enumerate(individuals):
                individual.update(self.gbest['solution'], iteration)

            if self.show_info:
                print('Iteration:' + str(iteration) )
                print(str(self.gbest['fitness'])+ '\n')
            self.history.append({'iteration': iteration, 'fitness': self.gbest['fitness']})

            ################# Recovery Operation #################
            if len(source_individuals) == int(self.main_pop_size):
                rev_percent = np.random.uniform(low = 0.1, high = 0.9)
                for individual in source_individuals[0 : int(len(source_individuals) * rev_percent)]:
                    individual.recover()

        return self.gbest['solution'], self.gbest['fitness'], self.history


    ################# Search by the Migrating Group based on the DE/1/rand algorithm  #################
    def migrating_x_to_y(self):
        self.Y_raw = []
        for k in range(len(self.migration_solutions)):
            self.Y_raw.append(self.func(self.migration_solutions[k]))
        self.Y = np.array(self.Y_raw)
        return self.Y

    def migrating_mutation(self,size_pop):
        X = self.migration_solutions
        random_idx = np.random.randint(0, size_pop, size = (size_pop, 3))
        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        self.V = X[r1, :] + 0.5 * (X[r2, :] - X[r3, :])
        mask = np.random.uniform(low=self.bound[0], high=self.bound[1], size=(size_pop, self.dim))
        self.V = np.where(self.V < self.bound[0], mask, self.V)
        self.V = np.where(self.V > self.bound[1], mask, self.V)
        return self.V

    def migrating_crossover(self,size_pop):
        mask = np.random.rand(size_pop, self.dim) < 0.3
        self.U = np.where(mask, self.V, self.migration_solutions)
        return self.U

    def migrating_selection(self):
        X = self.migration_solutions.copy()
        f_X = self.migrating_x_to_y().copy()
        self.migration_gbest['fitness'] = f_X.min()
        self.migration_gbest['solution'] = X[self.Y.argmin()]
        self.migration_solutions = U = self.U
        f_U = self.migrating_x_to_y()
        self.migration_solutions = np.where((f_X < f_U).reshape(-1, 1), X, U)
        return self.migration_solutions
