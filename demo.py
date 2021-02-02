from ampo.ampo import AMPO
import numpy as np

# Objective Function: a minimization optimization problem
def functiton(variables):
    return np.sum(variables**2)


if __name__ == '__main__':

    '''
    bound: the bound of the problem
    pop: population size. default value: 50.
    max_iters: maximum iterations
    p_ld_ls: the probability of a random search group individual being transformed into the local search (ls) group by the leader (ld) group individual. default value: 0.8.
    p_ls_ls: the probability of a random search group individual being transformed into the local search (ls) group by a local search (ls) group individual. default value: 0.8.
    pr: partition rate of the main pop (random search, local search, global search and leader groups) and the migrating group. default value: 0.6.
    w: the omega parameter used in the solution update of the global search group individuals. default value: 0.1.
    r: the gamma parameter used in the update of the local search group individuals. default value: 0.9.
    show_info: to print search information at each iteration
    '''

    algo = AMPO(func = functiton, dim=2, bound=[-100,100], pop=50, max_iters=1000, p_ld_ls=0.8, p_ls_ls=0.8, pr=0.6, w=0.1, r=0.9, show_info=False)
    best_solution, best_fitness, history = algo.run()
    print('Best solution:' + str(best_solution))
    print('Best fitness:' + str(best_fitness))

