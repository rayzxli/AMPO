# The Adaptive Multi-Population Optimization (AMPO) Algorithm


## Introduction
This is the implementation of the AMPO [1] in Python 3. The AMPO is a newly proposed multi-population based metaheuristic for  **global continuous optimization**. The algorithm hybridizes yet modifies several useful operations like mutation and memory retention from evolutionary algorithms and swarm intelligence (SI) techniques in a multi-population manner. Furthermore, the diverse control on multiple populations, solution cloning and reset operation are designed. Compared with other metaheuristics, the AMPO can attain an adaptive balance between the capabilities of exploration and exploitation for various optimization problems.

## Usage (An example)

```python
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

```

## Note
The design of the AMPO is partly inspired by virus spread  in nature. Our preliminary work called Virus Spread Optimization (VSO) can be seen in [2][3]. The design ideas of both algorithms are almost same. However, the AMPO is the simplified version of the VSO. Compared with the VSO, the AMPO has fewer user-controlled parameters and a faster speed.

## Publications
[1] Z. Li, V. Tam and L. K. Yeung, "An Adaptive Multi-Population Optimization Algorithm for Global Continuous Optimization," in IEEE Access, vol. 9, pp. 19960-19989, 2021, doi: 10.1109/ACCESS.2021.3054636. [[Download]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9336004) [[Cite]](https://ieeexplore.ieee.org/document/9336004/)


[2] Z. Li, V. Tam and L. K. Yeung, "A study on parameter sensitivity analysis of the virus spread optimization," 2020 IEEE Symposium Series on Computational Intelligence (SSCI), Canberra, Australia, 2020, pp. 1535-1542, doi: 10.1109/SSCI47803.2020.9308167. [[Download]](https://ieeexplore.ieee.org/document/9308167) [[Cite]](https://ieeexplore.ieee.org/document/9308167)

[3] Z. Li & V. Tam, "A novel meta-heuristic optimization algorithm inspired by the spread of viruses," arXiv preprint arXiv:2006.06282, 2020. [[Download]](https://arxiv.org/pdf/2006.06282)



