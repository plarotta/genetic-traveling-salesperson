import matplotlib.pyplot as plt
import numpy as np
import random
import secrets
from scipy.special import softmax
from numba import njit
from utils import fitness_sort, pick_n_random_individuals, cross_parents, flip_two_nodes, get_path_length
# from sklearn.preprocessing import normalize
np.random.seed(0)


def evolutionary_algo(n_generations, data, population_size):
  '''evolutionary algo for solving the traveling salesman problem.
  Mutation operator: flip_two_nodes()
  Breeding/crossing operator: cross_parents()
  Selection: fitness-proportionate
  
  Args:
    n_generations:
      number of generations to run the algortithm for
    data:
      array (1000,2) of the input dataset of cities. each
      row holds the x and y coordinate of its city

  Returns:
    fitness sorted population, shortest path length in population,
    and the sorted array of all fitnesses in population
  '''
  half_pop = int(population_size/2)
  initial_population = np.array([np.random.permutation(data) for _ in range(population_size)])

  for i in range(n_generations):
    
    offspring = np.zeros((population_size,1000,2))
    n_offspring = 0
    
    initial_population, current_best, fitnesses = fitness_sort(initial_population)
    initial_population = initial_population[:half_pop]
    fitnesses = fitnesses[:half_pop]
    probs = softmax(-fitnesses)#1/fitnesses*1/np.sum(1/fitnesses)
    # print(probs)
    # print(list(zip(probs,fitnesses)))
    
    while n_offspring < population_size:

      
      parent1, parent2 = pick_n_random_individuals(initial_population,2, weights = probs)
      if np.random.choice([True,False],p=[0.7,0.3]):

        n1, n2 = sorted([secrets.randbelow(1000), secrets.randbelow(1000)])

        kid1, kid2 = cross_parents(parent1, parent2, n1, n2) 

        mutated_kid1 = flip_two_nodes(kid1)
        kid1 = np.copy(mutated_kid1) if get_path_length(mutated_kid1) < get_path_length(kid1) else kid1
        
        # kid1 = kid1 if get_path_length(kid1) < p1_fit else np.copy(parent1)
            
        mutated_kid2 = flip_two_nodes(kid2)
        kid2 = np.copy(mutated_kid2) if get_path_length(mutated_kid2) < get_path_length(kid2) else kid2
        # kid2 = kid2 if get_path_length(kid2) < p2_fit else np.copy(parent2)
        # offspring[n_offspring,:,:] = kid1
        # offspring[n_offspring + 1,:,:] = kid2
        # n_offspring += 2
      
      else:
        kid1 = np.copy(parent1)
        kid2 = np.copy(parent2)
      
      offspring[n_offspring,:,:] = kid1
      offspring[n_offspring + 1,:,:] = kid2
      n_offspring += 2

      # offspring[n_offspring+2,:,:] = parent1

      # offspring[n_offspring+3,:,:] = parent2
      # n_offspring += 4


    
    print(f'generation #: {i}, generation best: {current_best}, diversity: {np.mean(np.var(initial_population,axis=0))}, most likely: {probs[0]}')
    initial_population = np.copy(offspring)# np.concatenate((initial_population, offspring)) #np.copy(offspring)#
  return(current_best)

if __name__ == '__main__':
    data = np.loadtxt('data/tsp.txt',delimiter=',')
    print(evolutionary_algo(2500,data, 48))



    