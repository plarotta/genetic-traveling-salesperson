import wandb
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from utilities import *

def evolutionary_algo(n_generations, data, population_size):
  '''evolutionary approach to solving the traveling salesman problem.
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

  cipher = gen_cipher(data)
  np.random.seed(42)
  run = wandb.init(name='circle 500',)
  half_pop = int(population_size*.33)
  initial_population =  np.array([np.random.permutation(data) for _ in range(population_size)])

  for i in tqdm(range(n_generations)):
    
    offspring = np.zeros((population_size,len(data),2))
    n_offspring = 0
    
    initial_population, current_best, fitnesses = fitness_sort(initial_population)
    best_indiv = initial_population[0]

    initial_population = initial_population[:half_pop]
    fitnesses = fitnesses[:half_pop]
    probs = softmax(-fitnesses)

    run.log({'generation #': i, 
             'generation best': current_best, 
             'diversity': np.mean(np.var(initial_population,axis=0)),
             'most likely': probs[0]})

    seen = set([])
    while n_offspring < population_size:

      parent1, parent2 = pick_n_random_individuals(initial_population, 2, weights = probs)

      # mating
      if np.random.choice([True,False],p=[0.8,0.2]):
        n1, n2 = sorted([np.random.randint(low=0, high=len(data)), np.random.randint(low=0, high=len(data))])
        kid1, kid2 = cross_parents(parent1, parent2, n1, n2, cipher) 

      else:
        kid1 = np.copy(parent1)
        kid2 = np.copy(parent2)

      # mutation
      if np.random.choice([True,False], p=[0.50,0.50]):

        kid1 = flip_two_nodes(kid1)
        kid2 = flip_two_nodes(kid2)

      k1_l = get_path_length(kid1)
      k2_l = get_path_length(kid2)
      if k1_l not in seen:
        offspring[n_offspring,:,:] = kid1
        n_offspring += 1
        seen.add(k1_l)

      if k2_l not in seen and n_offspring < population_size:
        offspring[n_offspring,:,:] = kid2
        n_offspring += 1
        seen.add(k2_l)

    initial_population = np.copy(offspring)
  run.finish()
  return(current_best, best_indiv)

def plot_data(tour):
    plt.plot([i[0] for i in tour], [i[1] for i in tour])
    plt.show()

if __name__ == '__main__':
    data = np.loadtxt('data/circle500.txt',delimiter=' ')
    best_l, best_indiv = evolutionary_algo(300000,data, 24)
    print(best_l)
    plot_data(best_indiv)




    