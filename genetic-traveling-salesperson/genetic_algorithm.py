import wandb
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from utilities import *

def evolutionary_algo(n_generations: int, 
                      data: np.array, 
                      population_size: int,
                      selection_fraction: float,
                      mating_probability: float,
                      mutation_probability: float,
                      wandb=True
                      ) -> tuple[float,np.array]:
  '''evolutionary approach to solving the traveling salesman problem.
  Mutation operator: flip_two_nodes()
  Breeding/crossing operator: cross_parents() [2-point crossover]
  Selection: fitness-proportionate 
  
  Args:
    n_generations:
      number of generations to run the algortithm for
    data:
      array of the input dataset of cities. each row holds the x and y 
      coordinate of its city. Array should be of shape (N,2) where N is
      the number of cities in the tour.
    population_size:
      number of individuals to generate at each generation

  Returns:
    tuple containing the shortest path length and the shortest path found
  '''

  cipher = gen_cipher(data)
  np.random.seed(42)
  run = wandb.init(name='circle 500') if wandb else None

  # initiate population
  half_pop = int(population_size*selection_fraction)
  initial_population =  np.array([np.random.permutation(data) for _ in range(population_size)])


  for i in tqdm(range(n_generations)):
    
    offspring = np.zeros((population_size,len(data),2))
    n_offspring = 0
    
    # selection
    initial_population, current_best, fitnesses = fitness_sort(initial_population)
    best_indiv = initial_population[0]
    initial_population = initial_population[:half_pop]
    fitnesses = fitnesses[:half_pop]
    probs = softmax(-fitnesses) 

    if run:
      run.log({'generation #': i, 
              'generation best': current_best, 
              'diversity': np.mean(np.var(initial_population,axis=0)),
              'most likely': probs[0]})
    else:
      print(f'generation #: {i}, generation best: {current_best}, diversity: {np.mean(np.var(initial_population,axis=0))}, most likely: {probs[0]}')

    # lookup table to make sure we dont fill up the population with the same individual
    seen = set([])

    while n_offspring < population_size:

      parent1, parent2 = pick_n_random_individuals(initial_population, 2, weights = probs)

      # crossover event
      if np.random.choice([True,False],p=[mating_probability, 1-mating_probability]):
        n1, n2 = sorted([np.random.randint(low=0, high=len(data)), 
                         np.random.randint(low=0, high=len(data))])
        kid1, kid2 = cross_parents(parent1, parent2, n1, n2, cipher) 

      else:
        kid1 = np.copy(parent1)
        kid2 = np.copy(parent2)

      # mutation event
      if np.random.choice([True,False], p=[mutation_probability,1-mutation_probability]):

        kid1 = flip_two_nodes(kid1)
        kid2 = flip_two_nodes(kid2)

      # only add the kids to the offspring population if they arent already there
      for kid in [kid1, kid2]:
        k_l = get_path_length(kid)
        if k_l not in seen and n_offspring < population_size:
          offspring[n_offspring,:,:] = kid
          n_offspring += 1
          seen.add(k_l)

    # offspring become the starting population for next generation
    initial_population = np.copy(offspring)

  run.finish()
  return(current_best, best_indiv)

def plot_data(tour):
    '''little function for visualizing tours'''
    plt.plot([i[0] for i in tour], [i[1] for i in tour])
    plt.show()

if __name__ == '__main__':
    data = np.loadtxt('data/circle500.txt',delimiter=' ')
    best_l, best_indiv = evolutionary_algo(300000,data, 24)
    print(best_l)
    plot_data(best_indiv)




    