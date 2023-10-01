import matplotlib.pyplot as plt
import numpy as np
import random
import secrets
from scipy.special import softmax

def get_path_length(path: np.array) -> float:
    '''calculates the distance covered by the path'''
    tot = np.sum([
      np.linalg.norm(np.array(path[i+1])-np.array(path[i]))
      for i in range(len(path)-1)
      ])
    return(np.round(tot,3))

def randomly_find_path(iterations: int, data: np.array) ->list:
    '''randomly generates paths and returns the distance of the shortest one'''
    path_lengths = []
    best_dist = get_path_length(np.random.permutation(data))
    for i in range(iterations-1):
      new_path_length = get_path_length(np.random.permutation(data))
      if new_path_length < best_dist:
        path_lengths.append(new_path_length)
        best_dist = new_path_length
    return(path_lengths)

def flip_two_nodes(data: np.array) -> np.array:
  '''randomly swaps two cities in the path and returns the modified path'''
  modified_path = np.copy(data)
  indices = [secrets.randbelow(1000), secrets.randbelow(1000)]
  modified_path[indices[0],] = data[indices[1],]
  modified_path[indices[1],] = data[indices[0],]
  return(modified_path)


def hillclimber_find_path(max_depth: int, data: np.array) -> float:
  '''simple random-mutation hillclimber to find the shortest path 
  between the cities. the random mutation just randomly switches
  the order of 2 cities in the path'''
  tries = 0
  current_path = np.copy(data)
  current_path_length = get_path_length(current_path)
  while tries < max_depth:
    next_path = flip_two_nodes(current_path)
    next_path_length = get_path_length(next_path)
    tries += 1
    if next_path_length < current_path_length:
      current_path = next_path
      current_path_length = next_path_length
  return(get_path_length(current_path))


def generate_segment(segment: np.array, reference: np.array) -> np.array:
  '''regenerate segment in the order that it appears in reference'''
  inds = segment == reference[:, None]
  row_sums = inds.sum(axis = 2)
  i, j = np.where(row_sums == 2) 
  pos = np.ones(segment.shape[0], dtype = 'int64') * -1
  pos[j] = i
  return(reference[pos])

def cross_parents(parent1: np.array, 
                  parent2: np.array, 
                  crosspoint1: int, 
                  crosspoint2: int
                  ) -> tuple:
  
  segment_A = generate_segment(parent2[:crosspoint1,], parent1)
  segment_B = generate_segment(parent1[crosspoint1:crosspoint2,], parent2)
  segment_C = generate_segment(parent2[crosspoint2:,], parent1)

  kid1 = np.concatenate((parent1[:crosspoint1], segment_B, parent1[crosspoint2:]))
  kid2 = np.concatenate((segment_A, parent2[crosspoint1:crosspoint2], segment_C))
  
  return(kid1, kid2)

def pick_n_random_individuals(population: np.array, n: int, weights: list) -> list:
  '''TODO'''
  if weights is not None:
    out_idx = np.random.choice([i for i in range(len(population))], n, replace = False, p = weights)
    output_individuals = [population[out_idx[0],], population[out_idx[1],]]
  else: 
    output_individuals = [population[secrets.randbelow(len(population))] for _ in range(n)]
  # print(output_individuals)
  # input()
  return(output_individuals)

def fitness_sort(population: np.array, return_vals = False) -> np.array:
  '''TODO'''
  fit_vals = np.array([get_path_length(p) for p in population])
  if return_vals:
    return(population[fit_vals.argsort()[::]], min(fit_vals), np.sort(fit_vals))
  else:
    return(population[fit_vals.argsort()[::]], min(fit_vals))

def evolutionary_algo(n_generations, data):
  '''TODO: docstring'''
  initial_population = np.array([np.random.permutation(data) for _ in range(50)])

  for i in range(n_generations):
    offspring = np.zeros((26,1000,2))
    n_offspring = 0
    
    initial_population, current_best, fitnesses = fitness_sort(initial_population, True)
    initial_population = initial_population[:25,]
    
    while n_offspring < 26:
      parent1, parent2 = pick_n_random_individuals(initial_population,2, weights = softmax([1/w for w in fitnesses[:25]]))
      n1, n2 = sorted([secrets.randbelow(1000), secrets.randbelow(1000)])

      kid1, kid2 = cross_parents(parent1, parent2, n1, n2) #TODO

      if np.random.choice([True,False], p = [.9,.1]):
        mutated_kid1 = flip_two_nodes(kid1)
        kid1 = mutated_kid1 if get_path_length(mutated_kid1) < get_path_length(kid1) else kid1
          
      if np.random.choice([True,False], p = [.9,.1]):
        mutated_kid2 = flip_two_nodes(kid2)
        kid2 = mutated_kid2 if get_path_length(mutated_kid2) < get_path_length(kid2) else kid2

      offspring[n_offspring,:,:] = kid1
      offspring[n_offspring + 1,:,:] = kid1
      n_offspring += 2
    
    print(f'generation #: {i}, generation best: {current_best}')
    initial_population = np.concatenate((initial_population, offspring))

  return(current_best)

if __name__ == '__main__':
    data = np.loadtxt('data/tsp.txt',delimiter=',')
    print(get_path_length(data))
    # print(randomly_find_path(1000,data))
    # print(hillclimber_find_path(100000,data))
    print(evolutionary_algo(2500,data))



    