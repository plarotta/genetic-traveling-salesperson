'''speedy helper functions for tsp ga'''

import numpy as np
from numba import njit


@njit(nogil=True, fastmath=True)
def get_path_length(path: np.array) -> float:
    '''calculates the distance covered by the path.
    
    Args:
      path:
        An array representing a path through the cities in the 
        order that they appear in the array

    Returns:
      float representing euclidian length traveled through
      the path
    '''
    tot = np.array([
      np.linalg.norm(path[i+1]-path[i])
      for i in range(len(path)-1)
      ])
    tot = np.sum(tot)
    return(np.round(tot,3))

def randomly_find_path(iterations: int, data: np.array) ->list:
    '''randomly generates paths and returns the distance of the shortest one

    Args:
      iterations:
        An array representing a path through the cities in the 
        order that they appear in the array
      data:
        array (1000,2) of the input dataset of cities. each
        row holds the x and y coordinate of its city

    Returns:
      best path lengths
    '''
    path_lengths = []
    best_dist = get_path_length(np.random.permutation(data))
    for i in range(iterations-1):
      new_path_length = get_path_length(np.random.permutation(data))
      if new_path_length < best_dist:
        path_lengths.append(new_path_length)
        best_dist = new_path_length
    return(path_lengths)

@njit
def flip_two_nodes(data: np.array) -> np.array:
  '''randomly swaps two cities in the path and returns the modified path
  
  Args:
      data:
        array (1000,2) representing a path.

  Returns:
    modified version of data where two cities in the path are switched
  '''
  modified_path = np.copy(data)
  n1,n2 = [np.random.randint(low=0, high=len(data)), 
                    np.random.randint(low=0, high=len(data))]
  indices = [n1,n2] if n1 < n2 else [n2,n1]
  modified_path[indices[0],] = data[indices[1],]
  modified_path[indices[1],] = data[indices[0],]
  return(modified_path)


def hillclimber_find_path(max_depth: int, data: np.array) -> float:
  '''simple random-mutation hillclimber to find the shortest path 
  between the cities. the random mutation just randomly switches
  the order of 2 cities in the path
  
  Args:
      max_depth:
        how many iterations to run the search for.
      data:
        array (1000,2) representing a path.

  Returns:
    length of the shortest path found
  '''
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

# @njit
def generate_segment(segment: np.array, reference: np.array) -> np.array:
  '''regenerate segment in the order that it appears in reference
  
  Args:
    segment:
      path segment to be regenerated
    reference:
      segment where the order of cities will be extracted from

  Returns:
    modified version of segment where the order of the cities it 
    contains is adjusted to match the order they appear in the 
    reference
  '''
  sorter = reference.argsort()
  i = sorter[np.searchsorted(reference, segment, sorter=sorter)]
  out = segment[np.argsort(i)]
  return(out)

def gen_cipher(dataset: np.array):
  cipher = {}
  for idx, city in enumerate(dataset):
    tup_entry = tuple(city)
    cipher[tup_entry] = idx
    cipher[idx] = tup_entry
  return(cipher)

def cross_parents(parent1: np.array, 
                  parent2: np.array, 
                  crosspoint1: int, 
                  crosspoint2: int,
                  mating_prob: float,
                  cipher=None
                  ) -> tuple:
  '''evolutionary operator for crossing individuals in a population of solutions.
  in TSP, individual solutions are paths through the 1000 cities in tsp.txt, and 
  the mechanism of crossing over is a simple 2-point crossover
  
  Args:
    parent1:
      solution candidate 1
    parent2:
      solution candidate 2
    crosspoint1:
      for 2-point crossover
    crosspoint2:
      for 2-point crossover

  Returns:
    2 offspring solutions generated from thr 2-point crossover of the 2 parent
    candidate solutions
  '''

  if not np.random.choice([True,False],p=[mating_prob, 1-mating_prob]):
    return(parent1,parent2)
  
  p1 = np.array([cipher[tuple(city)] for city in parent1])
  p2 = np.array([cipher[tuple(city)] for city in parent2])

  segment_A = generate_segment(p2[:crosspoint1,], p1)
  segment_B = generate_segment(p1[crosspoint1:crosspoint2,], p2)
  segment_C = generate_segment(p2[crosspoint2:,], p1)

  kid1 = np.concatenate((p1[:crosspoint1], segment_B, p1[crosspoint2:]))
  kid2 = np.concatenate((segment_A, p2[crosspoint1:crosspoint2], segment_C))

  k1 = np.array([cipher[city] for city in kid1])
  k2 = np.array([cipher[city] for city in kid2])
  return(k1, k2)

def pick_n_random_individuals(population: np.array, n: int, weights: list) -> list:
  '''pick n random individuals from a population using fitness-proportionate
  selection. 
  
  Args:
    population:
      all candidate solutions in the current population
    n:
      number of individual solutions to choose
    weights:
      for 2-point crossover

  Returns:
    n solutions from the population picked using fitness-proportionate selection
  '''
  out_idx = np.random.choice([i for i in range(len(population))], n, replace = False, p = weights)
  output_individuals = [population[out_idx[0],], population[out_idx[1],]]
  return(output_individuals)

@njit
def fitness_sort(population: np.array) -> tuple[np.array,np.array]:
  '''sort population of solutions by fitness.

  Args:
    population:
      all candidate solutions in the current population

  Returns:
    fitness sorted population, shortest path length in population,
    and the sorted array of all fitnesses in population
  '''
  fit_vals = np.array([get_path_length(p) for p in population])
  sorted_fitnesses = np.sort(fit_vals)
  sorted_population = population[fit_vals.argsort()[::]]
  return(sorted_population, sorted_fitnesses[0], sorted_fitnesses)

@njit(nogil=True, fastmath=True)
def select_member(probabilities):
  '''roulette-wheel selection'''
  magic_number = np.random.random() # for selection
  running_sum = 0
  for idx,prob in enumerate(probabilities):
      running_sum += prob
      if running_sum >= magic_number:
          return(idx)
