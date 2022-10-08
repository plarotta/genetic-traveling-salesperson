import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os

#read in data
f = open("./TSP/tsp.txt", "r")
data_set = set()
while(True):
	line = f.readline()
	if not line:
		break
	line = [float(x) for x in line.strip().split(",")]
	data_set.add(tuple(line))
f.close

city_master = list(data_set)
n = len(city_master)
cities = [x for x in range(n)]

def get_path_length(path):
  tot = sum([np.linalg.norm(np.array(city_master[path[i+1]])-np.array(city_master[path[i]])) for i in range(len(path)-1)]) #this is the flashy one-liner
  return(tot)


def rf_simulator(iterations):
  cities = [x for x in range(n)]
  best_dist = 100000
  all_lengths = []

  for i in range(iterations):
    new_path = random.sample(cities, n)
    new_path_length = get_path_length(new_path)
    if new_path_length < best_dist:
      all_lengths.append(new_path_length)
      best_dist = new_path_length
  return(all_lengths)

def flip_path(path):
  path_copy = path.copy()
  indices = random.sample(range(1000), 2)
  path_copy[indices[0]], path_copy[indices[1]] = path_copy[indices[1]], path_copy[indices[0]]
  return(path_copy)


def rmhc(max_depth):
  tries = 0
  current_path = random.sample(cities, n)
  while tries < max_depth:
    next_path = flip_path(current_path)
    tries += 1
    if get_path_length(next_path) < get_path_length(current_path):
      current_path = next_path

  return(get_path_length(current_path))

def cross_parents(parent1, parent2, crosspoint1, crosspoint2):
  #segment_A = parent1[:crosspoint1]
  segment_B = parent1[crosspoint1:crosspoint2]
  #segment_C = parent1[crosspoint2:]
  order_B = []

  for city in segment_B:
    order_B.append(parent2.index(city))
  
  adjusted_B = [x for y, x in sorted(zip(order_B, segment_B))]

  kid1 = parent1[:crosspoint1] + adjusted_B + parent1[crosspoint2:]
  
  segment_A = parent2[:crosspoint1]
  #segment_B = parent2[crosspoint1:crosspoint2]
  segment_C = parent2[crosspoint2:]
  order_A = []
  order_C = []
  
  for city in segment_A:
    order_A.append(parent1.index(city))
  
  adjusted_A = [x for y, x in sorted(zip(order_A, segment_A))]

  for city in segment_C:
    order_C.append(parent1.index(city))
  
  adjusted_C = [x for y, x in sorted(zip(order_C, segment_C))]
  
  kid2 = adjusted_A + parent2[crosspoint1:crosspoint2] + adjusted_C
  
  return(kid1, kid2)

def evolutionary_algo(n_generations):
  initial_population = [random.sample(cities, 1000) for x in range(50)]
  fitnesses = [get_path_length(p) for p in initial_population]

  for i in range(n_generations):
    offspring = []
    n_offspring = 0
    
    initial_population = [x for y, x in sorted(zip(fitnesses, initial_population))] #sort by fitnesses
    initial_population = initial_population[:25]
    
    while n_offspring < 26:
      parent1, parent2 = random.sample(initial_population, 2) 
      roll_die = random.randrange(2)
      magic_num = random.randrange(2) #for crossover
      if roll_die == magic_num:
        n1, n2 = sorted(random.sample(range(1000), 2))
        kid1, kid2 = cross_parents(parent1, parent2, n1, n2)
        mut_num = random.randrange(1000)
        if random.randrange(1000) == mut_num:
          kid1 = flip_path(kid1)
        mut_num = random.randrange(1000)
        if random.randrange(1000) == mut_num:
          kid2 = flip_path(kid2)
        offspring.append(kid1); offspring.append(kid2)
        n_offspring += 2
    initial_population += offspring 
    fitnesses = [get_path_length(j) for j in initial_population]

  return(min(fitnesses))