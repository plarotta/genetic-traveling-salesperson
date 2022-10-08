import multiprocessing
import time
import pickle
import numpy as np
import random


def read_data():
  f = open("tsp.txt", "r")
  data_set = set()
  while(True):
      line = f.readline()
      if not line:
          break
      line = [float(x) for x in line.strip().split(",")]
      data_set.add(tuple(line))
  f.close
  return(data_set)

def get_path_length(path, city_master):
  tot = sum([np.linalg.norm(np.array(city_master[path[i+1]])-np.array(city_master[path[i]])) for i in range(len(path)-1)]) #this is the flashy one-liner
  return(tot)

def flip_path(path):
  path_copy = path.copy()
  indices = random.sample(range(len(path)), 2)
  path_copy[indices[0]], path_copy[indices[1]] = path_copy[indices[1]], path_copy[indices[0]]
  return(path_copy)

def cross_parents(parent1, parent2, crosspoint1, crosspoint2):
  segment_B = parent1[crosspoint1:crosspoint2]
  order_B = []
  for city in segment_B:
    order_B.append(parent2.index(city))
  adjusted_B = [x for y, x in sorted(zip(order_B, segment_B))]
  kid1 = parent1[:crosspoint1] + adjusted_B + parent1[crosspoint2:]
  segment_A = parent2[:crosspoint1]; segment_C = parent2[crosspoint2:]
  order_A = []; order_C = []
  for city in segment_A:
    order_A.append(parent1.index(city))
  adjusted_A = [x for y, x in sorted(zip(order_A, segment_A))]
  for city in segment_C:
    order_C.append(parent1.index(city))
  adjusted_C = [x for y, x in sorted(zip(order_C, segment_C))]
  kid2 = adjusted_A + parent2[crosspoint1:crosspoint2] + adjusted_C
  return(kid1, kid2)

def mutation(m_rate):
  return(random.choices([True, False], weights = [m_rate,1], k = 1))

def crossover_event(c_rate):
  return(random.choices([True, False], weights = [c_rate,1], k = 1))

def rf_simulator(iterations):
  lengths = []
  evals = []
  current = random.sample(range(1000), 1000)
  best_dist = get_path_length(current)
  for i in range(iterations):
    new_path = random.sample(range(1000), 1000)
    new_path_length = get_path_length(new_path)
    if new_path_length < best_dist:
      lengths.append(new_path_length)
      evals.append(i)
      best_dist = new_path_length
  return(lengths, evals)

def rmhc(max_depth):
  #generate path
  tries = 0
  current_path = random.sample(cities, n)
  current_length = get_path_length(current_path)
  lengths = []
  eval_n = []
  while tries < max_depth:
    next_path = flip_path(current_path)
    tries += 1
    next_length = get_path_length(next_path)
    if next_length < current_length:
      lengths.append(next_length)
      eval_n.append(tries)
      current_path = next_path
      current_length = next_length

  return(lengths, current_path, eval_n)


def evolutionary_algo(init_pop_size, mut_rate, crossover_rate, selection_method):
  data = read_data()
  city_master = list(data) 
  n_generations = int(100000/init_pop_size)
  initial_population = [random.sample(range(1000), 1000) for x in range(init_pop_size)]
  fitnesses = [get_path_length(p, city_master) for p in initial_population]
  mins = []
  for i in range(n_generations):
    offspring = set()
    if selection_method == "trunc":
      initial_population = [x for y, x in sorted(zip(fitnesses, initial_population))]
      initial_population = initial_population[:int(init_pop_size*1/4)]
      weights = [1 for i in initial_population]
    else:
      weights = [1/w for w in fitnesses]
    while len(offspring) < init_pop_size:
      parent1, parent2 = random.choices(initial_population, weights, k = 2) #roulette
      if crossover_event(crossover_rate):
        n1, n2 = sorted(random.sample(range(1,998), 2))
        kid1, kid2 = cross_parents(parent1, parent2, n1, n2)
        if mutation(mut_rate):
          kid1 = flip_path(kid1)
        if mutation(mut_rate):
          kid2 = flip_path(kid2)
        offspring.add(tuple(kid1)); offspring.add(tuple(kid2))
      else:
        offspring.add(tuple(parent1)); offspring.add(tuple(parent2))
    #add good parents?
    [offspring.add(tuple(p)) for p in initial_population[:int(len(initial_population)*.15)]]
    initial_population = [list(i) for i in offspring]
    fitnesses = [get_path_length(j, city_master) for j in initial_population]
    mins.append(min(fitnesses))

  return(mins, [x for y, x in sorted(zip(fitnesses, initial_population))][0])






res_GA_24 = []
res_GA_24_paths = []
res_GA_40 = []
res_GA_40_paths = []
random_res = {"lengths": [], "n_evals": []}
rmhc_res = {"lengths": [], "n_evals": []}

x_err_keys = [n for n in range(5000, 100000, 5000)]

for i in range(5):
    rf_length, rf_evals = rf_simulator(100000)
    random_res["lengths"].append(rf_length)
    random_res["n_evals"].append(rf_evals)

    rmhc_length, rmhc_evals = rmhc(100000)
    rmhc_res["lengths"].append(rmhc_length)
    rmhc_res["n_evals"].append(rmhc_evals)

    ga24_lengths, ga24_path = evolutionary_algo(24, .8, .95, "trunc")
    res_GA_24.append(ga24_lengths)
    res_GA_24_paths.append(ga24_path)

    ga40_lengths, ga40_path = evolutionary_algo(40, .8, .95, "trunc")
    res_GA_40.append(ga40_lengths)
    res_GA_40_paths.append(ga40_path)

