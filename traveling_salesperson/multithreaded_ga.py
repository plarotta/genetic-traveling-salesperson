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

def evolutionary_algo3(init_pop_size, mut_rate, crossover_rate, selection_method):
  data = read_data()
  city_master = list(data) 
  n_generations = int(50000/init_pop_size)
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




def threader(n, mut, cross, pop, sel):
    path_lengths, best_path = evolutionary_algo(n, mut, cross, pop, sel)
    filename_l = "GA_" +"sel-" + str(sel) + "_ngen-" + str(n) + "_pop-" + str(pop) + "_mrate-" + str(mut) + "_crate-" + str(cross) + "_LENGTHS"
    filename_p = "GA_" +"sel-" + str(sel) + "_ngen-" + str(n) + "_pop-" + str(pop) + "_mrate-" + str(mut) + "_crate-" + str(cross) + "_PATH"

    with open(filename_l, 'wb') as fp:
        pickle.dump(path_lengths, fp)

    with open(filename_p, 'wb') as fp:
        pickle.dump(best_path, fp)


if __name__ == '__main__':
  selection_methods = ["trunc", "roulette"]
  generations = 10000
  population_sizes = [20, 50, 100]
  mutation_rates = [0.1, 0.01, 0.001]
  crossover_rates = [0.3, 0.5, 0.8]
  
#   processes = []
  
  starttime = time.time()

#   for s in selection_methods:
#       for p in population_sizes:
#           for m in mutation_rates:
#               for c in crossover_rates:
#                 p = multiprocessing.Process(target=threader, args=(generations, m, c, p, s))
#                 processes.append(p)
#                 p.start()
        
#   for process in processes:
#     process.join()
  res = evolutionary_algo(100, .01, .6, 30, "trunc")
  print(res[0][-1])
  print('That took {} seconds'.format(time.time() - starttime))



  