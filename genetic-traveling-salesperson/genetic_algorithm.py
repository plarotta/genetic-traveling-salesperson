import wandb
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from utilities import *
from evo_tsp import GeneTSP

def genetic_agorithm(n_generations: int, 
                     data: np.array, 
                     population_size: int,
                     selection_fraction: float,
                     mating_probability: float,
                     mutation_probability: float,
                     wandb_logging=True
                     ) -> tuple[float,np.array]:
    np.random.seed(42)
    run = wandb.init() if wandb_logging is True else None

    if population_size % 2 != 0:
        population_size += 1 
    
    evolution = GeneTSP(data)
    
    # initiate population
    population =  evolution.produce_initial_population(population_size)
    all_time_best = 100000

    for i in range(n_generations):
        # input()
        pop_diversity = np.mean(np.var(population,axis=0))
        population, best, fitnesses = evolution.get_population_fitness(population)
        selection_probs = evolution.get_selection_probabilities(fitnesses)
        population = evolution.produce_next_generation(population_size, 
                                                       selection_probs, 
                                                       population, 
                                                       mutation_probability,
                                                       mating_probability)
        all_time_best = best if best < all_time_best else all_time_best
        population = population[:int(selection_fraction*len(population))]

        if wandb_logging is True:
            run.log({'Generation': i+1, "Generation Best": best, 'All-time Best': all_time_best, "Diversity": pop_diversity})
        else:
            print(f'Gen # {i+1}| Gen best: {best:.2f} | All-time best: {all_time_best:.2f} | Diversity: {pop_diversity:.2f}')

    
    # return(current_best, best_indiv)

if __name__ == '__main__':
    data = np.loadtxt('data/circle500.txt',delimiter=' ')
    genetic_agorithm(n_generations=5000, 
                     data=data, 
                     population_size=50, 
                     selection_fraction=.9, 
                     mating_probability=.6, 
                     mutation_probability=.85, 
                     wandb_logging=False)