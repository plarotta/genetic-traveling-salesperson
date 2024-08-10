import sys
import os
sys.path.append(os.getcwd())
import wandb
import numpy as np
from gene_tsp.utilities import *
from gene_tsp.evo_tsp import GeneTSP
from gene_tsp.gui import *
import threading


def genetic_agorithm(n_generations: int, 
                     data: np.array, 
                     population_size: int,
                     selection_fraction: float,
                     mating_probability: float,
                     mutation_probability: float,
                     gui=None,
                     plot_freq=1,
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
        pop_diversity = np.mean(np.var(population,axis=0))
        population, best, fitnesses = evolution.get_population_fitness(population)
        
        if i % plot_freq == 0 and gui is not None:
            gui.plot_member(population[0], i, best, pop_diversity,all_time_best)

        # selection pressure
        population = population[:int(selection_fraction*population_size)]
        fitnesses = fitnesses[:int(selection_fraction*population_size)]

        selection_probs = evolution.get_selection_probabilities(fitnesses)
        population = evolution.produce_next_generation(population_size,
                                                       selection_probs, 
                                                       population, 
                                                       mutation_probability,
                                                       mating_probability)


        all_time_best = best if best < all_time_best else all_time_best

        if wandb_logging is True:
            run.log({'Generation': i+1, "Generation Best": best, 'All-time Best': all_time_best, "Diversity": pop_diversity})
        else:
            print(f'Gen # {i+1}| Gen best: {best:.2f} | All-time best: {all_time_best:.2f} | Diversity: {pop_diversity:.2f}')
        
    return(best, population[0])

if __name__ == '__main__':
    data = np.loadtxt('data/tsp.txt',delimiter=',')

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    threading.Thread(target=genetic_agorithm, kwargs={"n_generations": 3000, 
                                                      "data": data, 
                                                      "population_size": 60, 
                                                      "selection_fraction":0.5, 
                                                      "mating_probability":.8, 
                                                      "mutation_probability":.9,
                                                      "gui": window, 
                                                      "plot_freq": 5,
                                                      "wandb_logging":False}).start()

    sys.exit(app.exec_())