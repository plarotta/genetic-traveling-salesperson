from evo_base import Evolution
from utilities import *
import numpy as np
from scipy.special import softmax


class GeneTSP(Evolution):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.cipher = gen_cipher(self.data)

    def mutate(self, member, mut_prob) -> np.array:
        if np.random.choice([True,False],p=[mut_prob, 1-mut_prob]):
            return(flip_two_nodes(member))
        else:
            return(member)
    
    def produce_initial_population(self, population_size) -> np.array:
        return(np.array([np.random.permutation(self.data) 
                         for _ in range(population_size)]))
    
    def get_member_fitness(self, member):
        return(get_path_length(member))
    
    def get_population_fitness(self, population):
        '''returns sorted population, best member, and sorted fitnesses'''
        return(fitness_sort(population))
    
    def get_selection_probabilities(self, fitnesses):
        return(softmax(-fitnesses))
    
    def select_individual(self, probabilities):
        '''roulette-wheel selection'''
        return(select_member(probabilities))
    
    def select_parents(self, probabilities, population):
        p1,p2 = [self.select_individual(probabilities) for _ in range(2)]
        return(population[p1], population[p2])
    
    def crossover(self, parent1, parent2, mating_prob):
        '''returns two offspring created using 2-point crossover'''

        # crossover points for 2-point crossover
        n1, n2 = sorted([np.random.randint(low=0, high=len(self.data)), 
                    np.random.randint(low=0, high=len(self.data))])
        return(cross_parents(parent1, parent2, n1, n2, mating_prob, self.cipher))   

    def produce_next_generation(self, population_size, probabilities, population, mut_p, mat_p):
        next_gen = np.zeros((population_size,len(self.data),2))

        for idx in range(0,population_size,2):
            parent1, parent2 = self.select_parents(probabilities, population)
            offspring = self.crossover(parent1, parent2, mat_p)
            offspring = tuple(self.mutate(child, mut_p) for child in offspring)
            next_gen[idx] = offspring[0]
            next_gen[idx+1] = offspring[1]
        return(next_gen)

        

if __name__ == '__main__':
    data = np.loadtxt('data/circle500.txt',delimiter=' ')
    a = GeneTSP(data)