from gene_tsp.evo_base import Evolution
from gene_tsp.utilities import *
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
        # probs = fitnesses/np.sum(fitnesses)
        # probs = [1-i for i in probs]
        # return(softmax(-fitnesses))
        n = len(fitnesses)
        prob_vector = [1.0/n for _ in range(n)]

        # two best
        prob_vector[0] = 8.0 * prob_vector[0]
        prob_vector[1] = 8.0 * prob_vector[1]

        #  top 50%
        for idx in range(2,int(n/2)):
            prob_vector[idx] = 2.0 * prob_vector[idx]

        # normalize
        tot = sum(prob_vector)
        prob_vector = [j / tot for j in prob_vector]
        return(prob_vector)
        # return(probs)
    
    def select_individual(self, probabilities):
        '''roulette-wheel selection'''
        return(select_member(probabilities))
    
    def select_parents(self, probabilities, population):
        p1 = self.select_individual(probabilities) 

        for _ in range(10): # 10 tries to find a different parent
            p2 = self.select_individual(probabilities) 
            if p2 != p1:
                return(population[p1], population[p2])
        
        # if fitness-prop failed to get a different parent, choose one randomly
        p2 = set(i for i in list(range(len(population))) if i != p1).pop() 
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

            if idx<population_size-1:
                next_gen[idx+1] = offspring[1]

            # next_gen[idx] = offspring[0] if get_path_length(parent1) > get_path_length(offspring[0]) else parent1
            # next_gen[idx+1] = offspring[1] if get_path_length(parent2) > get_path_length(offspring[1]) else parent2
        return(next_gen)
    
    def visualize_member(self, member):
        pass

        

if __name__ == '__main__':
    data = np.loadtxt('data/circle500.txt',delimiter=' ')
    a = GeneTSP(data)