'''General base class for evolutionary algorithms. Reused in several other projects'''

from abc import ABC, abstractmethod

class Evolution(ABC):
    @abstractmethod
    def mutate(self):
        pass
    @abstractmethod
    def crossover(self):
        pass
    @abstractmethod
    def select_parents(self):
        pass
    @abstractmethod
    def get_member_fitness(self):
        pass
    @abstractmethod
    def get_population_fitness(self):
        pass
    @abstractmethod
    def get_selection_probabilities(self):
        pass
    @abstractmethod
    def produce_initial_population(self):
        pass
    @abstractmethod
    def produce_next_generation(self):
        pass