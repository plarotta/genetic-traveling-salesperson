import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from numba import njit
from scipy.spatial.distance import pdist, squareform
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing
import time


def create_single_tour(tour_length=1000):
    #tour represented as a connectivity matrix
    tour = np.zeros((tour_length, tour_length))
    nodes = np.random.permutation(np.arange(tour_length)).reshape(tour_length,1)
    np.put_along_axis(tour, nodes, 1, 1)  
    assert np.all(np.sum(tour,axis=0) == np.ones(tour_length))
    assert np.all(np.sum(tour,axis=1) == np.ones(tour_length))

    return(tour)
# @njit
def two_point_mutation(tour):
    rows_to_switch = np.random.choice(np.arange(len(tour)), 2, replace=False)
    temp = np.copy(tour[rows_to_switch[0],:])
    tour[rows_to_switch[0],:] = np.copy(tour[rows_to_switch[1],:])
    tour[rows_to_switch[1],:] = temp
    assert np.all(np.sum(tour,axis=0) == np.ones(len(tour)))
    assert np.all(np.sum(tour,axis=1) == np.ones(len(tour)))
    return(tour)
# @njit
def tour_crossover(tour1, tour2):
    child1 = tour1@tour2
    assert np.all(np.sum(child1,axis=0) == np.ones(len(child1)))
    assert np.all(np.sum(child1,axis=1) == np.ones(len(child1)))
    return(child1)

def generate_distance_matrix(data_file_path):
    data = np.loadtxt(data_file_path,delimiter=',')
    print(data.shape, len(data))
    a = pdist(data, metric='euclidean')
    return(squareform(a))
@njit
def get_tour_length(tour, d_matrix):
    tour_dists = np.multiply(tour, d_matrix)
    return(np.sum(tour_dists))

def plot_data(tour, data_file_path):
    raw = np.loadtxt(data_file_path,delimiter=',')
    dat = np.zeros((len(tour),2))
    p1 = 0
    for row in range(len(tour)):
        dat[row,:] = raw[p1,:]
        p1 = np.argmax(tour[p1,:])
    
    plt.plot([i[0] for i in dat], [i[1] for i in dat])
    plt.show()
    return(dat)

'''  plt.ion()
    a = create_single_tour(1000)
    for_plot = get_plot_data(a,'data/tsp.txt' )
    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot([i[0] for i in for_plot], [i[1] for i in for_plot])



    # Loop
    for _ in range(50):
        # creating new Y values
        a = create_single_tour(1000)
        for_plot = get_plot_data(a,'data/tsp.txt' )
    
        # updating data values
        line1.set_xdata([i[0] for i in for_plot])
        line1.set_ydata([i[1] for i in for_plot])
    
        # drawing updated values
        figure.canvas.draw()
    
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        figure.canvas.flush_events()
    
        time.sleep(0.1)'''


def g_a(n_generations, population_size, crossover_probability, mutation_probability):
    
    #initialize d matrix
    distance_mat = generate_distance_matrix('data/tsp.txt')

    #initialize starting population
    population = np.array([create_single_tour() for _ in range(population_size)])
    

    for i in range(n_generations):
  
        fitnesses = np.array([get_tour_length(tour, distance_mat) for tour in population])

        if i % 50 == 0:
            plt.clf()
            print(population[np.argmin(fitnesses),:,:])
            plot_data(population[np.argmin(fitnesses),:,:], 'data/tsp.txt')
        # print(fitnesses)
        # input()
        print(f'gen {i+1}, min fitness: {np.min(fitnesses)}')
        selection_probabilities = softmax(-fitnesses)
        # fit selection
        parents = np.random.choice(np.arange(len(population)), size=2, p=selection_probabilities)
        offspring = np.zeros((population_size, 1000, 1000))
        n_offspring = 0

        while n_offspring < population_size:
            if np.random.choice([True,False], p=[crossover_probability,1-crossover_probability]):
                child = tour_crossover(population[parents[0],:,:], population[parents[1],:,:])
                if np.random.choice([True,False], p=[mutation_probability,1-mutation_probability]):
                    child = two_point_mutation(child)

                offspring[n_offspring,:,:] = child
                n_offspring += 1

            else:
                offspring[n_offspring,:,:] = population[parents[0],:,:]
                n_offspring += 1
                if n_offspring >= population_size:
                    continue
                offspring[n_offspring,:,:] = population[parents[1],:,:]
                n_offspring += 1
        
        population = np.copy(offspring)


if __name__ == "__main__":
    a = create_single_tour(5)
    print(a)
    print(two_point_mutation(a))
    # b = np.random.randint(1,66,size=(5,5))
    # print(np.multiply(a,b))
    # print(np.sum(np.multiply(a,b)))

    # # b = create_single_tour(5)
    # # print(a,b)
    # # print(tour_crossover(a,b))
    # m = generate_distance_matrix('data/tsp.txt')
    # g_a(1000,20,1, 0.5)
    # print(solve_tsp_simulated_annealing(m))

  
    
    
