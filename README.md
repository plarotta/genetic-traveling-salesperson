# Genetic Traveling Salesperson
This project captures some of the main takeaways from the first unit of Columbia's class on Evolutionary Computation. I also show how genetic algorithms can be used as gradient-free optimization methods by developing a simple implementation to solve the Traveling Salesperson problem.

## Evolutionary Computation & Genetic Programming
Evolutionary Algorithms (EAs) are a class of optimization algorithms inspired by the principles of natural selection and genetics. These algorithms draw inspiration from the process of evolution observed in biology to solve complex problems and search for optimal solutions. In an EA, a population of potential solutions is initialized, each representing a candidate solution to the problem at hand. Through successive generations, these solutions evolve and improve their fitness by undergoing genetic operations such as crossover (recombination) and mutation.
![1_V22K8UExlLuaegeevJRhdQ](https://github.com/plarotta/genetic-traveling-salesperson/assets/20714356/707981f8-f762-42d1-a1ed-5a0c002b8908)

Genetic Programming (GP) is an evolutionary algorithm inspired by the process of natural selection and genetics. It is utilized in computer science and optimization to automatically evolve computer programs to perform a specific task. In GP, a population of candidate programs, represented as tree structures or graphs, undergoes iterative generations of evolution. Each program is evaluated for its fitness in solving the given problem, and those with higher fitness have a better chance of "surviving" and producing offspring in the next generation. Genetic operators like crossover and mutation are applied to the programs, simulating the genetic recombination and mutation found in biological evolution. This iterative process continues until a satisfactory solution or an optimal program is discovered. GP is versatile and has been successfully applied to a wide range of problems, including symbolic regression, machine learning, and automated code generation.

More succinctly:
```
Function GeneticProgram():
    // Initialization
    population = InitializePopulation()

    // Evolutionary loop
    repeat until termination criteria met:
        // Evaluation
        EvaluateFitness(population)

        // Selection
        parents = SelectParents(population)

        // Crossover (Recombination)
        offspring = Crossover(parents)

        // Mutation
        Mutate(offspring)

        // Replacement
        population = Replace(population, parents, offspring)

    // Output
    best_solution = GetBestSolution(population)
    return best_solution 
```
## Traveling Salesperson Problem
The Traveling Salesperson Problem (TSP) is a classic optimization puzzle where the goal is to find the most efficient route that visits a set of cities exactly once and returns to the starting point. Imagine a salesperson aiming to minimize the total distance traveled while covering all destinations. Despite its seemingly straightforward premise, the TSP is notoriously challenging because the number of possible routes grows exponentially with the number of cities. Solving the TSP efficiently is essential in various fields, such as logistics, transportation planning, and circuit design, making it a fundamental problem in computer science and optimization. Read more [here](https://en.wikipedia.org/wiki/Travelling_salesman_problem).

![image](https://github.com/plarotta/genetic-traveling-salesperson/assets/20714356/374069d6-75f2-4539-b3d7-babe9fe3d84d)

### Solving the Traveling Salesperson Problem through GP
For a simple GP like the one described above, the only operators we need to define for our specific optimization problem are the following:
- Encoder: how we represent candidate solutions so that we can build populations of candidate solutions and perform mutations and crossover events on individual solutions
  - For TSP, we encode each candidate solution as a tour of cities represented as a list. Each city is given by a tuple (x,y) denoting 2D position, so a candidate solution consists of a list of cities *list[tuple]* where the order in the list is the order the cities are explored in.
- Fitness: goodness of a candidate solution
  - Here the quality of a candidate solution is simply the path length of the tour as defined by the total euclidean distance covered.
- Mutations: operator which applies small modifications to candidate solutions
  - In TSP, mutation of a candidate solution simply swaps two cities in the tour.
- Crossover: operator which combines 2 candidate solutions together and produces offspring candidate solutions
  - Here we do a 2-point mutation.![image](https://github.com/plarotta/genetic-traveling-salesperson/assets/20714356/6ea1844f-2085-4a6f-8066-854f21b420f9)

- Selection: this mechanism generates selection to increase the overall fitness of the population of candidate solutions.
  - To achieve this we chose to only pass on the top 33% fittest candidate solutions from one generation to the next

### Data
There are 4 datasets of increasing difficulty included in this repo, and I also included the script _easy_gen.py_ to aid in generating more data if desired. The first 3 datasets are points sampled from the perimeter of a unit circle. These datasets are great because we know that the optimal path is always along the perimeter so the shortest path length we can reach is exactly equal to 2π. Circle10 contains 10 points, circle50 contains 50, and circle500 contains 500. The last datafile is tsp.txt which contains a more complex distribution of 1000 points, and it is the hardest of the 4 datasets. From the [Christofides algorithm](https://en.wikipedia.org/wiki/Christofides_algorithm) we found that the length of the optimal path is around 11. 

## Run this repo
I prefer to keep the python environments of different projects isolated, and my package manager of choice is [Mamba](https://github.com/mamba-org/mamba). Mamba is a fast (written in C), drop-in replacement for conda. It really shines in large ML projects where solving the environment can take hours in conda. It is not necessary for this project, but I still highly recommend Mamba. 

To install Mamba, make sure you DO NOT have conda install and follow the install guide here: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html.

Whether you chose to use mamba or to stick to conda, I recommend for the sake of best practice to build the environment for this project from the provided yml file.

Run 

```conda env create --name YOUR_DESIRED_ENVIRONMENT_NAME --file=environment.yml```  to create the environment

and

```conda activate YOUR_DESIRED_ENVIRONMENT_NAME``` to activate it

With this you should now be able to run genetic_algorithm.py directly via 
```python genetic_algorithm.py```
You can modify the code under ```if __name__ == __main__``` to change the dataset and to adjust the number of generations to run for as well as the population size. 

You'll notice an @njit decorator on some of my helper functions defined in utilities.py, and its main purpose is to speed things up. With Numpy we can use the Numba library to take advantage of [just-in-time compilation ](https://people.duke.edu/~ccc14/sta-663-2016/18C_Numba.html) to massively speed up simple computations. With JIT, the Python code of an individual function only needs to be compiled down to machine code once, and this machine code is cached in memory for quick access. 


```python train_gui.py --data medium --n_gen 12000 --pop_size 40 --select_frac 0.70 --mating_prob .85 --mut_prob .7 --plot_freq 10```