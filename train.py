from gene_tsp.genetic_algorithm import genetic_agorithm
import argparse
import numpy as np


def main(args):
    if args.data == "easy":
        data = np.loadtxt('data/easy.txt',delimiter=' ')
    elif args.data == "medium":
        data = np.loadtxt('data/medium.txt',delimiter=' ')
    elif args.data == "hard":
        data = np.loadtxt('data/hard.txt',delimiter=' ')
    elif args.data == "challenge":
        data = np.loadtxt('data/challenge.txt',delimiter=',')
    else:
        raise ValueError("Invalid dataset")
    
    genetic_agorithm(n_generations=args.n_gen, 
                     data=data,
                     population_size=args.pop_size,
                     selection_fraction=args.select_frac,
                     mating_probability=args.mating_prob,
                     mutation_probability=args.mut_prob,
                     wandb_logging=args.wandb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store")
    parser.add_argument("--n_gen", action="store", type=int)
    parser.add_argument("--pop_size", action="store", type=int)
    parser.add_argument("--select_frac", action="store", type=float)
    parser.add_argument("--mating_prob", action="store", type=float)
    parser.add_argument("--mut_prob", action="store", type=float)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    main(args)


