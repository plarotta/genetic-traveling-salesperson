from gene_tsp.genetic_algorithm import genetic_agorithm
from gene_tsp.gui import *
import threading
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store")
    parser.add_argument("--n_gen", action="store", type=int)
    parser.add_argument("--pop_size", action="store", type=int)
    parser.add_argument("--select_frac", action="store", type=float)
    parser.add_argument("--mating_prob", action="store", type=float)
    parser.add_argument("--mut_prob", action="store", type=float)
    parser.add_argument("--plot_freq", action="store", type=int)

    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

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

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    threading.Thread(target=genetic_agorithm, kwargs={"n_generations": args.n_gen, 
                                                      "data": data, 
                                                      "population_size": args.pop_size, 
                                                      "selection_fraction":args.select_frac, 
                                                      "mating_probability":args.mating_prob, 
                                                      "mutation_probability":args.mut_prob,
                                                      "gui":window, 
                                                      "plot_freq": args.plot_freq,
                                                      "wandb_logging":False}).start()

    sys.exit(app.exec_())