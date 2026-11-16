"""Generate fixed TSP/CVRP datasets used by the paper implementations."""
import argparse
from pathlib import Path
import numpy as np
from utils.data_utils import save_dataset

def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()

def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=['all', 'tsp', 'vrp', 'cvrp'], default='all')
    parser.add_argument('--name', default='dataset')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dataset_size', type=int, default=10000)
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100])
    parser.add_argument('--data_dir', default=str(Path(__file__).resolve().parent / 'data'))
    args = parser.parse_args()
    np.random.seed(args.seed)
    for problem in (['tsp', 'vrp'] if args.problem == 'all' else ['vrp' if args.problem == 'cvrp' else args.problem]):
        for size in args.graph_sizes:
            output = Path(args.data_dir) / problem / f'{problem}{size}_{args.name}_seed{args.seed}.pkl'
            output.parent.mkdir(parents=True, exist_ok=True)
            generate = generate_tsp_data if problem == 'tsp' else generate_vrp_data
            save_dataset(generate(args.dataset_size, size), str(output))
            print(output)
