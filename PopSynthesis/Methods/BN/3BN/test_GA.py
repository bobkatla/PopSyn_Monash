import pandas as pd


"""
    Some parameters:
    - How to pick parents and the individuals for next round (maybe not the same as eval func)
    - How many solutions per gen
    - How to do crossover (producing offsprings)
    - How to mutate and where/who, to what extent at what rate
    - How many generations or the termination criteria
    - Heuristic approach, how? 
"""

def learn_BN(data):
    model = None
    return model


def sample_BN(model, n):
    df = None
    return df


def eval_func(indi):
    score = 0
    return score

def cross_entropy(dist1, dist2):
    NotImplemented


def mutation(indi, BN_model, partition_rate=0.25, num_keep_atts=3, num_child=5):
    arr_solutions = []
    # find the best fit atts
    for _ in range(num_child):
        # partition randomly based on the ratio
        # BN inference for the rest of them atts in mutation
        # combine again
        arr_solutions.append(None)
    return arr_solutions


def crossover(pa1, pa2, partition_rate=0.4):
    offspring = []
    # partition randomly based on the ratio for pa1
    # partition randomly based on the ratio for pa2
    # swap
    return offspring


def eval_loop(first_gen):
    # Select the individuals for the operations (the criteria is unknown)
    # Producing offsprings (reproduction/ crossover)
    # Mutate offspring
    # Eval each of individual with the eval func
    # Select the "best" for next round (or replacement)
    NotImplemented


def EvoProg():
    # Initial solutions/ population
    # Run loop
    # Pick the final solution
    NotImplemented

def main():
    NotImplemented


if __name__ == "__main__":
    main()