"""
    Plot save extrinsice score average files for policies.
"""
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=str, nargs="+", help="Paths to the "
        "policy score files that you wish to plot.")

    args        = parser.parse_args()
    score_files = args.scores

    sns.set()

    score_arrays = []
    score_names  = []
    for sf in score_files:
        name = " ".join(os.path.basename(sf).split(".")[0].split("_"))
        score_names.append(name)
        with open(sf, "rb") as in_f:
            score_arrays.append(np.loadtxt(in_f))

    size = -1
    for sa in score_arrays:
        if size >= 0:
            assert sa.size == size, "number of iterations must match!"
        size = sa.size
    
    iterations = np.arange(size)

    plt.xlabel("Iterations")
    plt.ylabel("Extrinsic Score Average")

    for scores in score_arrays:
        plt.plot(iterations, scores)
    
    plt.legend(labels=score_names)
    plt.show()
