"""
Plot randomly for now
"""
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


def main():
    file_loc = "../data/output/"
    all_files = glob.glob(os.path.join(file_loc, "*_0.0001_0.0005.npy*"))
    results = {}
    for f in all_files:
        name = f.split("_")[1]
        if name != "rmse":
            print(name)
            data = np.load(f)
            results[name] = data

    y = np.linspace(0.0001, 0.0005, 5)
    for k in results:
        plt.plot(y, results[k], label=k)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
