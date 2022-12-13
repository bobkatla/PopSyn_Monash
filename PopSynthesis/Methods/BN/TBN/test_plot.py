import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    SRMSE_result = np.load('Testing/GA_results_SRMSE_2.npy')
    RMSD_result = np.load('Testing/GA_results_RMSD_2.npy')
    fig, axs = plt.subplots(2)
    X = list(range(1, len(SRMSE_result)+1))
    axs[0].plot(X, SRMSE_result)
    axs[1].plot(X, RMSD_result)
    plt.xlabel('Iteration')
    axs[0].set_ylabel('SRMSE')
    axs[1].set_ylabel('RMSD')
    plt.show()    
    