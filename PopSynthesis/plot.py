import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    all_results = np.load('Final_results_1_5.npy')
    SRMSE_result = all_results[0]
    RMSD_result = all_results[1]
    fig, axs = plt.subplots(2)
    X = list(range(1, len(SRMSE_result[0])+1))

    axs[0].plot(X, SRMSE_result[0], label='IPF')
    axs[0].plot(X, SRMSE_result[1], label='BN')
    axs[0].plot(X, SRMSE_result[2], label='Evol')

    axs[1].plot(X, RMSD_result[0], label='IPF')
    axs[1].plot(X, RMSD_result[1], label='BN')
    axs[1].plot(X, RMSD_result[2], label='Evol')

    plt.xlabel('Iteration')
    axs[0].set_ylabel('SRMSE')
    axs[1].set_ylabel('RMSD')

    axs[0].legend()
    axs[1].legend()
    plt.show()    
    