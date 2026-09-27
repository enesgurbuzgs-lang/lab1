import numpy as np
import matplotlib.pyplot as plt


def plot_histograms(matrix):
    matrix = np.asarray(matrix)

    for i, row in enumerate(matrix):
        plt.figure()
        plt.hist(row)
        plt.title(f"Row {i}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

    for j, column in enumerate(matrix.T):
        plt.figure()
        plt.hist(column)
        plt.title(f"Column {j}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()