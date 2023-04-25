import sys
import numpy as np
import matplotlib.pyplot as plt

def get_data(points, seed=None):
    if seed != None:
        np.random.seed(2)
    data = np.random.rand(points, 2)
    return data

def plot_data(data):
    plt.scatter(data[:, 0], data[:,1])
    plt.show()


data = get_data(100, 2)
plot_data(data)