"""This example illustrates MARBLE for a vector field on a flat surface."""
import numpy as np
import sys
from MARBLE import plotting, preprocessing, dynamics, net, postprocessing
import matplotlib.pyplot as plt


def f0(x):
    return x * 0 + np.array([-1, -1])

def f1(x):
    return x * 0 + np.array([1, 1])

def f2(x):
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] + 1) ** 2 + x[:, [1]] ** 2 + eps)
    u = x[:, [1]] / norm
    v = -(x[:, [0]] + 1) / norm
    return np.hstack([u, v])

def f3(x):
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] - 1) ** 2 + x[:, [1]] ** 2 + eps)
    u = x[:, [1]] / norm
    v = -(x[:, [0]] - 1) / norm
    return np.hstack([u, v])

# def f2(x):
#     eps = 1e-1
#     norm = np.sqrt((x[:, [0]]) ** 2 + x[:, [1]] ** 2 + eps)
#     u = -(x[:, [0]] ) / norm
#     v = -(x[:, [1]] ) / norm
#     return np.hstack([u, v])

# def f3(x):
#     eps = 1e-1
#     norm = np.sqrt((x[:, [0]]) ** 2 + x[:, [1]] ** 2 + eps)
#     u = (x[:, [0]] ) / norm
#     v = (x[:, [1]] ) / norm
#     return np.hstack([u, v])


def main():

    # generate simple vector fields
    # f0: linear, f1: point source, f2: point vortex, f3: saddle
    n = 512
    x = [dynamics.sample_2d(n, [[-1, -1], [1, 1]], "random", seed=i) for i in range(4)]
    y = [f0(x[0]), f1(x[1]), f2(x[2]), f3(x[3])]  # evaluated functions

    # construct data object
    data = preprocessing.construct_dataset(x, y)

    # train model
    model = net(data, params={'inner_product_features': False, 
                              'diffusion': False})
    model.fit(data)

    # evaluate model on data
    data = model.transform(data)
    data = postprocessing.cluster(data)
    data = postprocessing.embed_in_2D(data)

    # plot results
    titles = ["Linear left", "Linear right", "Vortex right", "Vortex left"]
    plotting.fields(data, titles=titles, col=2, width=0.01)
    plt.savefig('fields.svg')
    plotting.embedding(data, data.y.numpy(), titles=titles, clusters_visible=True)
    plt.savefig('embedding.svg')
    plotting.histograms(data, titles=titles)
    plt.savefig('histogram.svg')
    plotting.neighbourhoods(data)
    plt.savefig('neighbourhoods.svg')
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
