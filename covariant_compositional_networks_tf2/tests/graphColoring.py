import networkx as nx
from covariant_compositional_networks_tf2.CCN_Model import CCN_Model
import numpy as np
import tensorflow as tf
from ordered_set import OrderedSet

from covariant_compositional_networks_tf2.CCN_Model import CCN_Model
channels_in = 5
feature_vector_shape = [1]
k = 2
model = CCN_Model(lr=1e-5, lr_decay_rate=0.8, lr_min=2e-7, loss=tf.losses.logcosh, nonlinearity=tf.nn.leaky_relu,
                  feature_vector_shape=feature_vector_shape, num_layers=4, k=k,
                  channels_in=[channels_in, 4, 3,3, 1])


def randomNPGraph(n, p, diagonal=True, undirected=True):
    adjM = np.random.binomial(1, p, (n, n))
    if diagonal:
        for i in range(len(adjM)):
            adjM[i, i] = 1
    if undirected:
        xy = np.mgrid[0:n:1, 0:n:1].reshape(2, -1).T.reshape(n, n, 2)
        adjM = np.where(xy[..., 1] > xy[..., 0], adjM, adjM.T)
    return adjM


def randomGraphColoring(n, m, max_color=None):
    if max_color is None:
        max_color = m
    coloring = np.zeros((n, max_color))
    indices = list((np.arange(n), np.random.randint(m, size=n)))
    coloring[tuple(indices)] = 1
    return coloring


def checkGraphColoringError(adjM, coloring):
    neighbours = [np.where(adjM[i] == 1) for i in range(len(adjM))]
    errors = np.array(
        [[np.sum(coloring[i] * coloring[j]) if i != j and j in neighbours[i][0] else 0 for j in range(len(adjM))] for i
         in range(len(adjM))])
    sum_of_errors = np.sum(errors) / 2
    return sum_of_errors


def checkIfGraphConnected(adjM):
    G = nx.from_numpy_matrix(adjM)
    return nx.is_connected(G)


def generateGraphColoring(size, n_range, m_range, p_range):
    m_max = m_range[1]
    graphs = []
    while True:
        n = np.random.randint(n_range[0], n_range[1])
        m = np.random.randint(m_range[0], m_range[1])
        p = np.random.uniform(p_range[0], p_range[1])
        # print("n: " + str(n) +", m: " +str(m)+ ", p: " + str(p))
        NPGraph = randomNPGraph(n, p)
        connected = checkIfGraphConnected(NPGraph)
        if not connected:
            continue
        coloring = randomGraphColoring(n, m, max_color=m_max)
        coloringError = checkGraphColoringError(NPGraph, coloring)
        coloringError = 1000.0 if coloringError ==0 else -1000
        parts = [OrderedSet([i]) for i in range(len(NPGraph))]
        graph = [NPGraph, coloring, coloringError, parts]
        graphs.append(graph)
        if len(graphs) >= size:
            break
    return graphs


# n = 7  # nodes
# m = 4  # colors
# p = 0.4  # edge probability
# NPGraph = randomNPGraph(n, p)
# coloring = randomGraphColoring(n, m, max_color=None)
#
# connected = checkIfGraphConnected(NPGraph)
# coloringError = checkGraphColoringError(NPGraph, coloring)
# print(coloring)
# print(connected)
# print(coloringError)

graphs = list(zip(*generateGraphColoring(200, (3, 6), (channels_in-1, channels_in), (0.2, 0.5))))
print(graphs)

print(np.unique(graphs[2], return_counts=True))

adjM = graphs[0]
X = graphs[1]
Y = graphs[2]
parts = graphs[3]
X,Y = model.createTensors(X,Y)
model.fit(X, Y, adjM, parts, 100)
