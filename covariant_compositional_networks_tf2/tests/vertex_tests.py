import networkx as nx
from covariant_compositional_networks_tf2.CCN_Model import CCN_Model
import numpy as np
import tensorflow as tf
from ordered_set import OrderedSet
from covariant_compositional_networks_tf2.tests.graphColoring import randomNPGraph, checkIfGraphConnected

from covariant_compositional_networks_tf2.CCN_Model import CCN_Model
channels_in = 5
feature_vector_shape = [1]
k = 2
model = CCN_Model(optimizer= tf.keras.optimizers.Adam(lr = 0.005), loss=tf.losses.binary_crossentropy, nonlinearity=tf.nn.relu,
                  feature_vector_shape=feature_vector_shape, num_layers=1, k=k, batch_update_size=60, l1_reg=0.004, save_every=2,
                  channels_in=[channels_in, 40])

def avg_vertex_degree(graph, diagonal):
    return np.sum([np.sum(graph[i])-diagonal for i in range(len(graph))])/len(graph)

def generate_avg_vertex_degree(size, n_range, p_edge, diagonal=True):
    graphs = []
    while len(graphs)<size:
        NPGraph = randomNPGraph(np.random.randint(*np.range))
        if checkIfGraphConnected(NPGraph):
            feature_vec = np.ones(len(NPGraph))
            parts = [OrderedSet([i]) for i in range(len(NPGraph))]
            avg_vtx_dgre = avg_vertex_degree(NPGraph, diagonal)
            graph = [NPGraph, feature_vec, avg_vtx_dgre, parts]
            graphs.append(graph)
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
data_size = 600
graphs = list(zip(*generate_avg_vertex_degree(data_size, (3, 7), .5)))
graphsValid = list(zip(*generate_avg_vertex_degree(250, (3, 7), 0.5)))

Xval, Yval = model.createTensors(graphsValid[1], graphsValid[2])
model.add_valid(Xval, Yval, graphsValid[0], graphsValid[3])

#print(graphs)
uq=np.unique(graphs[2], return_counts=True)
print(np.unique(graphs[2], return_counts=True))
classW =  data_size/(2 * uq[1])

classWdict = {clazz:weight for clazz, weight in zip(uq[0], classW)}
model.class_weights = classWdict
print(classWdict)
adjM = graphs[0]
X = graphs[1]
Y = graphs[2]
parts = graphs[3]
X,Y = model.createTensors(X,Y)
model.fit(X, Y, adjM, parts, 1000)
