from covariant_compositional_networks_tf2.CCN_Model import CCN_Model
import tensorflow as tf
from functools import reduce
from operator import mul
from ordered_set import OrderedSet
import numpy as np
from sklearn.metrics import accuracy_score
from graphColoring import randomNPGraph, checkIfGraphConnected


channels_in = 2
feature_vector_shape=[1]
k=2
model = CCN_Model(lr =2e-4, lr_decay_rate=0.95, lr_min=3e-6, loss = tf.losses.logcosh, nonlinearity=tf.nn.tanh,
                  feature_vector_shape=feature_vector_shape, num_layers=5, k=k, channels_in=[channels_in,4,5,4,3,1])

helperVar = reduce(mul, [channels_in] + feature_vector_shape)

inp = [[tf.Variable(tf.reshape(tf.range(helperVar, dtype=tf.float32) + 1, [channels_in] + [1] * k + feature_vector_shape )),
        tf.Variable(tf.reshape(tf.range(helperVar, dtype=tf.float32) + helperVar + 1, [channels_in] + [1] * k + feature_vector_shape)),
        tf.Variable(tf.reshape(tf.range(helperVar, dtype=tf.float32) + helperVar * 2 + 1, [channels_in]+ [1] * k + feature_vector_shape))],
       # 2 feature vectors
       # for k =2 its [ [[[1,2,3]]], [[[5,6,7]]] ]
       np.array([[1, 1, 0],
                 [1, 1, 1],
                 [0, 1, 1]]),  # adjacency matrix of DIRECTED graph - node[0] will gather inputs from [0] and [1]
       # and node[1] only from [1]
       [OrderedSet([0]), OrderedSet([1]),  OrderedSet([2])]]  # parts - P(0) = {0}, and P(1) = {1} - cummulative receptive field



def complete_graph(nodes):
    node_features = [tf.Variable(tf.reshape(tf.ones(helperVar, dtype=tf.float32), [channels_in] + [1] * k + feature_vector_shape ))] * nodes
    adjM = np.ones((nodes, nodes))
    parts = [OrderedSet([i]) for i in range(nodes)]
    return [node_features, adjM, parts]

def uncomplete_graph(nodes):
    node_features = [tf.Variable(tf.reshape(tf.ones(helperVar, dtype=tf.float32), [channels_in] + [1] * k + feature_vector_shape ))] * nodes

    while True:
        adjM = randomNPGraph(nodes, 0.5, diagonal = True, undirected = True)
        if checkIfGraphConnected(adjM) and not np.array_equal(adjM, np.ones(adjM.shape)):
            break

    parts = [OrderedSet([i]) for i in range(nodes)]
    return [node_features, adjM, parts]

def gen_graphs(num, nodes, p=.5):
    graphs = []
    for _ in range(num):
        if np.random.rand()>p:
            G, y = complete_graph(nodes), tf.constant([1.])
            graphs.append(G + [y])

        else:
            G, y = uncomplete_graph(nodes), tf.constant([0.])
            graphs.append(G + [y])
    return graphs

def train_test_split(graphs, train_fraction=0.8):
    split_idx = int(np.round(train_fraction*len(graphs), decimals=0))

    g_train, g_test = graphs[:split_idx], graphs[split_idx:]

    return g_train, g_test

def data_preparation(n = 200, nodes = 4):
    graphs = gen_graphs(n, nodes)

    idxs = np.arange(len(graphs), dtype=np.int)

    return train_test_split(graphs, train_fraction=0.8)

#result = model.predict(inp[0], inp[1], inp[2])
#resultSum = np.sum(result)
#y = tf.constant([2.0])
#print(y)
#print(result)

def train_and_test(model, data, epochs=100):
    g_train, g_test = data

    for epoch in range(epochs):
        for i in range(g_train):
            features, adjM, parts, y = g_train[i]
            model.fit(features, y, adjM, parts)

    predicted = []
    target = []
    for i in range(len(g_test)):
        features, adjM, parts, y = g_train[i]
        target.append(y)
        predicted.append(model.predict(features, adjM, parts))

    print("Accuracy of the model is {}".format(accuracy_score(y_test, predicted)))

if __name__=='__main__':
    train_and_test(model, data_preparation())

# for i in range(100):
#
#     #model.fit(inp[0], y, inp[1], inp[2])
#     for i in range(len(y)):
#
#         result = model.predict(inp[0], inp[1], inp[2])
#         print(result)

#list(reversed(inp[0]))
# inpSwap = [inp[0][1], inp[0][2], inp[0][0]]
# adjMOther = np.array([[1, 1, 1], [1, 1, 1], [1,1,1]])
# result = model.predict(inp[0], adjMOther, inp[2])
# print(result)
