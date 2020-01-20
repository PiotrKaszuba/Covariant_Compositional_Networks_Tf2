from covariant_compositional_networks_tf2.CCN_Model import CCN_Model
import tensorflow as tf
from functools import reduce
from operator import mul
from ordered_set import OrderedSet
import numpy as np


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

def uncomplete_graph(nodes):
    node_features = [tf.Variable(tf.reshape(tf.ones(helperVar, dtype=tf.float32), [channels_in] + [1] * k + feature_vector_shape ))] * nodes

    below_diag =
    adjM = np.ones((nodes, nodes))
    parts = [OrderedSet([i]) for i in range(nodes)]

def gen_graphs(num, nodes, p):
    graphs = []
    Ys = []
    for _ in range(num):
        if np.random.rand()>p:
            return complete_graph(nodes), tf.constant([1.])
        else:
            return uncomplete_graph(nodes), tf.constant([0.])

full_graph = inp
full_graph[1] =
print(inp)
result = model.predict(inp[0], inp[1], inp[2])
#resultSum = np.sum(result)
y = tf.constant([2.0])
#print(y)
print(result)

for i in range(100):

    model.fit(inp[0], y, inp[1], inp[2])

    result = model.predict(inp[0], inp[1], inp[2])
    print(result)

#list(reversed(inp[0]))
inpSwap = [inp[0][1], inp[0][2], inp[0][0]]
adjMOther = np.array([[1, 1, 1], [1, 1, 1], [1,1,1]])
result = model.predict(inp[0], adjMOther, inp[2])
print(result)
