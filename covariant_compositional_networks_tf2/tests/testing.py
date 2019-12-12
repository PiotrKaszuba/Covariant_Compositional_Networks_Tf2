from functools import reduce
from operator import mul
from string import ascii_lowercase
from itertools import combinations
import numpy as np
import tensorflow as tf
from ordered_set import OrderedSet

# These are constants:
# max k = 9
ascii_chi = ascii_lowercase[:9]
ascii_f = ascii_lowercase[9:18]
# max feature vector dimensions = 7
ascii_feature_vector = ascii_lowercase[18:25]

ascii_channels_in = ascii_lowercase[25]

ascii_contractions = ascii_f

# Functions:

# One way of getting X (chi) matrix - it doesn't work
# def permutationMatrixOnNumbers_AToB(a, b):
#      new_len = len(b)
#      #a = a | [0]*(len(b) - len(a))
#      A = np.zeros((new_len, new_len))
#      A[np.argsort(a), np.arange(len(a))] = 1
#      B = np.zeros((new_len, new_len))
#      B[np.argsort(b), np.arange(new_len)] = 1
#      return A@B.transpose()

# Different way of getting X (chi) matrix
def alternativePermutationMatrix_AToB(a, b):
    return np.array(np.apply_along_axis(lambda x, y: np.in1d(y, x), 0, np.expand_dims(a, 0),
                                        np.expand_dims(b, 0)), dtype=int)


# a function that returns appropriate promotion einsum expression for a given k and feature vector shape
def getEinsumExpression(k, feature_vector_shape, channels_in = None):
    # if k==1:
    #      return 'ai,if->af'
    # if k==2:
    #     #      return 'ai,bj,ijf->abf'
    str_to_join = []
    for i in range(k):
        str_to_join.append(ascii_chi[i]) # chi_dim1
        str_to_join.append(ascii_f[i]) #chi_dim2
        str_to_join.append(',')
    if channels_in is not None:
        str_to_join.append(ascii_channels_in) # channels in
    for i in range(k):
        str_to_join.append(ascii_f[i]) # p tensor dims
    for i in range(len(feature_vector_shape)):
        str_to_join.append(ascii_feature_vector[i]) # feature vector

    str_to_join.append('->')
    if channels_in is not None:
        str_to_join.append(ascii_channels_in) # promotion saved channels
    for i in range(k):
        str_to_join.append(ascii_chi[i]) # promotion k dims
    for i in range(len(feature_vector_shape)):
        str_to_join.append(ascii_feature_vector[i]) # promotion saved feature vector dims


    return ''.join(str_to_join)


# def recurseNumberCombinationsSummingUpToN(arr, index, num,
#                               reducedNum):
#
#     if (reducedNum < 0):
#         return;
#
#
#     if (reducedNum == 0):
#
#         for i in range(index):
#             print(arr[i], end=" ");
#         print("");
#         return;
#
#
#     prev = 1 if (index == 0) else arr[index - 1]
#
#
#     for k in range(prev, num + 1):
#         arr[index] = k
#         recurseNumberCombinationsSummingUpToN(arr, index + 1, num,
#                                   reducedNum - k)
#
# def numberCombinationsSummingUpToN(n):
#     recurseNumberCombinationsSummingUpToN(np.zeros(shape=[n], dtype=np.int32), 0, n, n)
#
#
# numberCombinationsSummingUpToN(5)
#
# def getContractions(k, k_increase, feature_vector_shape):
#     contract_symbols = ascii_contractions[:k+k_increase]
#     combinations_of_out_symbols = combinations(contract_symbols, k)
#
#     for combination_out in combinations_of_out_symbols:
#         combinations_of_inp_symbols = numberCombinationsSummingUpToN(5)
#         for combination_inp in combinations_of_inp_symbols:
#
#
#     str_to_join = []
#     for i in range(k+k_increase):
#         str_to_join.append(ascii_contractions[i])
#     for i in range(len(feature_vector_shape)):
#         str_to_join.append(feature_vector_shape)
#     str_to_join.append('->')


def getActivationEinsum(k, feature_vector_shape, channels_in=None):
    str_to_join = []
    str_to_join.append(ascii_chi[1])  # channels_out
    if channels_in is not None:
        str_to_join.append(ascii_channels_in) # channels in
    str_to_join.append(ascii_chi[0]) # qs
    for i in range(len(feature_vector_shape)):
        str_to_join.append(ascii_feature_vector[i]) #feature vectors

    str_to_join.append(',')
    if channels_in is not None:
        str_to_join.append(ascii_channels_in) # channels_in
    str_to_join.append(ascii_chi[0]) #qs
    for i in range(k):
        str_to_join.append(ascii_f[i]) #p tensor dims
    for i in range(len(feature_vector_shape)):
        str_to_join.append(ascii_feature_vector[i]) #feature vectors
    str_to_join.append('->')
    for i in range(k):
        str_to_join.append(ascii_f[i]) # p tensor dims
    str_to_join.append(ascii_chi[1])  # channels out
    for i in range(len(feature_vector_shape)):
        str_to_join.append(ascii_feature_vector[i]) # feature vectors
    return ''.join(str_to_join)

def getContractions(k, k_increase, feature_vector_shape, channels_in = None):
    str_to_join = []
    str_feature_vector = []
    if channels_in is not None:
        str_to_join.append(ascii_channels_in) # channels in
    for i in range(k+k_increase):
        str_to_join.append(ascii_contractions[i]) # all dims to contract
    for i in range(len(feature_vector_shape)):
        str_to_join.append(ascii_feature_vector[i]) # feature vectors
        str_feature_vector.append(ascii_feature_vector[i])
    str_to_join.append('->')
    str_to_join.append(ascii_channels_in) # channels_in
    contract_symbols = ascii_contractions[:k + k_increase]
    combinations_of_out_symbols = combinations(contract_symbols, k)
    expressions = [''.join(str_to_join)+''.join(combination)+''.join(str_feature_vector) for combination in combinations_of_out_symbols]
    return expressions



# these are constructor / build params:
k = 2
feature_vector_shape = [1]
channels_out = 1
channels_in = 1
permutationFunction = alternativePermutationMatrix_AToB  # Choose which permutation function to use (same results)
einsum_expr = getEinsumExpression(k, feature_vector_shape, channels_in)  # promotion expression
contractions_expressions = getContractions(k, 1, feature_vector_shape, channels_in)
einsum_activation = getActivationEinsum(k, feature_vector_shape, channels_in)
nonlinearity = tf.nn.relu

activations_shape_len = 1 +k + len(feature_vector_shape)
activations_channels_position_to_swap = k
activation_swap_channels_list = [activations_channels_position_to_swap] + [i for i in range(activations_shape_len) if i != activations_channels_position_to_swap]
# W = tf.Variable(tf.random.uniform([channels_out, channels_in] + [len(contractions_expressions)] + feature_vector_shape, minval=-1))
# bias = tf.Variable(tf.random.uniform([channels_out] + feature_vector_shape, minval=-1))
W = tf.Variable(tf.ones([channels_out, channels_in] + [len(contractions_expressions)] + feature_vector_shape))
bias = tf.Variable(tf.ones([channels_out] + feature_vector_shape))

# This is an exemplary layer input
helperVar = reduce(mul, [channels_in]+feature_vector_shape)
#
inp = [[tf.Variable(tf.reshape(tf.range(helperVar,dtype=tf.float32) + 1, [channels_in] + [1] * k + feature_vector_shape )),
        tf.Variable(tf.reshape(tf.range(helperVar,dtype=tf.float32) + helperVar+1, [channels_in] + [1] * k + feature_vector_shape)),
        tf.Variable(tf.reshape(tf.range(helperVar, dtype=tf.float32) + helperVar*2+1,
                               [channels_in]+ [1] * k + feature_vector_shape))],
       # 2 feature vectors
       # for k =2 its [ [[[1,2,3]]], [[[5,6,7]]] ]
       np.array([[1, 1, 0], [1, 1, 1], [0,1,1]]),  # adjacency matrix of DIRECTED graph - node[0] will gather inputs from [0] and [1]
       # and node[1] only from [1]
       [OrderedSet([0]), OrderedSet([1]),  OrderedSet([2])]]  # parts - P(0) = {0}, and P(1) = {1} - cummulative receptive field
with tf.GradientTape() as gt:
    gt.watch(inp[0])
    # take the input and simulate layer behaviour:

    print(inp[0])
    # Organizing inputs
    # only the first (tensors) should pass gradients to update W's
    # only the first and the third (tensors, parts) will change when propagating throughout network
    # parts will accumulate with receptive fields
    # tensors are activations from previous layer
    # adjM is constant 2D square matrix - used to retrieve number of neurons by its size
    # and to gather neurons to define new layer parts based on children parts
    tensors, adjM, parts = inp

    # extract number of neurons from adjM number of rows (adjM is 2D square matrix)
    # this is here for option to decrease number of neurons in the following layers by shrinking adjM
    # e.g. neurons over leaf nodes in graph
    num_neurons = len(adjM)

    # contains information which neurons to gather signal from (for every neuron list)
    receptive_fields = [tf.where(adjM[i] == 1)[:, 0] for i in range(num_neurons)]

    # new, cumulative receptive fields (parts) based on adjM (for every neuron in current layer)
    # for every neuron i;
    # parts of every neuron in the receptive field of 'i' are reduced with union to get cumulative receptive fields
    new_parts = [reduce(OrderedSet.union, [parts[tensor_child_index] for tensor_child_index in receptive_fields[i]])
                 for i in range(num_neurons)]



    #a_tensor = tf.convert_to_tensor(permutationFunction(parts[0], new_parts[0]) , dtype=tf.float32)

    #one_prom = tf.einsum(einsum_expr, *([a_tensor]*k + [tensors[0]] ) )
    # for every neuron i;
    # create promotion chi matrix for every neuron/node in i's receptive field
    chis = [{tensor_child_index.numpy(): tf.convert_to_tensor(permutationFunction(parts[tensor_child_index], new_parts[i]), dtype=tf.float32)
             for tensor_child_index in receptive_fields[i]}
            for i in range(num_neurons)]

    # for every neuron i;
    # promote every activation of nodes in i's receptive field
    # IMPORTANT:
    # (probably) This is where tf functions should start to be used because new structures are formed based on previous ones
    # and these new structures will ultimately 'transform' and mix with W to create activations
    promotions = [[tf.einsum(einsum_expr, *([chis[i][tensor_child_index.numpy()]] * k + [tensors[tensor_child_index]]))
                   for tensor_child_index in receptive_fields[i]]
                  for i in range(num_neurons)]

    stacked = [tf.stack(promotions[i], axis=1) for i in range(num_neurons)]

    qs = [tf.stack([tf.einsum(expression, stacked[i]) for expression in contractions_expressions], axis=1) for i in range(num_neurons)]
    f = 0
    activations = [
        tf.transpose(
             nonlinearity(
                                    tf.add(tf.einsum(einsum_activation, W, qs[neuron_ind]), bias)

                        ),
                    activation_swap_channels_list)
     for neuron_ind in range(num_neurons)]

    print(qs[0])
    print(activations)




gradient = gt.gradient(activations, bias)



print(gradient)
# from covariant_compositional_networks_tf2.CCN_Layer import CCN_Layer
# tf.executing_eagerly()
#
# l1 = CCN_Layer()
# l2 = CCN_Layer()([l1, 'l'])
# model = tf.keras.Model(inputs=l1, outputs=l2, dynamic=True)



