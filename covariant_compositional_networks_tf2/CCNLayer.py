from functools import reduce
from operator import mul
from string import ascii_lowercase
from itertools import combinations
import numpy as np
import tensorflow as tf
from ordered_set import OrderedSet

class CCN_Layer:


    # These are constants:
    # max k = 9
    ascii_chi = ascii_lowercase[:9]
    ascii_f = ascii_lowercase[9:18]
    # max feature vector dimensions = 7
    ascii_feature_vector = ascii_lowercase[18:25]

    ascii_channels_in = ascii_lowercase[25]

    ascii_contractions = ascii_f

    def __init__(self, k = 2, feature_vector_shape = [1], channels_in = 1, channels_out = 1, nonlinearity = tf.nn.relu, weights_init = 'uniform'):
        self.k = k
        self.feature_vector_shape = feature_vector_shape
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.nonlinearity = nonlinearity

        self.permutationFunction = self.alternativePermutationMatrix_AToB  # Choose which permutation function to use (same results)
        self.einsum_expr = self.getEinsumExpression(k, feature_vector_shape, channels_in)  # promotion expression
        self.contractions_expressions = self.getContractions(k, 1, feature_vector_shape, channels_in)
        self.einsum_activation = self.getActivationEinsum(k, feature_vector_shape, channels_in)
        activations_shape_len = 1 + k + len(feature_vector_shape)
        activations_channels_position_to_swap = k
        self.activation_swap_channels_list = [activations_channels_position_to_swap] + [i for i in
                                                                                   range(activations_shape_len) if
                                                                                   i != activations_channels_position_to_swap]

        self.initialize_weights(weights_init)
    def initialize_weights(self, type_init):
        if type_init == 'uniform':
            self.W = tf.Variable(
                tf.random.uniform([self.channels_out, self.channels_in] + [len(self.contractions_expressions)] + self.feature_vector_shape,
                                  minval=-1))
            self.bias = tf.Variable(tf.random.uniform([self.channels_out] + self.feature_vector_shape, minval=-1))
            return
        raise NotImplementedError


    def call(self, X, adjM, parts):
        print("ok")
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

        # a_tensor = tf.convert_to_tensor(permutationFunction(parts[0], new_parts[0]) , dtype=tf.float32)

        # one_prom = tf.einsum(einsum_expr, *([a_tensor]*k + [tensors[0]] ) )
        # for every neuron i;
        # create promotion chi matrix for every neuron/node in i's receptive field
        chis = [{tensor_child_index.numpy(): tf.convert_to_tensor(
            self.permutationFunction(parts[tensor_child_index], new_parts[i]), dtype=tf.float32)
                 for tensor_child_index in receptive_fields[i]}
                for i in range(num_neurons)]

        # for every neuron i;
        # promote every activation of nodes in i's receptive field
        # IMPORTANT:
        # (probably) This is where tf functions should start to be used because new structures are formed based on previous ones
        # and these new structures will ultimately 'transform' and mix with W to create activations
        promotions = [
            [tf.einsum(self.einsum_expr, *([chis[i][tensor_child_index.numpy()]] * self.k + [X[tensor_child_index]]))
             for tensor_child_index in receptive_fields[i]]
            for i in range(num_neurons)]

        stacked = [tf.stack(promotions[i], axis=1) for i in range(num_neurons)]
        #temp=[[tf.einsum(expression, stacked[i]) for expression in self.contractions_expressions] for i in range(num_neurons)]
        qs = [tf.stack([tf.einsum(expression, stacked[i]) for expression in self.contractions_expressions], axis=1) for i in
              range(num_neurons)]
        activations = [
            tf.transpose(
                self.nonlinearity(
                    tf.add(tf.einsum(self.einsum_activation, self.W, qs[neuron_ind]), self.bias)

                ),
                self.activation_swap_channels_list)
            for neuron_ind in range(num_neurons)]

        return activations, adjM, new_parts


    def alternativePermutationMatrix_AToB(self, a, b):
        return np.array(np.apply_along_axis(lambda x, y: np.in1d(y, x), 0, np.expand_dims(a, 0),
                                            np.expand_dims(b, 0)), dtype=int)

    # a function that returns appropriate promotion einsum expression for a given k and feature vector shape
    def getEinsumExpression(self, k, feature_vector_shape, channels_in=None):
        # if k==1:
        #      return 'ai,if->af'
        # if k==2:
        #     #      return 'ai,bj,ijf->abf'
        str_to_join = []
        for i in range(k):
            str_to_join.append(self.ascii_chi[i])  # chi_dim1
            str_to_join.append(self.ascii_f[i])  # chi_dim2
            str_to_join.append(',')
        if channels_in is not None:
            str_to_join.append(self.ascii_channels_in)  # channels in
        for i in range(k):
            str_to_join.append(self.ascii_f[i])  # p tensor dims
        for i in range(len(feature_vector_shape)):
            str_to_join.append(self.ascii_feature_vector[i])  # feature vector

        str_to_join.append('->')
        if channels_in is not None:
            str_to_join.append(self.ascii_channels_in)  # promotion saved channels
        for i in range(k):
            str_to_join.append(self.ascii_chi[i])  # promotion k dims
        for i in range(len(feature_vector_shape)):
            str_to_join.append(self.ascii_feature_vector[i])  # promotion saved feature vector dims

        return ''.join(str_to_join)

    def getActivationEinsum(self, k, feature_vector_shape, channels_in=None):
        str_to_join = []
        str_to_join.append(self.ascii_chi[1])  # channels_out
        if channels_in is not None:
            str_to_join.append(self.ascii_channels_in)  # channels in
        str_to_join.append(self.ascii_chi[0])  # qs
        for i in range(len(feature_vector_shape)):
            str_to_join.append(self.ascii_feature_vector[i])  # feature vectors

        str_to_join.append(',')
        if channels_in is not None:
            str_to_join.append(self.ascii_channels_in)  # channels_in
        str_to_join.append(self.ascii_chi[0])  # qs
        for i in range(k):
            str_to_join.append(self.ascii_f[i])  # p tensor dims
        for i in range(len(feature_vector_shape)):
            str_to_join.append(self.ascii_feature_vector[i])  # feature vectors
        str_to_join.append('->')
        for i in range(k):
            str_to_join.append(self.ascii_f[i])  # p tensor dims
        str_to_join.append(self.ascii_chi[1])  # channels out
        for i in range(len(feature_vector_shape)):
            str_to_join.append(self.ascii_feature_vector[i])  # feature vectors
        return ''.join(str_to_join)

    def getContractions(self, k, k_increase, feature_vector_shape, channels_in=None):
        str_to_join = []
        str_feature_vector = []
        if channels_in is not None:
            str_to_join.append(self.ascii_channels_in)  # channels in
        for i in range(k + k_increase):
            str_to_join.append(self.ascii_contractions[i])  # all dims to contract
        for i in range(len(feature_vector_shape)):
            str_to_join.append(self.ascii_feature_vector[i])  # feature vectors
            str_feature_vector.append(self.ascii_feature_vector[i])
        str_to_join.append('->')
        str_to_join.append(self.ascii_channels_in)  # channels_in
        contract_symbols = self.ascii_contractions[:k + k_increase]
        combinations_of_out_symbols = combinations(contract_symbols, k)
        expressions = [''.join(str_to_join) + ''.join(combination) + ''.join(str_feature_vector) for combination in
                       combinations_of_out_symbols]
        return expressions
