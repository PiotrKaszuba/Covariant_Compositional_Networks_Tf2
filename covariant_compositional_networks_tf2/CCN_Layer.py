import tensorflow as tf
from functools import reduce
from operator import mul
from string import ascii_lowercase

import numpy as np
import tensorflow as tf
from ordered_set import OrderedSet

# These are constants:
# max k = 9
ascii_chi = ascii_lowercase[:9]
ascii_f = ascii_lowercase[9:18]
# max feature vector dimensions = 8
ascii_feature_vector = ascii_lowercase[18:]

def alternativePermutationMatrix_AToB(a, b):
    return np.array(np.apply_along_axis(lambda x, y: np.in1d(y, x), 0, np.expand_dims(a, 0),
                                        np.expand_dims(b, 0)), dtype=int)


# a function that returns appropriate promotion einsum expression for a given k and feature vector shape
def getEinsumExpression(k, feature_vector_shape):
    # if k==1:
    #      return 'ai,if->af'
    # if k==2:
    #      return 'ai,bj,ijf->abf'
    str_to_join = []
    for i in range(k):
        str_to_join.append(ascii_chi[i])
        str_to_join.append(ascii_f[i])
        str_to_join.append(',')
    for i in range(k):
        str_to_join.append(ascii_f[i])
    for i in range(len(feature_vector_shape)):
        str_to_join.append(ascii_feature_vector[i])
    str_to_join.append('->')
    for i in range(k):
        str_to_join.append(ascii_chi[i])
    for i in range(len(feature_vector_shape)):
        str_to_join.append(ascii_feature_vector[i])

    return ''.join(str_to_join)

# THIS IS NOT IMPLEMENTED YET
# Check tests/testing.py

class CCN_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(CCN_Layer, self).__init__()
        self.k = 2
        self.feature_vector_shape = [3]
        self.permutationFunction = alternativePermutationMatrix_AToB  # Choose which permutation function to use (same results)
        self.einsum_expr = getEinsumExpression(self.k, self.feature_vector_shape)  # promotion expression

    def build(self, input_shape):
        # add weights
        pass
        #raise NotImplementedError

        # self.kernel = self.add_variable("kernel",
        #                                 shape=[int(input_shape[-1]),
        #                                        self.num_outputs])
    @tf.function
    def call(self, input):
        # Organizing inputs
        # only the first (tensors) should pass gradients to update W's
        # only the first and the third (tensors, parts) will change when propagating throughout network
        # parts will accumulate with receptive fields
        # tensors are activations from previous layer
        # adjM is constant 2D square matrix - used to retrieve number of neurons by its size
        # and to gather neurons to define new layer parts based on children parts
        tensors, adjM, parts = input

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

        a_tensor = tf.convert_to_tensor(self.permutationFunction(parts[0], new_parts[0]), dtype=tf.float32)

        one_prom = tf.einsum(self.einsum_expr, *([a_tensor] * k + [tensors[0]]))
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
            [tf.einsum(self.einsum_expr, *([chis[i][tensor_child_index.numpy()]] * k + [tensors[tensor_child_index]]))
             for tensor_child_index in receptive_fields[i]]
            for i in range(num_neurons)]

        # print(promotions)
        return [promotions, adjM, new_parts]
