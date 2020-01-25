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


    def __init__(self, k = 2, feature_vector_shape = [1], channels_in = 1, channels_out = 1, nonlinearity = tf.nn.relu, weights_init = 'uniform', mix_promotions_with_adjM = True ):
        self.k = k
        self.feature_vector_shape = feature_vector_shape
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.nonlinearity = nonlinearity

        self.permutationFunction = self.alternativePermutationMatrix_AToB  # Choose which permutation function to use (same results)
        self.einsum_expr = self.getEinsumExpression(k, feature_vector_shape, channels_in)  # promotion expression

        self.einsum_activation = self.getActivationEinsum(k, feature_vector_shape, channels_in)
        activations_shape_len = 1 + k + len(feature_vector_shape)
        activations_channels_position_to_swap = k
        self.activation_swap_channels_list = [activations_channels_position_to_swap] + [i for i in
                                                                                   range(activations_shape_len) if
                                                                                   i != activations_channels_position_to_swap]
        self.tensordot_swap_channel_list = [0, 1] + (np.arange(k)+2).tolist() + [2+k+len(feature_vector_shape), 3+k+len(feature_vector_shape)] + (np.arange(len(feature_vector_shape))+ 2+k).tolist()

        self.mix_promotions_with_adjM = mix_promotions_with_adjM
        self.contractions_expressions, self.contractions_add_operators = self.getContractions(k, 1 + (2 if mix_promotions_with_adjM else 0), feature_vector_shape, channels_in)
        self.initialize_weights(weights_init)

        self.operandDict = {'two_one' : {} , 'three' : {}}
    def initialize_weights(self, type_init):
        # self.momentumW = tf.zeros([self.channels_out, self.channels_in] + [len(self.contractions_expressions)] + self.feature_vector_shape)
        # self.momentumB = tf.zeros([self.channels_out] + self.feature_vector_shape)
        if type_init == 'uniform':
            self.W = tf.Variable(
                tf.random.uniform([self.channels_out, self.channels_in] + [len(self.contractions_expressions)] + self.feature_vector_shape,
                                  minval=-1)/5)
            self.bias = tf.Variable(tf.random.uniform([self.channels_out] + self.feature_vector_shape, minval=-1)/5)
            return
        raise NotImplementedError


    def call(self, X, adjM, parts):
        #print("ok")
        # extract number of neurons from adjM number of rows (adjM is 2D square matrix)
        # this is here for option to decrease number of neurons in the following layers by shrinking adjM
        # e.g. neurons over leaf nodes in graph
        num_neurons = len(adjM)

        # contains information which neurons to gather signal from (for every neuron list)
        receptive_fields = [tf.where(adjM[i] == 1)[:, 0] for i in range(num_neurons)]

        # if self.q_dim_security == 'adjM':
        #     newAdjM = np.array(adjM)
        #     indices = list(zip(*np.where(adjM == 1)))
        #     for row,col in indices:
        #         newAdjM[row] += adjM[col]
        #     oldAdjM = adjM
        #     adjM=np.clip(newAdjM, a_min=0, a_max=1)
        #
        #     new_parts_delta = [
        #         OrderedSet(np.where(oldAdjM[i] == 1)[0].tolist())
        #         for i in range(num_neurons)]
        #     new_parts = [OrderedSet.union(parts[i], new_parts_delta[i]) for i in range(num_neurons)]
        # else:





        # new, cumulative receptive fields (parts) based on adjM (for every neuron in current layer)
        # for every neuron i;
        # parts of every neuron in the receptive field of 'i' are reduced with union to get cumulative receptive fields

        # we iter through parts so that initial order of parts is preserved (because it is left operand of union)
        # we do only if it is part of receptive field so that original logic is kept
        # also we use union with receptive field to make sure the first step has receptive field inside parts (further unions wont change a thing)
        new_parts = [
            reduce(OrderedSet.union, [parts[neighbour_idx] for neighbour_idx in OrderedSet.union(parts[node_idx], receptive_fields[node_idx].numpy().tolist()) if neighbour_idx in receptive_fields[node_idx]])
            for node_idx in range(num_neurons)]


        # for every neuron i;
        # create promotion chi matrix for every neuron/node in i's receptive field
        chis = [{neighbour_idx.numpy(): tf.convert_to_tensor(
            self.permutationFunction(parts[neighbour_idx], new_parts[i]), dtype=tf.float32)
                 for neighbour_idx in receptive_fields[i]}
                for i in range(num_neurons)]

        # for every neuron i;
        # promote every activation of nodes in i's receptive field
        # IMPORTANT:
        # (probably) This is where tf functions should start to be used because new structures are formed based on previous ones
        # and these new structures will ultimately 'transform' and mix with W to create activations

        # we use new_parts for neighbour_idx (even there might be no promotion) so that we can gather promotions
        # and fill 0's for next step in appropriate order
        promotions = [
            [tf.einsum(self.einsum_expr, *([chis[i][neighbour_idx]] * self.k + [X[neighbour_idx]]))
             if neighbour_idx in receptive_fields[i] else tf.zeros([self.channels_in] + [len(new_parts[i])] * self.k + self.feature_vector_shape )
             for neighbour_idx in new_parts[i]]
            for i in range(num_neurons)]
        ##
        stacked = [tf.stack(promotions[i], axis=1) for i in range(num_neurons)]

        if self.mix_promotions_with_adjM:
            adjMs_forNodes = [tf.cast(tf.constant(adjM[tuple([new_parts[i]])][:,new_parts[i]]), 'float32') for i in range(num_neurons)]
            stacked = [ tf.transpose(
                tf.tensordot(stacked[i], adjMs_forNodes[i], axes=0),  self.tensordot_swap_channel_list)
            for i in range(num_neurons)]


        #temp=[[tf.einsum(expression, stacked[i]) for expression in self.contractions_expressions] for i in range(num_neurons)]
        qs = [tf.stack([tf.einsum(expression, *([stacked[i]] + operator(len(new_parts[i])) ) ) for expression, operator in zip(self.contractions_expressions, self.contractions_add_operators)], axis=1)
              for i in range(num_neurons)]

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

    def get_or_create_operand(self, name, func, x):
        operand = self.operandDict[name].get(x)
        if operand is None:
            operand = func(x)
            self.operandDict[name][x] = operand
        return operand

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
        add_operators = [lambda x: [] for i in range(len(expressions))]

        if k_increase == 3:

            str_to_join = []
            str_feature_vector = []
            if channels_in is not None:
                str_to_join.append(self.ascii_channels_in)  # channels in
            for i in range(k + k_increase):
                str_to_join.append(self.ascii_contractions[i])  # all dims to contract
            for i in range(len(feature_vector_shape)):
                str_to_join.append(self.ascii_feature_vector[i])  # feature vectors
                str_feature_vector.append(self.ascii_feature_vector[i])
            str_to_join.append(',')
            to_join = ''.join(str_to_join)
            str_to_join2 = []
            str_to_join2.append('->')
            str_to_join2.append(self.ascii_channels_in)
            contract_symbols = self.ascii_contractions[:k + k_increase]
            combinations_of_out_symbols = list(combinations(contract_symbols, k))
            two_one_operand_creation = lambda x: [tf.cast(tf.einsum('ab,bc->abc', tf.ones([x, x]), tf.eye(x)), tf.float32)]
            three_operand_creation = lambda x: [tf.cast( tf.einsum('ab,bc->abc', tf.eye(x), tf.eye(x)), tf.float32)]

            two_one_operand = lambda x: self.get_or_create_operand('two_one', two_one_operand_creation, x)
            three_operand = lambda x: self.get_or_create_operand('three', three_operand_creation, x)

            symbols_order = [[0,1,2], [1,0,2], [2,0,1], [0,1,2]]
            operands_order = [two_one_operand, two_one_operand, two_one_operand, three_operand]
            for combin in combinations_of_out_symbols:
                to_join2 = ''.join(str_to_join2) + ''.join(combin) + ''.join(str_feature_vector)

                symbols_left = [symb for symb in contract_symbols if symb not in combin]
                for order, operand in zip(symbols_order, operands_order):
                    inner_join = []
                    for idx in order:
                        inner_join.append(symbols_left[idx])
                    expressions.append(to_join+ ''.join(inner_join) + to_join2)
                    add_operators.append(operand)





            # str_to_join = []
            # str_feature_vector = []
            # if channels_in is not None:
            #     str_to_join.append(self.ascii_channels_in)  # channels in
            #
            # str_to_join2 = []
            # for i in range(len(feature_vector_shape)):
            #     str_to_join2.append(self.ascii_feature_vector[i])  # feature vectors
            #     str_feature_vector.append(self.ascii_feature_vector[i])
            # str_to_join2.append('->')
            # str_to_join2.append(self.ascii_channels_in)  # channels_in
            # contract_symbols = self.ascii_contractions[:k]
            # for symb in contract_symbols:
            #     str_to_join2.append(symb)
            # str_to_join2 = str_to_join2 +str_feature_vector
            #
            # contract_symbols_combinations = self.ascii_contractions[k:k+k_increase-1]
            # channels_to_contract = list(range(k+k_increase))
            # combs = combinations(channels_to_contract, k_increase)
            # for comb in combs:
            #     for inner_comb_val in comb:
            #         str_inner = []
            #         chars_idx = 0
            #         for i in range(k+k_increase):
            #             if i in comb:
            #                 if i == inner_comb_val:
            #                     str_inner.append( contract_symbols_combinations[-1])
            #                 else:
            #                     str_inner.append(contract_symbols_combinations[0])
            #             else:
            #                 str_inner.append(contract_symbols[chars_idx])
            #                 chars_idx+=1
            #
            #         expressions.append(''.join(str_to_join) + ''.join(str_inner) + ''.join(str_to_join2) )
            #
            # combs = combinations(channels_to_contract, k_increase)
            # for comb in combs:
            #     str_inner = []
            #     chars_idx = 0
            #     for i in range(k+k_increase):
            #         if i in comb:
            #             str_inner.append(contract_symbols_combinations[0])
            #         else:
            #             str_inner.append(contract_symbols[chars_idx])
            #             chars_idx+=1
            #     expressions.append(''.join(str_to_join) + ''.join(str_inner) + ''.join(str_to_join2))







        #expressions = [''.join(str_to_join) + ''.join(combination) + ''.join(str_feature_vector)]

        return expressions, add_operators
