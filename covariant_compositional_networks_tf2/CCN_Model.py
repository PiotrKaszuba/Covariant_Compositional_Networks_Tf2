import tensorflow as tf
from covariant_compositional_networks_tf2.CCNLayer import CCN_Layer
import numpy as np
from functools import reduce
from operator import add
import pickle
class CCN_Model:
    # channels in should be num_layers + 1 length and the last value will actually be channels out from CCN model
    def __init__(self, loss, optimizer, num_layers = 2, k = 2, feature_vector_shape= [1], channels_in = [1,1,1], dense_channels_out = 1, dense_activation=tf.keras.activations.sigmoid, nonlinearity = tf.nn.relu, weight_init = 'uniform', neuron_final_aggregation = tf.reduce_mean, batch_update_size=10, l1_reg = 0.0, save_every=10, mix_promotions_with_adjM=True):
        self.num_layers = num_layers
        self.k = k
        self.feature_vector_shape = feature_vector_shape
        self.channels_in = channels_in
        self.dense_channels_out = dense_channels_out
        self.dense_activation = dense_activation
        self.nonlinearity = nonlinearity
        self.weight_init = weight_init
        self.loss = loss
        self.mix_promotions_with_adjM = mix_promotions_with_adjM

        self.build_model()

        self.optimizer = optimizer

        self.neuron_final_aggregation = neuron_final_aggregation
        self.batch_update_size = batch_update_size
        self.l1_reg = l1_reg
        #self.beta =momentum

        self.initialize_weights(self.weight_init)
        self.class_weights = {}
        self.save_every = save_every
        self.Xval = None
    def initialize_weights(self, type_init):
        # self.momentumW = tf.zeros([self.channels_out, self.channels_in] + [len(self.contractions_expressions)] + self.feature_vector_shape)
        # self.momentumB = tf.zeros([self.channels_out] + self.feature_vector_shape)
        if type_init == 'uniform':
            self.W = tf.Variable(
                tf.random.uniform([self.channels_in[-1], self.dense_channels_out],
                                  minval=-1)/5)
            self.bias = tf.Variable(tf.random.uniform([self.dense_channels_out], minval=-1)/5)
            return
        raise NotImplementedError

    def build_model(self):
        self.layers = [CCN_Layer(self.k, self.feature_vector_shape, self.channels_in[i], self.channels_in[i+1], self.nonlinearity, self.weight_init, mix_promotions_with_adjM=self.mix_promotions_with_adjM)
                       for i in range(self.num_layers)]

    def shuffle(self, X,Y,adjM,parts):
        to_shuffle = list(zip(X,Y,adjM,parts))
        np.random.shuffle(to_shuffle)
        X,Y,adjM,parts = zip(*to_shuffle)
        return X,Y,adjM,parts

    def predict(self, X, adjM, parts):
        for i in range(self.num_layers):
            X, adjM, parts = self.layers[i].call(X,adjM,parts)

        X = [tf.math.reduce_mean(x, range(1,len(tf.shape(x)))) for x in X]
        X = self.neuron_final_aggregation(X, 0)
        X=tf.reshape(X, (self.channels_in[-1],))

        out = tf.add(tf.einsum('n,nm->m', X, self.W), self.bias)

        out = self.dense_activation(out)
        return out

    def createTensors(self,X, Y):
        X = [[tf.reshape(
            tf.cast(tf.Variable(n), tf.float32), [self.channels_in[0]] + [1] * self.k + self.feature_vector_shape)
            for n in x]
         for x in X]
        Y = [tf.cast(tf.Variable(y), tf.float32) for y in Y]
        return X,Y

    def add_valid(self, X,Y,adjM,parts):
        self.Xval = X
        self.Yval = Y
        self.adjMval = adjM
        self.partsval = parts
    def validate_acc_binary(self):
        if self.Xval is None:
            return
        tests = 0
        true = 0
        for x, matrix, part, y in zip(self.Xval, self.adjMval, self.partsval, self.Yval):
            Y_pred = self.predict(x, matrix, part)
            pred = tf.cast((Y_pred > 0.5), tf.int32)
            if pred == tf.cast(y, tf.int32):
                true+=1
            tests+=1
            if tests %10 == 0:
                print('validating.. ' + str(tests))
        print('validate accuracy: ' + str(true/tests))

    def fit(self, X, Y, adjM, parts, num_epochs):

        model_params = [self.W, self.bias] + reduce(add,
                              [[self.layers[q].W, self.layers[q].bias] for q in range(self.num_layers)])

        grad_id = 0
        grads_list = []
        for i in range(num_epochs):

            if i%self.save_every == self.save_every-1:
                params = [param.numpy() for param in model_params]
                with open('params', 'wb') as file:
                    pickle.dump(params, file)

            self.validate_acc_binary()
            X, Y, adjM, parts = self.shuffle(X,Y,adjM,parts)


            print("epoch: " +str(i))
            data_num = 0
            total_loss = 0

            for x, matrix, part, y in zip(X,adjM, parts, Y):
                weight = self.class_weights.get(y.numpy())
                if weight is None:
                    weight = 1.0
                max_grad = -np.inf

                with tf.GradientTape(persistent= True) as gt:
                    gt.watch(x)
                    Y_pred = self.predict(x, matrix, part)
                    reg_loss = self.l1_reg * tf.reduce_sum([tf.reduce_sum(tf.abs(param)) for param in model_params] )
                    loss = self.loss(Y_pred, y) + reg_loss
                    total_loss += loss

                    grads = gt.gradient(loss, model_params)
                    grads = list(map(lambda x: x*weight, grads))
                    grads_list.append(grads)
                    grad_id += 1

                if grad_id >= self.batch_update_size:
                    grad_id = 0
                    summed = list(zip(*grads_list))
                    summed = list(map(lambda x: tf.reduce_sum(x, axis=0), summed))
                    self.optimizer.apply_gradients(zip(summed, model_params)) ## TO CHECK
                    grads_list.clear()
                print("result: " + str(Y_pred) + ", loss " + str(data_num) + ": " + str(loss)+ ", reg_loss: " + str(reg_loss))
                data_num +=1
            # self.lr = self.lr * self.lr_decay_rate
            # if self.lr < self.lr_min:
            #     self.lr = self.lr_min
            print("epoch total loss: " +str(total_loss))