import tensorflow as tf
from covariant_compositional_networks_tf2.CCNLayer import CCN_Layer
import numpy as np
class CCN_Model:
    # channels in should be num_layers + 1 length and the last value will actually be channels out from model
    def __init__(self, lr, loss, lr_min=0, lr_decay_rate = 1, num_layers = 2, k = 2, feature_vector_shape= [1], channels_in = [1,1,1], nonlinearity = tf.nn.relu, weight_init = 'uniform', momentum=0.9):
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_min = lr_min
        self.num_layers = num_layers
        self.k = k
        self.feature_vector_shape = feature_vector_shape
        self.channels_in = channels_in
        self.nonlinearity = nonlinearity
        self.weight_init = weight_init
        self.loss = loss
        self.build_model()
        self.beta =momentum

        # self.denseOutW = np.random.random()
        # self.denseOutB =

    def build_model(self):
        self.layers = [CCN_Layer(self.k, self.feature_vector_shape, self.channels_in[i], self.channels_in[i+1], self.nonlinearity, self.weight_init)
                       for i in range(self.num_layers)]

    def shuffle(self, X,Y,adjM,parts):
        to_shuffle = list(zip(X,Y,adjM,parts))
        np.random.shuffle(to_shuffle)
        X,Y,adjM,parts = zip(*to_shuffle)
        return X,Y,adjM,parts

    def predict(self, X, adjM, parts):
        for i in range(self.num_layers):
            X, adjM, parts = self.layers[i].call(X,adjM,parts)
        X = tf.math.reduce_sum(X, range(2,len(tf.shape(X))))
        X = tf.reduce_max(X, 0)
        X=tf.reshape(X, (self.channels_in[-1],))
        return X

    def createTensors(self,X, Y):
        X = [[tf.reshape(
            tf.cast(tf.Variable(n), tf.float32), [self.channels_in[0]] + [1] * self.k + self.feature_vector_shape)
            for n in x]
         for x in X]
        Y = [tf.cast(tf.Variable(y), tf.float32) for y in Y]
        return X,Y
    def fit(self, X, Y, adjM, parts, num_epochs):
        for i in range(num_epochs):
            X, Y, adjM, parts = self.shuffle(X,Y,adjM,parts)
            print("epoch: " +str(i))
            data_num = 0
            total_loss = 0
            for x, matrix, part, y in zip(X,adjM, parts, Y):
                max_grad = -np.inf
                with tf.GradientTape(persistent= True) as gt:
                    gt.watch(x)
                    Y_pred = self.predict(x, matrix, part)
                    loss = self.loss(Y_pred, y)
                    total_loss += loss
                for q in range(self.num_layers):
                    gradW = gt.gradient(loss, self.layers[q].W)
                    gradB = gt.gradient(loss, self.layers[q].bias)


                    WUpdate = self.layers[q].momentumW * self.beta + (1-self.beta) * gradW
                    BUpdate = self.layers[q].momentumB * self.beta + (1 - self.beta) * gradB
                    max_grad = max(max_grad, max(np.max(WUpdate), np.max(BUpdate)))
                    self.layers[q].momentumW = WUpdate
                    self.layers[q].momentumB = BUpdate
                    self.layers[q].W.assign_sub(WUpdate * self.lr)
                    self.layers[q].bias.assign_sub(BUpdate * self.lr)
                print("result: " + str(Y_pred) + ", loss " + str(data_num) + ": " + str(loss) + ", maxgrad: " + str(max_grad))
                data_num +=1
            self.lr = self.lr * self.lr_decay_rate
            if self.lr < self.lr_min:
                self.lr = self.lr_min
            print("epoch total loss: " +str(total_loss))