import tensorflow as tf
from covariant_compositional_networks_tf2.CCNLayer import CCN_Layer
class CCN_Model:
    # channels in should be num_layers + 1 length and the last value will actually be channels out from model
    def __init__(self, lr, loss, lr_min=0, lr_decay_rate = 1, num_layers = 2, k = 2, feature_vector_shape= [1], channels_in = [1,1,1], nonlinearity = tf.nn.relu, weight_init = 'uniform'):
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

    def build_model(self):
        self.layers = [CCN_Layer(self.k, self.feature_vector_shape, self.channels_in[i], self.channels_in[i+1], self.nonlinearity, self.weight_init)
                       for i in range(self.num_layers)]


    def predict(self, X, adjM, parts):
        for i in range(self.num_layers):
            X, adjM, parts = self.layers[i].call(X,adjM,parts)
        X = tf.math.reduce_sum(X)
        X=tf.reshape(X, (1,))
        return X

    def fit(self, X, Y, adjM, parts):
        with tf.GradientTape(persistent= True) as gt:
            gt.watch(X)
            Y_pred = self.predict(X, adjM, parts)
            loss = self.loss(Y_pred, Y)

        for i in range(self.num_layers):
            gradW = gt.gradient(loss, self.layers[i].W)
            gradB = gt.gradient(loss, self.layers[i].bias)

            self.layers[i].W.assign_sub(gradW * self.lr)
            self.layers[i].bias.assign_sub( gradB * self.lr)
        self.lr = self.lr * self.lr_decay_rate
        if self.lr < self.lr_min:
            self.lr = self.lr_min