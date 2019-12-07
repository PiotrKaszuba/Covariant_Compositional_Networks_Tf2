import tensorflow as tf


# THIS IS NOT IMPLEMENTED YET
# Check testing.py

class CCN_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(CCN_Layer, self).__init__()

    def build(self, input_shape):
        # add weights
        raise NotImplementedError

        # self.kernel = self.add_variable("kernel",
        #                                 shape=[int(input_shape[-1]),
        #                                        self.num_outputs])

    def call(self, input):
        raise NotImplementedError
        # layer behaviour - inputs are [prev_activations, adjM, parts]
        # outputs are [current_activations, adjM, new_parts]
        # based on adjM and parts - there should be a DYNAMIC computational graph defined
        # IMPORTANT:
        # expactations about dynamic graphs: separate graph definition for every input (on layer level) / 'data point'
