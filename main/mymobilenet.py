from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, Dense
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.utils.vis_utils import plot_model
from keras.layers import DepthwiseConv2D
from keras import backend as K

class MyMobileNetV2:
    def __init__(self):
        # Value is -1 for TF backend
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    
    def convBlock(self, inputs, filters, kernel, strides):
        # check for chennel axis first, if your are using TF backend , you don't need this.
        # normal conv2D
        layer = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        layer = BatchNormalization(axis=self.channel_axis)(layer)

        return Activation('relu')(layer)


    def bottleneckBlock(self, inputs, filters, kernel, exp_factor, strides, res=False):
        # expand channel shape
        expanded = K.int_shape(inputs)[self.channel_axis] * exp_factor
        # nomal convolution, dpeth = channels * ef
        # original W * H * M => W * H * (M * ef), (M * ef) number of (1 * 1 * M) filters
        layer = self.convBlock(inputs, expanded, (1, 1), (1, 1))

        # depthwise conv, depth = 1
        # original W * H * M => W * H * M, M number of (k * k * 1) filters
        layer = DepthwiseConv2D(kernel, strides=(strides, strides), depth_multiplier=1, padding='same')(layer)
        layer = BatchNormalization(axis=self.channel_axis)(layer)
        layer = Activation('relu')(layer)

        # Pointwise convolution, dpeth = filters (N)
        # original W * H * M => W * H * N, N number of (1 * 1 * M) filters
        layer = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(layer)
        layer = BatchNormalization(axis=self.channel_axis)(layer)
        # No relu activation here
        # Or you can try self.convBlock(layer, filters, (1, 1), (1, 1), )

        # H(x) = F(x) (output_layer) + x (input), ensured same shape
        if res:
            layer = add([layer, inputs])
        return layer

    def invertedResidualBlock(self, inputs, filters, kernel, exp_factor, strides, repeats=1):
        '''
        exp_factor: expansion factor for the channel
        repeats: for how many times should the residual block of stride 1 (same output) be repeated
        '''

        # stides not equal to 1
        layer = self.bottleneckBlock(inputs, filters, kernel, exp_factor, strides)
        for i in range(1, repeats):
            layer = self.bottleneckBlock(layer, filters, kernel, exp_factor, 1, True)
        return layer


    def MobileNetv2Conv(self, input_shape=(224,224,3)):
        inputs = Input(shape=input_shape)
        layer = self.convBlock(inputs, 32, (3, 3), strides=(2, 2))
        # initial repeat parameters: 1,2,3,4,3,3,1
        layer = self.invertedResidualBlock(layer, 16, (3, 3),  exp_factor=1, strides=1, repeats=1)
        layer = self.invertedResidualBlock(layer, 24, (3, 3),  exp_factor=6, strides=2, repeats=2)
        layer = self.invertedResidualBlock(layer, 32, (3, 3),  exp_factor=6, strides=2, repeats=3)
        layer = self.invertedResidualBlock(layer, 64, (3, 3),  exp_factor=6, strides=2, repeats=4)
        layer = self.invertedResidualBlock(layer, 96, (3, 3),  exp_factor=6, strides=1, repeats=3)
        layer = self.invertedResidualBlock(layer, 160, (3, 3), exp_factor=6, strides=2, repeats=3)
        layer = self.invertedResidualBlock(layer, 320, (3, 3), exp_factor=6, strides=1, repeats=1)

        # Equivalent to flatten layer, expand to 1280
        layer = self.convBlock(layer, 1280, (1, 1), strides=(1, 1))

        model = Model(inputs, layer)
        return model

    
    def MobileNetv2FC(self, layer, num_class):
        '''
            FC layers, may not be used, you can write your own Fully connected layers in 
            fineune.py
        '''
        inputs = Input(shape=layer.shape)
        layer = GlobalAveragePooling2D()(layer)
        #layer = Dense(512,activation='relu')(layer) 
        #layer = Dropout(0.25)(layer)
        layer = Dense(256, activation='relu')(layer) 
        layer = Dropout(0.25)(layer)
        preds = Dense(num_class,activation='softmalayer')(layer) 
        model = Model(inputs=input,outputs=preds)
        return model
