# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

l2 = tf.keras.regularizers.l2

def adaIN(inputs, style, epsilon=1e-5):

    in_mean, in_var = tf.nn.moments(inputs, axes=[1,2], keepdims=True)
    st_mean, st_var = tf.nn.moments(style, axes=[1,2], keepdims=True)
    in_std, st_std = tf.sqrt(in_var + epsilon), tf.sqrt(st_var + epsilon)

    return st_std * (style - in_mean) / in_std + st_mean

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, weight_decay, conv_fn):
        super(block, self).__init__()
        self.conv_fn = conv_fn
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        self.conv = tf.keras.layers.DepthwiseConv2D(kernel_size=self.kernel_size,
                                                    strides=self.strides,
                                                    padding=self.padding,
                                                    use_bias=self.use_bias,
                                                    depthwise_regularizer=self.weight_decay)
        self.IN = InstanceNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=self.kernel_size,
                                                     strides=self.strides,
                                                     padding=self.padding,
                                                     use_bias=self.use_bias,
                                                     depthwise_regularizer=self.weight_decay)
        self.IN2 = InstanceNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding=self.padding,
                                            use_bias=self.use_bias,
                                            kernel_regularizer=self.weight_decay)
        self.IN3 = InstanceNormalization()
        self.relu3 = tf.keras.layers.ReLU()

        self.conv4 = tf.keras.layers.Conv2D(filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding=self.padding,
                                            use_bias=self.use_bias,
                                            kernel_regularizer=self.weight_decay)
        self.IN4 = InstanceNormalization()
        self.relu4 = tf.keras.layers.ReLU()

    def call(self, inputs):
        if self.conv_fn:
            x = tf.pad(inputs, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
            x = self.conv3(x)
            x = self.IN3(x)
            x = self.relu3(x)

            x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
            x = self.conv4(x)
            x = self.IN4(x)

            x += inputs

            x = self.relu4(x)
        else:
            x = tf.pad(inputs, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
            x = self.conv(x)
            x = self.IN(x)
            x = self.relu(x)

            x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
            x = self.conv2(x)
            x = self.IN2(x)

            x += inputs

            x = self.relu2(x)
        
        return x

class Grouped_residual_conv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, weight_decay, repeat):
        super(Grouped_residual_conv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        self.repeat = repeat
    def call(self, inputs):
        _, H, W, filters_ = inputs.get_shape()
        f_1_input = inputs[:, :, :, 0:filters_ // 2] # 0:16
        f_2_input = inputs[:, :, :, filters_ // 2:] # 16:32
        #f_3_input = inputs[:, :, :, filters_ // 2:filters_*3 // 4] # 32:48
        #f_4_input = inputs[:, :, :, filters_*3 // 4:] # 48:64

        for _ in range(self.repeat):
            f_1_input = block(filters_ // 2, self.kernel_size, self.strides, self.padding, self.use_bias, self.weight_decay, conv_fn=False)(f_1_input)

        for _ in range(self.repeat):
            f_2_input = block(filters_ // 2, self.kernel_size, self.strides, self.padding, self.use_bias, self.weight_decay, conv_fn=True)(f_2_input)

        #for _ in range(self.repeat):
        #    f_3_input = block(self.kernel_size, self.strides, self.padding, self.use_bias, self.weight_decay)(inputs)

        #for _ in range(self.repeat):
        #    f_4_input = block(self.kernel_size, self.strides, self.padding, self.use_bias, self.weight_decay)(inputs)

        return tf.concat([f_1_input, f_2_input], -1)
        #return f_1_input + f_2_input

def V7_generator(input_shape=(256, 256, 3),
                 style_shape_1=(256, 256, 64),
                 style_shape_2=(128, 128, 128),
                 style_shape_3=(64, 64, 256),
                 repeat=1,
                 weight_decay=0.000005):

    h = inputs = tf.keras.Input(input_shape)
    s_1 = styles_1 = tf.keras.Input(style_shape_1)
    s_2 = styles_2 = tf.keras.Input(style_shape_2)
    s_3 = styles_3 = tf.keras.Input(style_shape_3)

    def residual_block(x):  # 글로벌한 스타일

        h = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
        h = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=3,
                                   strides=1,
                                   padding="valid",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay))(h)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h)   # [64, 64, 256]

        h = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
        h = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=3,
                                   strides=1,
                                   padding="valid",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay))(h)
        h = InstanceNormalization()(h)
        h += x

        h = tf.keras.layers.ReLU()(h)   # [64, 64, 256]

        return h

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]    세부적인 스타일    (그러면 세부적인 스타일을 조금더 늘리면 돼지않아?)

    h = Grouped_residual_conv(filters=64,
                              kernel_size=3,
                              strides=1,
                              padding="valid",
                              use_bias=False,
                              weight_decay=l2(weight_decay),
                              repeat=1)(h)  #  파라미터가 보이지 않아서 내가 직접 계산해야한다. 576

    #h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    #h = tf.keras.layers.Conv2D(filters=64,
    #                           kernel_size=3,
    #                           strides=1,
    #                           padding="valid",
    #                           use_bias=False,
    #                           kernel_regularizer=l2(weight_decay))(h)
    #h = InstanceNormalization()(h)
    #h = tf.keras.layers.ReLU()(h)   # [256, 256, 128]    세부적인 스타일

    #h = Grouped_residual_conv(kernel_size=3,
    #                          strides=1,
    #                          padding="valid",
    #                          use_bias=False,
    #                          weight_decay=l2(weight_decay),
    #                          repeat=1)(h)  #  파라미터가 보이지 않아서 내가 직접 계산해야한다. 2,304
    
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]    세부적인 스타일

    h = Grouped_residual_conv(filters=128,
                              kernel_size=3,
                              strides=1,
                              padding="valid",
                              use_bias=False,
                              weight_decay=l2(weight_decay),
                              repeat=1)(h)  # 1,152

    #h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    #h = tf.keras.layers.Conv2D(filters=128,
    #                           kernel_size=3,
    #                           strides=2,
    #                           padding="valid",
    #                           use_bias=False,
    #                           kernel_regularizer=l2(weight_decay))(h)
    #h = InstanceNormalization()(h)
    #h = tf.keras.layers.ReLU()(h)   # [128, 128, 256]    세부적인 스타일

    #h = Grouped_residual_conv(kernel_size=3,
    #                          strides=1,
    #                          padding="valid",
    #                          use_bias=False,
    #                          weight_decay=l2(weight_decay),
    #                          repeat=1)(h)  # 4,608

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s_3)
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 256]

    h = tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s_2)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]       세부적인 스타일

    h = Grouped_residual_conv(filters=128,
                              kernel_size=3,
                              strides=1,
                              padding="valid",
                              use_bias=False,
                              weight_decay=l2(weight_decay),
                              repeat=1)(h)  # 2,304

    #h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    #h = tf.keras.layers.Conv2D(filters=128,
    #                           kernel_size=3,
    #                           strides=1,
    #                           padding="valid",
    #                           use_bias=False,
    #                           kernel_regularizer=l2(weight_decay))(h)
    #h = adaIN(h, s_2)
    #h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]    세부적인 스타일

    #h = Grouped_residual_conv(kernel_size=3,
    #                          strides=1,
    #                          padding="valid",
    #                          use_bias=False,
    #                          weight_decay=l2(weight_decay),
    #                          repeat=1)(h)  # 1,152

    h = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = adaIN(h, s_1)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]    세부적인 스타일

    h = Grouped_residual_conv(filters=64,
                              kernel_size=3,
                              strides=1,
                              padding="valid",
                              use_bias=False,
                              weight_decay=l2(weight_decay),
                              repeat=1)(h)  # 1,152

    #h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    #h = tf.keras.layers.Conv2D(filters=64,
    #                           kernel_size=3,
    #                           strides=1,
    #                           padding="valid",
    #                           use_bias=False,
    #                           kernel_regularizer=l2(weight_decay))(h)
    #h = adaIN(h, s_1)
    #h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]    세부적인 스타일

    #h = Grouped_residual_conv(kernel_size=3,
    #                          strides=1,
    #                          padding="valid",
    #                          use_bias=False,
    #                          weight_decay=l2(weight_decay),
    #                          repeat=1)(h)  # 576

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=[inputs, styles_1, styles_2, styles_3], outputs=h)

def style_map(input_shape=(256, 256, 3),
              weight_decay=0.000005):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=4,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]

    h = Grouped_residual_conv(filters=64,
                              kernel_size=3,
                              strides=1,
                              padding="valid",
                              use_bias=False,
                              weight_decay=l2(weight_decay),
                              repeat=1)(h)  #  파라미터가 보이지 않아서 내가 직접 계산해야한다. 576
    s_1 = h

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=4,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]

    h = Grouped_residual_conv(filters=128,
                              kernel_size=3,
                              strides=1,
                              padding="valid",
                              use_bias=False,
                              weight_decay=l2(weight_decay),
                              repeat=1)(h)  #  파라미터가 보이지 않아서 내가 직접 계산해야한다. 2,304
    s_2 = h

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=4,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 256]

    h = Grouped_residual_conv(filters=256,
                              kernel_size=3,
                              strides=1,
                              padding="valid",
                              use_bias=False,
                              weight_decay=l2(weight_decay),
                              repeat=1)(h)  #  파라미터가 보이지 않아서 내가 직접 계산해야한다. 4,608
    s_3 = h

    return tf.keras.Model(inputs=inputs, outputs=[s_1, s_2, s_3])

def discriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):

    dim_ = dim
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)


    return tf.keras.Model(inputs=inputs, outputs=h)
