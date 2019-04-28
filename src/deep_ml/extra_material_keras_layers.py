# epochs = 120


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x





def resnet_v2(input_shape, depth, num_classes=10):

    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    enc_feat = Input(shape=(256,))
    
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # out-size = (None, 8, 8, 256)
    
    
    # start attention block
#     res_feat = Input(shape=(hh,hh,cc))
#     enc_feat = Input(shape=(cc,))
    res_feat = x
    enc_feat_c = RepeatVector(64)(enc_feat)
#     enc_feat_c = K.repeat_elements(enc_feat, 64, axis=1)
    enc_feat_c = Reshape(target_shape=(8,8,256))(enc_feat_c)
    
    x = keras.layers.concatenate([res_feat, enc_feat_c], axis=-1) #(8,8,512)
#     print(x)
#     print(Conv2D(256,(1, 1), padding='same')(x))
    x = Conv2D(256,(1, 1), #gfhfghfg
               padding='same',
               kernel_initializer='he_normal')(x) #(8,8,256)
    x = Activation('relu')(x)
#     print(x)
    # softmax to calculate weights
    x = Reshape(target_shape=(64,256))(x)
#     print(x)
    x = Dense(1, activation='softmax')(x) #(8,8,1)
#     print(x)
    x = Reshape(target_shape=(8,8,1))(x)
#     print(x)
#     x = Dense(1)(x) #(8,8,1)
#     print(x)
    # weighted multiply
    x = multiply([x, res_feat]) # (8,8,256)
#     print(x)
    # skip connection
    x = keras.layers.concatenate([x, res_feat], axis=-1)
#     print(x)
#     print(x)
    x = Conv2D(256,(1, 1), 
               padding='same',
               kernel_initializer='he_normal')(x) #(8,8,256)
#     print(x)
#     print(x)
#     att_feat = Model(inputs=[res_feat, enc_feat], outputs=x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(256,(3, 3), 
               padding='same',
               kernel_initializer='he_normal')(x) #(8,8,256)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    
    final_out = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)


    model = Model(inputs=[inputs, enc_feat], outputs=final_out)
    
    
#     x = AveragePooling2D(pool_size=8)(x)
#     y = Flatten()(x)
#     outputs = Dense(num_classes,
#                     activation='softmax',
#                     kernel_initializer='he_normal')(y)

    # Instantiate model.
#     model = Model(inputs=inputs, outputs=x)
    return model


resnet = resnet_v2(input_shape=input_shape, depth=depth)
resnet.summary()













def _decoder():
    input_img = Input(shape=(2, 2, 64), name='decoded_feature')
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model2 = Model(input_img, decoded)
    return model2

dec = _decoder()
dec.summary()









#####

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

def _encoder():
    input_img = Input(shape=(32, 32, 3), name='encoded_feature')

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    model1 = Model(input_img, encoded)
    
    return model1



enc         = _encoder()
enc.summary()







#####

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    new_block1 = x # ======================> add here
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    new_block2 = x
    # out-size = (None, 8, 8, 256)
    
    
    
    #################################
    # features

#     # inp = Input(shape=(7,7,1024))

#     feat1 = Conv2D(128, (1, 1), dilation_rate=2, padding='same')(inp)
#     feat11 = Conv2D(256, (3, 3), dilation_rate=2, padding='same')(feat1)

    #################################


    #################################
    # channel attention

    c_global_feat = keras.layers.AveragePooling2D(pool_size=(8,8))(new_block2)
    c_global_feat = Conv2D(256, (1, 1), 
                           padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(c_global_feat)
#     c_global_feat = BatchNormalization()(c_global_feat)
    c_global_feat = Activation('relu')(c_global_feat)
    
    c_global_feat = Conv2D(256, (1, 1), 
                           padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(c_global_feat)
    c_global_feat = Activation('sigmoid')(c_global_feat)

    channel_weighted_feat = keras.layers.Multiply()([new_block2, c_global_feat])
    channel_weighted_feat = Conv2D(256, (1, 1), padding='same')(channel_weighted_feat)
    # new_feat = keras.layers.Add()([feat11, weighted_feat])

    #################################


    #################################
    # spatial attention

    s_global_feat = Conv2D(256, (1, 1), 
                           padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(new_block2)
    s_global_feat = Activation('relu')(s_global_feat)
    
    s_global_feat = Conv2D(1, (1, 1), 
                           padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(s_global_feat)
    s_global_feat = Activation('sigmoid')(s_global_feat)

    # keras.layers.Multiply()([feat11, Dense(1)(feat11)])

    spatial_weighted_feat = keras.layers.Multiply()([new_block2, s_global_feat])
    spatial_weighted_feat = Conv2D(256, (1, 1), padding='same')(spatial_weighted_feat)

    #################################


    #################################
    # concat both attention features

    concat_s_c_feat = keras.layers.Concatenate(axis=-1)([channel_weighted_feat, spatial_weighted_feat])
    concat_s_c_feat = Conv2D(256, (1, 1), 
                             padding='same',
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(1e-4))(concat_s_c_feat)

    final_feat = keras.layers.Add()([new_block1, concat_s_c_feat])

    #################################
    # parallel_attention = Model(inp, final_feat)
    # parallel_attention.summary()
    
    x = final_feat
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
#     x = Conv2D(256, (1, 1), 
#                padding='same',
#                kernel_initializer='he_normal',
#                kernel_regularizer=l2(1e-4))(x)
    
    
    
    
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


# model = resnet_v2(input_shape=input_shape, depth=depth)

# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=lr_schedule(0)),
#               metrics=['accuracy'])
# model.summary()
# # print(model_type)








#####

class protoNet(tf.keras.Model):
    def __init__(self, hidden_dims, output_dims):
#     def __init__(self):
        super(protoNet, self).__init__()
#         tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1)),
        self.conv1 = tf.keras.layers.Conv2D(hidden_dims,kernel_size=5, strides=2,padding="SAME")
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
#         self.drop = tf.keras.layers.Dropout(0.25)
        self.act1 = tf.keras.layers.Activation('relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.conv2 = tf.keras.layers.Conv2D(hidden_dims,kernel_size=3, strides=1,padding="SAME")
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        
        self.conv3 = tf.keras.layers.Conv2D(hidden_dims,kernel_size=3, strides=1,padding="SAME")
        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.Activation('relu')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        
        self.conv4 = tf.keras.layers.Conv2D(output_dims,kernel_size=3, strides=1,padding="SAME")
        self.batchnorm4 = tf.keras.layers.BatchNormalization()
        self.act4 = tf.keras.layers.Activation('relu')
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
                
        self.conv5 = tf.keras.layers.Conv2D(output_dims,kernel_size=3, strides=1,padding="SAME")
        self.batchnorm5 = tf.keras.layers.BatchNormalization()
        self.act5 = tf.keras.layers.Activation('relu')
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        
        self.flatten = tf.keras.layers.Flatten()       
        self.dense1 = tf.keras.layers.Dense(output_dims)








#####

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()









#####

from keras.models import Model
from keras.layers import Input, LSTM, Dense

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

We train our model in two lines, while monitoring the loss on a held-out set of 20% of the samples.

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


