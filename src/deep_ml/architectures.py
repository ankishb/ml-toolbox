

import tensorflow as tf
def get_unet():
    Conv2DTranspose = tf.keras.layers.Conv2DTranspose
    concatenate = tf.keras.layers.concatenate
    Model = tf.keras.models.Model
    Conv2D = tf.keras.layers.Conv2D
    MaxPool2D = tf.keras.layers.MaxPool2D
    Input = tf.keras.layers.Input
    mae = tf.keras.losses.mae
    Adam = tf.keras.optimizers.Adam
    accuracy = tf.keras.metrics.mean_squared_error
    
    img_rows = 96
    img_cols = 96
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=mae, metrics=[accuracy])

    model.summary()
    done = 1
    return done
get_unet()










__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 96, 96, 1)    0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 96, 96, 32)   320         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 96, 96, 32)   9248        conv2d[0][0]                     
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 48, 48, 32)   0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 48, 48, 64)   18496       max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 48, 48, 64)   36928       conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 24, 24, 64)   0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 24, 24, 128)  73856       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 24, 24, 128)  147584      conv2d_4[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 12, 12, 128)  0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 12, 12, 256)  295168      max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 12, 12, 256)  590080      conv2d_6[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 6, 6, 256)    0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 6, 6, 512)    1180160     max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 6, 6, 512)    2359808     conv2d_8[0][0]                   
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 12, 12, 256)  524544      conv2d_9[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 12, 12, 512)  0           conv2d_transpose[0][0]           
                                                                 conv2d_7[0][0]                   
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 12, 12, 256)  1179904     concatenate[0][0]                
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 12, 12, 256)  590080      conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 24, 24, 128)  131200      conv2d_11[0][0]                  
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 24, 24, 256)  0           conv2d_transpose_1[0][0]         
                                                                 conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 24, 24, 128)  295040      concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 24, 24, 128)  147584      conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 48, 48, 64)   32832       conv2d_13[0][0]                  
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 48, 48, 128)  0           conv2d_transpose_2[0][0]         
                                                                 conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 48, 48, 64)   73792       concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 48, 48, 64)   36928       conv2d_14[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 96, 96, 32)   8224        conv2d_15[0][0]                  
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 96, 96, 64)   0           conv2d_transpose_3[0][0]         
                                                                 conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 96, 96, 32)   18464       concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 96, 96, 32)   9248        conv2d_16[0][0]                  
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 96, 96, 1)    33          conv2d_17[0][0]                  
==================================================================================================
Total params: 7,759,521
Trainable params: 7,759,521
Non-trainable params: 0
__________________________________________________________________________________________________




import h5py
from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D

def print_summary():
    print("SqueezeNet")
    print("resnet_v2")

def SqueezeNet(nb_classes, inputs=(3, 224, 224)):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    """

    input_img = Input(shape=inputs)
    
    conv1 = Convolution2D(
        96, (7, 7), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        strides=(2, 2), 
        padding='same', name='conv1',
        data_format="channels_first")(input_img)
    
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), 
        strides=(2, 2), 
        name='maxpool1',
        data_format="channels_first")(conv1)
    
    fire2_squeeze = Convolution2D(
        16, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire2_squeeze',
        data_format="channels_first")(maxpool1)
    
    fire2_expand1 = Convolution2D(
        64, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire2_expand1',
        data_format="channels_first")(fire2_squeeze)
    
    fire2_expand2 = Convolution2D(
        64, (3, 3), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire2_expand2',
        data_format="channels_first")(fire2_squeeze)
    
    merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        16, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire3_squeeze',
        data_format="channels_first")(merge2)
    
    fire3_expand1 = Convolution2D(
        64, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire3_expand1',
        data_format="channels_first")(fire3_squeeze)
    
    fire3_expand2 = Convolution2D(
        64, (3, 3), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire3_expand2',
        data_format="channels_first")(fire3_squeeze)
    
    merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Convolution2D(
        32, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire4_squeeze',
        data_format="channels_first")(merge3)
    
    fire4_expand1 = Convolution2D(
        128, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire4_expand1',
        data_format="channels_first")(fire4_squeeze)
    
    fire4_expand2 = Convolution2D(
        128, (3, 3), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire4_expand2',
        data_format="channels_first")(fire4_squeeze)
    
    merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
    
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), 
        strides=(2, 2), 
        name='maxpool4',
        data_format="channels_first")(merge4)

    fire5_squeeze = Convolution2D(
        32, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire5_squeeze',
        data_format="channels_first")(maxpool4)
    
    fire5_expand1 = Convolution2D(
        128, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire5_expand1',
        data_format="channels_first")(fire5_squeeze)
    
    fire5_expand2 = Convolution2D(
        128, (3, 3), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire5_expand2',
        data_format="channels_first")(fire5_squeeze)
    
    merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])

    fire6_squeeze = Convolution2D(
        48, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire6_squeeze',
        data_format="channels_first")(merge5)
    
    fire6_expand1 = Convolution2D(
        192, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire6_expand1',
        data_format="channels_first")(fire6_squeeze)
    
    fire6_expand2 = Convolution2D(
        192, (3, 3), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire6_expand2',
        data_format="channels_first")(fire6_squeeze)
    
    merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])

    fire7_squeeze = Convolution2D(
        48, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire7_squeeze',
        data_format="channels_first")(merge6)
    
    fire7_expand1 = Convolution2D(
        192, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire7_expand1',
        data_format="channels_first")(fire7_squeeze)
    
    fire7_expand2 = Convolution2D(
        192, (3, 3), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire7_expand2',
        data_format="channels_first")(fire7_squeeze)
    
    merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

    fire8_squeeze = Convolution2D(
        64, (1, 1), activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire8_squeeze',
        data_format="channels_first")(merge7)
    
    fire8_expand1 = Convolution2D(
        256, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire8_expand1',
        data_format="channels_first")(fire8_squeeze)
    
    fire8_expand2 = Convolution2D(
        256, (3, 3), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire8_expand2',
        data_format="channels_first")(fire8_squeeze)
    
    merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), 
        strides=(2, 2), 
        name='maxpool8',
        data_format="channels_first")(merge8)
    
    fire9_squeeze = Convolution2D(
        64, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire9_squeeze',
        data_format="channels_first")(maxpool8)
    
    fire9_expand1 = Convolution2D(
        256, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire9_expand1',
        data_format="channels_first")(fire9_squeeze)
    
    fire9_expand2 = Convolution2D(
        256, (3, 3), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='same', 
        name='fire9_expand2',
        data_format="channels_first")(fire9_squeeze)
    
    merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(
        0.5, 
        name='fire9_dropout')(merge9)
    
    conv10 = Convolution2D(
        nb_classes, (1, 1), 
        activation='relu', 
        kernel_initializer='glorot_uniform',
        padding='valid',
        name='conv10',
        data_format="channels_first")(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D(data_format='channels_first')(conv10)
    
    softmax = Activation("softmax", name='softmax')(global_avgpool10)

    return Model(inputs=input_img, outputs=softmax)


__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 3, 224, 224)  0                                            
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 96, 112, 112) 14208       input_1[0][0]                    
__________________________________________________________________________________________________
maxpool1 (MaxPooling2D)         (None, 96, 55, 55)   0           conv1[0][0]                      
__________________________________________________________________________________________________
fire2_squeeze (Conv2D)          (None, 16, 55, 55)   1552        maxpool1[0][0]                   
__________________________________________________________________________________________________
fire2_expand1 (Conv2D)          (None, 64, 55, 55)   1088        fire2_squeeze[0][0]              
__________________________________________________________________________________________________
fire2_expand2 (Conv2D)          (None, 64, 55, 55)   9280        fire2_squeeze[0][0]              
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 128, 55, 55)  0           fire2_expand1[0][0]              
                                                                 fire2_expand2[0][0]              
__________________________________________________________________________________________________
fire3_squeeze (Conv2D)          (None, 16, 55, 55)   2064        concatenate_1[0][0]              
__________________________________________________________________________________________________
fire3_expand1 (Conv2D)          (None, 64, 55, 55)   1088        fire3_squeeze[0][0]              
__________________________________________________________________________________________________
fire3_expand2 (Conv2D)          (None, 64, 55, 55)   9280        fire3_squeeze[0][0]              
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 128, 55, 55)  0           fire3_expand1[0][0]              
                                                                 fire3_expand2[0][0]              
__________________________________________________________________________________________________
fire4_squeeze (Conv2D)          (None, 32, 55, 55)   4128        concatenate_2[0][0]              
__________________________________________________________________________________________________
fire4_expand1 (Conv2D)          (None, 128, 55, 55)  4224        fire4_squeeze[0][0]              
__________________________________________________________________________________________________
fire4_expand2 (Conv2D)          (None, 128, 55, 55)  36992       fire4_squeeze[0][0]              
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 256, 55, 55)  0           fire4_expand1[0][0]              
                                                                 fire4_expand2[0][0]              
__________________________________________________________________________________________________
maxpool4 (MaxPooling2D)         (None, 256, 27, 27)  0           concatenate_3[0][0]              
__________________________________________________________________________________________________
fire5_squeeze (Conv2D)          (None, 32, 27, 27)   8224        maxpool4[0][0]                   
__________________________________________________________________________________________________
fire5_expand1 (Conv2D)          (None, 128, 27, 27)  4224        fire5_squeeze[0][0]              
__________________________________________________________________________________________________
fire5_expand2 (Conv2D)          (None, 128, 27, 27)  36992       fire5_squeeze[0][0]              
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 256, 27, 27)  0           fire5_expand1[0][0]              
                                                                 fire5_expand2[0][0]              
__________________________________________________________________________________________________
fire6_squeeze (Conv2D)          (None, 48, 27, 27)   12336       concatenate_4[0][0]              
__________________________________________________________________________________________________
fire6_expand1 (Conv2D)          (None, 192, 27, 27)  9408        fire6_squeeze[0][0]              
__________________________________________________________________________________________________
fire6_expand2 (Conv2D)          (None, 192, 27, 27)  83136       fire6_squeeze[0][0]              
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 384, 27, 27)  0           fire6_expand1[0][0]              
                                                                 fire6_expand2[0][0]              
__________________________________________________________________________________________________
fire7_squeeze (Conv2D)          (None, 48, 27, 27)   18480       concatenate_5[0][0]              
__________________________________________________________________________________________________
fire7_expand1 (Conv2D)          (None, 192, 27, 27)  9408        fire7_squeeze[0][0]              
__________________________________________________________________________________________________
fire7_expand2 (Conv2D)          (None, 192, 27, 27)  83136       fire7_squeeze[0][0]              
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 384, 27, 27)  0           fire7_expand1[0][0]              
                                                                 fire7_expand2[0][0]              
__________________________________________________________________________________________________
fire8_squeeze (Conv2D)          (None, 64, 27, 27)   24640       concatenate_6[0][0]              
__________________________________________________________________________________________________
fire8_expand1 (Conv2D)          (None, 256, 27, 27)  16640       fire8_squeeze[0][0]              
__________________________________________________________________________________________________
fire8_expand2 (Conv2D)          (None, 256, 27, 27)  147712      fire8_squeeze[0][0]              
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 512, 27, 27)  0           fire8_expand1[0][0]              
                                                                 fire8_expand2[0][0]              
__________________________________________________________________________________________________
maxpool8 (MaxPooling2D)         (None, 512, 13, 13)  0           concatenate_7[0][0]              
__________________________________________________________________________________________________
fire9_squeeze (Conv2D)          (None, 64, 13, 13)   32832       maxpool8[0][0]                   
__________________________________________________________________________________________________
fire9_expand1 (Conv2D)          (None, 256, 13, 13)  16640       fire9_squeeze[0][0]              
__________________________________________________________________________________________________
fire9_expand2 (Conv2D)          (None, 256, 13, 13)  147712      fire9_squeeze[0][0]              
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 512, 13, 13)  0           fire9_expand1[0][0]              
                                                                 fire9_expand2[0][0]              
__________________________________________________________________________________________________
fire9_dropout (Dropout)         (None, 512, 13, 13)  0           concatenate_8[0][0]              
__________________________________________________________________________________________________
conv10 (Conv2D)                 (None, 10, 13, 13)   5130        fire9_dropout[0][0]              
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 10)           0           conv10[0][0]                     
__________________________________________________________________________________________________
softmax (Activation)            (None, 10)           0           global_average_pooling2d_1[0][0] 
==================================================================================================
Total params: 740,554
Trainable params: 740,554
Non-trainable params: 0
__________________________________________________________________________________________________







##################
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
    # softmax to calculate weights
    x = Reshape(target_shape=(64,256))(x)
    x = Dense(1, activation='softmax')(x) #(8,8,1)
    x = Reshape(target_shape=(8,8,1))(x)
    # weighted multiply
    x = multiply([x, res_feat]) # (8,8,256)
    # skip connection
    x = keras.layers.concatenate([x, res_feat], axis=-1)
    x = Conv2D(256,(1, 1), 
               padding='same',
               kernel_initializer='he_normal')(x) #(8,8,256)

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

input_shape=(32, 32, 3)
resnet = resnet_v2(input_shape=input_shape, depth=depth)
resnet.summary()

____________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 3)    0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 16)   448         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 16)   64          conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 16)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 16)   272         activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 16)   64          conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 32, 16)   0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 16)   2320        activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 32, 16)   64          conv2d_3[0][0]                   
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 16)   0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 64)   1088        activation_1[0][0]               
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 64)   1088        activation_3[0][0]               
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32, 32, 64)   0           conv2d_5[0][0]                   
                                                                 conv2d_4[0][0]                   
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 32, 64)   256         add_1[0][0]                      
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 32, 64)   0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 16)   1040        activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 16)   64          conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 32, 16)   0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 16)   2320        activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 32, 16)   64          conv2d_7[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 16)   0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 64)   1088        activation_6[0][0]               
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 32, 64)   0           add_1[0][0]                      
                                                                 conv2d_8[0][0]                   
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 32, 64)   256         add_2[0][0]                      
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 32, 64)   0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 32, 32, 16)   1040        activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 32, 32, 16)   64          conv2d_9[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 32, 32, 16)   0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 32, 32, 16)   2320        activation_8[0][0]               
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 32, 32, 16)   64          conv2d_10[0][0]                  
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 32, 32, 16)   0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 32, 64)   1088        activation_9[0][0]               
__________________________________________________________________________________________________
add_3 (Add)                     (None, 32, 32, 64)   0           add_2[0][0]                      
                                                                 conv2d_11[0][0]                  
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 32, 32, 64)   256         add_3[0][0]                      
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 32, 32, 64)   0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 16, 16, 64)   4160        activation_10[0][0]              
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 16, 16, 64)   256         conv2d_12[0][0]                  
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 16, 16, 64)   0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 16, 16, 64)   36928       activation_11[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 16, 16, 64)   256         conv2d_13[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 16, 16, 64)   0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 16, 16, 128)  8320        add_3[0][0]                      
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 16, 16, 128)  8320        activation_12[0][0]              
__________________________________________________________________________________________________
add_4 (Add)                     (None, 16, 16, 128)  0           conv2d_15[0][0]                  
                                                                 conv2d_14[0][0]                  
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 16, 16, 128)  512         add_4[0][0]                      
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 16, 16, 128)  0           batch_normalization_13[0][0]     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 16, 16, 64)   8256        activation_13[0][0]              
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 16, 16, 64)   256         conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 16, 16, 64)   0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 16, 16, 64)   36928       activation_14[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 16, 16, 64)   256         conv2d_17[0][0]                  
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 16, 16, 64)   0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 16, 16, 128)  8320        activation_15[0][0]              
__________________________________________________________________________________________________
add_5 (Add)                     (None, 16, 16, 128)  0           add_4[0][0]                      
                                                                 conv2d_18[0][0]                  
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 16, 16, 128)  512         add_5[0][0]                      
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 16, 16, 128)  0           batch_normalization_16[0][0]     
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 16, 16, 64)   8256        activation_16[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 16, 16, 64)   256         conv2d_19[0][0]                  
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 16, 16, 64)   0           batch_normalization_17[0][0]     
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 16, 16, 64)   36928       activation_17[0][0]              
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 16, 16, 64)   256         conv2d_20[0][0]                  
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 16, 16, 64)   0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 16, 16, 128)  8320        activation_18[0][0]              
__________________________________________________________________________________________________
add_6 (Add)                     (None, 16, 16, 128)  0           add_5[0][0]                      
                                                                 conv2d_21[0][0]                  
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 16, 16, 128)  512         add_6[0][0]                      
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 16, 16, 128)  0           batch_normalization_19[0][0]     
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 8, 8, 128)    16512       activation_19[0][0]              
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 8, 8, 128)    512         conv2d_22[0][0]                  
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 8, 8, 128)    0           batch_normalization_20[0][0]     
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 8, 8, 128)    147584      activation_20[0][0]              
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 8, 8, 128)    512         conv2d_23[0][0]                  
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 8, 8, 128)    0           batch_normalization_21[0][0]     
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 8, 8, 256)    33024       add_6[0][0]                      
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 8, 8, 256)    33024       activation_21[0][0]              
__________________________________________________________________________________________________
add_7 (Add)                     (None, 8, 8, 256)    0           conv2d_25[0][0]                  
                                                                 conv2d_24[0][0]                  
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 8, 8, 256)    1024        add_7[0][0]                      
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 8, 8, 256)    0           batch_normalization_22[0][0]     
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 8, 8, 128)    32896       activation_22[0][0]              
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 8, 8, 128)    512         conv2d_26[0][0]                  
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 8, 8, 128)    0           batch_normalization_23[0][0]     
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 8, 8, 128)    147584      activation_23[0][0]              
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 8, 8, 128)    512         conv2d_27[0][0]                  
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 8, 8, 128)    0           batch_normalization_24[0][0]     
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 8, 8, 256)    33024       activation_24[0][0]              
__________________________________________________________________________________________________
add_8 (Add)                     (None, 8, 8, 256)    0           add_7[0][0]                      
                                                                 conv2d_28[0][0]                  
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 8, 8, 256)    1024        add_8[0][0]                      
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 8, 8, 256)    0           batch_normalization_25[0][0]     
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 8, 8, 128)    32896       activation_25[0][0]              
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 8, 8, 128)    512         conv2d_29[0][0]                  
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 8, 8, 128)    0           batch_normalization_26[0][0]     
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 8, 8, 128)    147584      activation_26[0][0]              
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 8, 8, 128)    512         conv2d_30[0][0]                  
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 8, 8, 128)    0           batch_normalization_27[0][0]     
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 8, 8, 256)    33024       activation_27[0][0]              
__________________________________________________________________________________________________
add_9 (Add)                     (None, 8, 8, 256)    0           add_8[0][0]                      
                                                                 conv2d_31[0][0]                  
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 256)          0                                            
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 8, 8, 256)    1024        add_9[0][0]                      
__________________________________________________________________________________________________
repeat_vector_1 (RepeatVector)  (None, 64, 256)      0           input_2[0][0]                    
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 8, 8, 256)    0           batch_normalization_28[0][0]     
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 8, 8, 256)    0           repeat_vector_1[0][0]            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 8, 8, 512)    0           activation_28[0][0]              
                                                                 reshape_1[0][0]                  
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 8, 8, 256)    131328      concatenate_1[0][0]              
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 8, 8, 256)    0           conv2d_32[0][0]                  
__________________________________________________________________________________________________
reshape_2 (Reshape)             (None, 64, 256)      0           activation_29[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64, 1)        257         reshape_2[0][0]                  
__________________________________________________________________________________________________
reshape_3 (Reshape)             (None, 8, 8, 1)      0           dense_1[0][0]                    
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 8, 8, 256)    0           reshape_3[0][0]                  
                                                                 activation_28[0][0]              
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 8, 8, 512)    0           multiply_1[0][0]                 
                                                                 activation_28[0][0]              
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 8, 8, 256)    131328      concatenate_2[0][0]              
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 8, 8, 256)    1024        conv2d_33[0][0]                  
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 8, 8, 256)    0           batch_normalization_29[0][0]     
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 8, 8, 256)    590080      activation_30[0][0]              
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 8, 8, 256)    1024        conv2d_34[0][0]                  
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 8, 8, 256)    0           batch_normalization_30[0][0]     
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 1, 1, 256)    0           activation_31[0][0]              
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 256)          0           average_pooling2d_1[0][0]        
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           2570        flatten_1[0][0]                  
==================================================================================================
Total params: 1,704,043
Trainable params: 1,697,803
Non-trainable params: 6,240
__________________________________________________________________________________________________