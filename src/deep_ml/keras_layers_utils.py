
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, l1, l1_l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

from keras.layers.advanced_activations import LeakyReLU



keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)


    spatial_weighted_feat = keras.layers.Multiply()([new_block2, s_global_feat])
    spatial_weighted_feat = keras.layers.Add()([new_block2, s_global_feat])


def get_all_variable_docs():
    """
    Arguments

    filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    padding: one of "valid" or "same" (case-insensitive). Note that "same" is slightly inconsistent across backends with strides != 1, as described here
    dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
    dilation_rate: An integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
    depth_multiplier: The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to filters_in * depth_multiplier.

    """

from keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, DepthwiseConv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D

def conv_block( inputs, 
                num_filters,
                kernel_size=3,
                strides=1,
                activation='relu',
                batch_norm=True,
                l2_reg=1e-4,
                padding='valid',
                kernel_init='he_normal',
                pool_size=2,
                dilation_rate=1):

    x = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, activation=activation, use_bias=True, kernel_initializer=kernel_init, bias_initializer='zeros', kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    
    return x


####################################################################################
####################################################################################
Conv2D
SeparableConv2D
DepthwiseConv2D
Conv2DTranspose
Cropping2D
UpSampling2D
ZeroPadding2D
####################################################################################
####################################################################################



[source]
Conv2D
def conv2d_layer( inputs, 
            num_filters,
            kernel_size=3,
            strides=1,
            activation='relu',
            l2_reg=1e-4,
            padding='valid',
            kernel_init='he_normal',
            dil_rate=1):
    return keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dil_rate, activation=activation, use_bias=True, kernel_initializer=kernel_init, bias_initializer='zeros', kernel_regularizer=l2(l2_reg))




[source]
SeparableConv2D
def sep_conv2d_layer(num_filters,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    l2_reg=1e-4,
                    padding='valid',
                    kernel_init='he_normal',
                    dil_rate=1):
    """ Separable convolutions consist in first performing a depthwise spatial convolution (which acts on each input channel separately) followed by a pointwise convolution which mixes together the resulting output channels. The depth_multiplier argument controls how many output channels are generated per input channel in the depthwise step.
    """
    x = keras.layers.SeparableConv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dil_rate, depth_multiplier=1, activation=activation, use_bias=True, depthwise_initializer=kernel_init, pointwise_initializer=kernel_init, bias_initializer='zeros', depthwise_regularizer=l2(l2_reg), pointwise_regularizer=l2(l2_reg))
    return x





[source]
DepthwiseConv2D
def depth_conv2d_layer( num_filters,
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        depth_mul=1,
                        depth_l2_reg=1e-4,
                        padding='valid',
                        kernel_init='he_normal',
                        dil_rate=1):
    x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding, depth_multiplier=depth_mul, activation=activation, use_bias=True, depthwise_initializer=kernel_init, bias_initializer='zeros', depthwise_regularizer=l2(l2_reg))
    return x;



[source]
Conv2DTranspose
def conv2d_trans_layer( num_filters,
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        depth_mul=1,
                        depth_l2_reg=1e-4,
                        padding='valid',
                        kernel_init='he_normal',
                        dil_rate=1,
                        output_pad=None):
    """
    Arguments:
    output_padding: An integer or tuple/list of 2 integers, specifying the amount of padding along the height and width of the output tensor. Can be a single integer to specify the same value for all spatial dimensions. The amount of output padding along a given dimension must be lower than the stride along that same dimension. If set to None (default), the output shape is inferred.
    dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.

    new_rows = ((rows - 1) * strides[0] + kernel_size[0]
                - 2 * padding[0] + output_padding[0])
    new_cols = ((cols - 1) * strides[1] + kernel_size[1]
                - 2 * padding[1] + output_padding[1])
    """
    x = keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, output_padding=output_pad, dilation_rate=dil_rate, activation=activation, use_bias=True, kernel_initializer=kernel_init, bias_initializer='zeros', kernel_regularizer=l2(l2_reg))
    return x





[source]
Cropping2D
keras.layers.Cropping2D(cropping=((0, 0), (0, 0)))

Arguments
    cropping: ((top_crop, bottom_crop), (left_crop, right_crop))


Examples

# Crop the input 2D images or feature maps
model = Sequential()
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))
# now model.output_shape == (None, 24, 20, 3)
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Cropping2D(cropping=((2, 2), (2, 2))))
# now model.output_shape == (None, 20, 16, 64)


[source]
UpSampling2D
keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')

Repeats the rows and columns of the data by size[0] and size[1] respectively.

Arguments
    size: int, or tuple of 2 integers. The upsampling factors for rows and columns.
    interpolation: A string, one of nearest or bilinear. Note that CNTK does not support yet the bilinear upscaling and that with Theano, only size=(2, 2) is possible.



[source]
ZeroPadding2D
keras.layers.ZeroPadding2D(padding=(1, 1))

This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.

Arguments
    padding: ((top_pad, bottom_pad), (left_pad, right_pad))

















[source]
Conv1D
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', , dilation_rate=1, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=l2(1e-4))


Arguments
    padding: One of "valid", "causal" or "same" (case-insensitive). "valid" means "no padding". "same" results in padding the input such that the output has the same length as the original input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t + 1:]. A zero padding is used such that the output has the same length as the original input. Useful when modeling temporal data where the model should not violate the temporal order. See WaveNet: A Generative Model for Raw Audio, section 2.1.
    dilation_rate: an integer or tuple/list of a single integer, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.


[source]
SeparableConv1D
keras.layers.SeparableConv1D(filters, kernel_size, strides=1, padding='valid', , dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='he_normal', pointwise_initializer='he_normal', bias_initializer='zeros', depthwise_regularizer=l2(1e-4), pointwise_regularizer=l2(1e-4))


[source]
Cropping1D
keras.layers.Cropping1D(cropping=(1, 1))

It crops along the time dimension (axis 1).

Arguments
    cropping: How many units should be trimmed off at the beginning and end of the cropping dimension (axis 1). 



[source]
UpSampling1D
keras.layers.UpSampling1D(size=2)

Repeats each temporal step size times along the time axis.

Arguments
    size: integer. Upsampling factor.

(batch, steps, features) ==> (batch, upsampled_steps, features).


[source]
ZeroPadding1D
keras.layers.ZeroPadding1D(padding=1)

Arguments
    padding: How many zeros to add at the beginning and at the end of the padding dimension ((left_pad, right_pad)).

(batch, axis_to_pad, features) ==> (batch, padded_axis, features)

#############################################################################
#############################################################################










#############################################################################
#############################################################################
def pooling_2d_layer(inputs, pooling_name="max_pool", pool_size=2, strides=None, padding='valid'):
    """Pooling operation for spatial data."""
    if pooling_name == "max_pool":
        x = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
    if pooling_name == "avg_pool":
        x = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
    if pooling_name == "global_max_pool":
        x = GlobalMaxPooling2D()(inputs)
    if pooling_name == "global_avg_pool":
        x = GlobalAveragePooling2D()(inputs)
    return x

def pooling_1d_layer(inputs, pooling_name="max_pool", pool_size=2, strides=None, padding='valid'):
    """Pooling operation for temporal data."""
    if pooling_name == "max_pool":
        x = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
    if pooling_name == "avg_pool":
        x = AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
    if pooling_name == "global_max_pool":
        x = GlobalMaxPooling1D()(inputs)
    if pooling_name == "global_avg_pool":
        x = GlobalAveragePooling1D()(inputs)
    return x
#############################################################################
#############################################################################








#############################################################################
#############################################################################
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, Permute, RepeatVector

keras.layers.Flatten()
keras.layers.Reshape(target_shape)
keras.layers.Activation(activation)


def dense_layer(inputs, units, activation='linear', kernel_init='glorot_uniform', l2_reg=1e-4):
    x = Dense(units=units, activation=activation, use_bias=True, kernel_initializer=kernel_init, bias_initializer='zeros', kernel_regularizer=l2(l2_reg))(inputs)
    return x


def dropout_layer(inputs, dropout_rate=0.3, seed=1234, noise_shape=None):
    """
    Arguments:
    rate: float between 0 and 1. Fraction of the input units to drop.
    noise_shape: 1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. For instance, if your inputs have shape (batch_size, timesteps, features) and you want the dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features).
    """
    x = Dropout(rate=dropout_rate, noise_shape=noise_shape, seed=seed)(inputs)
    return x



def permute_layer(inputs, dims):
    """ Permutes the dimensions of the input according to a given pattern.(Useful for e.g. connecting RNNs and convnets together.)

    Arguments
        dims: Tuple of integers. Permutation pattern, does not include the samples dimension. Indexing starts at 1. For instance, (2, 1) permutes the first and second dimension of the input.
    
    Example:    model.add(Permute((2, 1), input_shape=(10, 64))) # == (None, 64, 10)
    """
    x = Permute(dims=dims)(inputs)
    return x



def repeat_vector(inputs, no_of_times):
    """ (num_samples, features) ==> (num_samples, n, features)"""
    x = RepeatVector(no_of_times)(inputs)
    return x
#############################################################################
#############################################################################






#############################################################################
#############################################################################
from keras.layers import Add, Subtract, Multiply, Maximum, Minimum, Average, Dot, Concatenate

def simple_functional_layers(list_of_inputs, func_name, concat_axis=-1, dot_prod_axis=-1, dot_norm=Flase):
    """return a functional layer, which work as Add()([x1, x2]) 

    list_of_inputs: list of inputs as [x1, x2, x3]

    For concatenate: default axes=-1
    For dot: Layer that computes a dot product between samples in two tensors.

    E.g. if applied to a list of two tensors a and b of shape (batch_size, n), the output will be a tensor of shape (batch_size, 1) where each entry i will be the dot product between a[i] and b[i].

    Arguments

    dot_prod_axis: Integer or tuple of integers, axis or axes along which to take the dot product.
    dot_norm: Whether to L2-normalize samples along the dot product axis before taking the dot product. If set to True, then the output of the dot product is the cosine proximity between the two samples.
    """
    if not isinstance(list_of_inputs, list):
        raise Exception('list_of_inputs should be a list as [x1, x2]') 

    if func_name == "add": x = Add()
    if func_name == "subtract": x = Subtract()
    if func_name == "multiply": x = Multiply()
    if func_name == "minimum": x = Minimum()
    if func_name == "maximum": x = Maximum()
    if func_name == "average": x = Average()
    if func_name == "concatenate": x = Concatenate(axis=concat_axis)
    if func_name == "dot": x = Dot(axis=dot_prod_axis, normalize=dot_norm)

    return x(list_of_inputs)
#############################################################################
#############################################################################





#############################################################################
#############################################################################
def plot_model_graph(model, file_name="model.png"):
    """
        show_shapes (defaults to False) controls whether output shapes are shown in the graph.
        show_layer_names (defaults to True) controls whether layer names are shown in the graph.
    """
    from keras.utils import plot_model
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=file_name)
    print("Saved model graph with the file name: ", file_name)


def plot_model_history(model):
    """plot model losses and accucracies"""
    fig, ax = plt.subplots(1,2, figsize=(18, 5))
    # Plot training & validation accuracy values
    ax[0].plot(model.history.history['acc'], '-p')
    ax[0].plot(model.history.history['val_acc'], '-p')
    ax[0].set_title('Model accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    ax[1].plot(model.history.history['loss'], '-p')
    ax[1].plot(model.history.history['val_loss'], '-p')
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Test'], loc='upper left')
#############################################################################
#############################################################################






#############################################################################
#############################################################################
from keras.layers import LeakyReLU, PReLU, ELU, ThresholdedReLU

def activation_func_layer(activation_func='relu', leaky_alpha=0.3, elu_alpha=1, threshold_theta=1, relu_neg_slope=0, relu_threshold=0, relu_max_value=None):
    """
    functional_form:

        leaky_relu: f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
        prelu: f(x) = alpha * x for x < 0, f(x) = x for x >= 0, where alpha is a learned array with the same shape as x.
        elu: f(x) =  alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
        threshold_relu: f(x) = x for x > theta, f(x) = 0 otherwise.
        relu: f(x) = max_value for x >= max_value, f(x) = x for threshold <= x < max_value, f(x) = negative_slope * (x - threshold) otherwise.
            (default relu is max(x, 0)), 
    
    Additional_info:
    prelu:
        PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

        alpha_initializer: initializer function for the weights.
        alpha_regularizer: regularizer for the weights.
        alpha_constraint: constraint for the weights.
        shared_axes: the axes along which to share learnable parameters for the activation function. For example, if the incoming feature maps are from a 2D convolution with output shape (batch, height, width, channels), and you wish to share parameters across space so that each filter only has one set of parameters, set shared_axes=[1, 2].
    
    threshold_relu: Helpful in autoencoders
    relu:
        max_value: float >= 0. Maximum activation value.
        negative_slope: float >= 0. Negative slope coefficient.
        threshold: float. Threshold value for thresholded activation.
    """
    if activation_func == "leaky_relu":
        x = LeakyReLU(alpha=leaky_relu)
    if activation_func == "prelu":
        x = PReLU()
    if activation_func == "elu":
        x = ELU(alpha=elu_alpha)
    if activation_func == "threshold_relu":
        x = ThresholdedReLU(theta=threshold_theta)
    if activation_func == 'relu':
        x = ReLU(max_value=relu_max_value, negative_slope=relu_neg_slope, threshold=relu_threshold)

#############################################################################
#############################################################################










