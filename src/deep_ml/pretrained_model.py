from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate

import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
import pickle
import os, cv2
%matplotlib inline



from keras.applications import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2


#Xception 			88 MB 	0.790 	0.945 	22,910,480 		126
#VGG16 				528 MB 	0.713 	0.901 	138,357,544 	23
#VGG19 				549 MB 	0.713 	0.900 	143,667,240 	26
#ResNet50 			98 MB 	0.749 	0.921 	25,636,712 		-
#ResNet50V2 		98 MB 	0.760 	0.930 	25,613,800 		-
#ResNeXt50 			96 MB 	0.777 	0.938 	25,097,128 		-
#InceptionV3 		92 MB 	0.779 	0.937 	23,851,784 		159
#InceptionResNetV2 	215 MB 	0.803 	0.953 	55,873,736 		572
#MobileNet 			16 MB 	0.704 	0.895 	4,253,864 		88
#MobileNetV2 		14 MB 	0.713 	0.901 	3,538,984 		88
#DenseNet121 		33 MB 	0.750 	0.923 	8,062,504 		121


def print_summary():
    print("MobileNet")
    print("ResNet50")
    print("InceptionV3")
    print("Xception")
    print("InceptionResNetV2")
    print("DenseNet121")
    print("MobileNetV2")
    print("VGG16")
    print("SqueezeNet")



def get_pretrained_model(model_name, download_weights):
    """return pretrained model instance
    Args:
        model_name: [mobilenet, resnet, inception, xception, 
                     inception_resnet, densenet, mobilenet_v2]
        download_weights: whether to download or not [True, false]
    """
    if download_weights == False:
        download_weights = None
    else:
        download_weights = 'imagenet'

    if model_name == "mobilenet":
        print("Total params: 3,228,864")
        return MobileNet(input_shape=(224,224,3), 
                        include_top=False, 
                        weights=download_weights)
    if model_name == "resnet":
        print("Total params: 23,587,712")
        return ResNet50(input_shape=(input_size,input_size,3), 
                        include_top=False, 
                        weights=download_weights)
    if model_name == "inception":
        print("Total params: 21,802,784")
        return InceptionV3(input_shape=(input_size,input_size,3), 
                           include_top=False,
                           weights=download_weights)
    if model_name == "xception":
        print("Total params: 20,861,480")
        return Xception(input_shape=(input_size,input_size,3), 
                        include_top=False, 
                        weights=download_weights)

    if model_name == "inception_resnet":
        print("Total params: 54,336,736")
        return InceptionResNetV2(input_shape=(input_size,input_size,3), 
                                include_top=False, 
                                weights=download_weights)
    if model_name == "densenet":
        print("Total params: 7,037,504")
        return DenseNet121(input_shape=(input_size,input_size,3), 
                           include_top=False, 
                           weights=download_weights)
    if model_name == "mobilenet_v2":
        print("Total params: 2,257,984")
        return MobileNetV2(input_shape=(input_size,input_size,3), 
                           include_top=False, 
                           weights=download_weights)


class BaseFeatureExtractor(object):
    """docstring for ClassName"""
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    def normalize(self, image):
        raise NotImplementedError("error message")       

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)



class NetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size, model_name, download_weights):
        """return model-feature instance
        Args:
            input_size: input size of the image to model
            model_name: [mobilenet, resnet, inception, xception, 
                         inception_resnet, densenet, mobilenet_v2]
        example:
            input_size  = 224
            input_image = Input(shape=(input_size, input_size, 3))
            feature_extractor = NetFeature(input_size, "mobilenet", False)
            grid_h, grid_w = feature_extractor.get_output_shape()  
            # following features output can be used as input in another layer      
            features = feature_extractor.extract(input_image)
            dense_output = Dense(10)(features)
        """
        input_image = Input(shape=(input_size, input_size, 3))
        mobilenet = get_pretrained_model(model_name, download_weights)
        # mobilenet.load_weights(MOBILENET_BACKEND_PATH)
        x = mobilenet(input_image)
        self.feature_extractor = Model(input_image, x)  

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.
        return image

input_size = 224
mobilenet = NetFeature(input_size, "mobilenet", False)
mobilenet.feature_extractor.summary()



    
class SqueezeNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):

        # define some auxiliary variables and the fire module
        sq1x1  = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu   = "relu_"

        def fire_module(x, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
            x     = Activation('relu', name=s_id + relu + sq1x1)(x)

            left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
            left  = Activation('relu', name=s_id + relu + exp1x1)(left)

            right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
            right = Activation('relu', name=s_id + relu + exp3x3)(right)

            x = concatenate([left, right], axis=3, name=s_id + 'concat')

            return x

        # define the model of SqueezeNet
        input_image = Input(shape=(input_size, input_size, 3))

        x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        self.feature_extractor = Model(input_image, x)  
        self.feature_extractor.load_weights(SQUEEZENET_BACKEND_PATH)


# Example
input_size = 224
max_box_per_image = TRUE_BOX_BUFFER # ANCHORS * BOX
nb_box = BOX
nb_class = CLASS


input_image     = Input(shape=(input_size, input_size, 3))
true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))  

feature_extractor = MobileNetFeature(input_size)

# print(feature_extractor.get_output_shape())    
grid_h, grid_w = feature_extractor.get_output_shape()        
features = feature_extractor.extract(input_image)            

# make the object detection layer
output = Conv2D(nb_box * (4 + 1 + nb_class), 
                (1,1), strides=(1,1), 
                padding='same', 
                name='DetectionLayer', 
                kernel_initializer='lecun_normal')(features)
output = Reshape((grid_h, grid_w, nb_box, 4 + 1 + nb_class))(output)
output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)


# initialize the weights of the detection layer
layer = model.layers[-4]
weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape)/(grid_h*grid_w)
new_bias   = np.random.normal(size=weights[1].shape)/(grid_h*grid_w)

layer.set_weights([new_kernel, new_bias])
model.summary()

