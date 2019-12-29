
import numpy as np
from keras.utils import multi_gpu_model

def build_gpu_model():
	"""
	Example:
	with tf.device('/cpu:0'):
		model = Facenet(input_shape=input_shape, depth=depth)

	parallel_model = multi_gpu_model(model, gpus=2)
	parallel_model.compile(
		loss='categorical_crossentropy',
		optimizer='adam,
		metrics=['accuracy']
	)
	"""



# Augmentation
def training_with_real_time_augmentation():
	"""
	Example: 

	print('Using real-time data augmentation.')
	# This will do preprocessing and realtime data augmentation:
	datagen = ImageDataGenerator(
	    featurewise_center            = False,  # set input mean to 0 over the dataset
	    samplewise_center             = False,  # set each sample mean to 0
	    featurewise_std_normalization = False,  # divide inputs by std of the dataset
	    samplewise_std_normalization  = False,  # divide each input by its std
	    zca_whitening                 = False,  # apply ZCA whitening
	    zca_epsilon                   = 1e-06,  # epsilon for ZCA whitening
	    rotation_range                = 0,  # randomly rotate images in the range (degrees, 0 to 180)
	    width_shift_range             = 0.1,# randomly shift images horizontally (fraction of total width)
	    height_shift_range            = 0.1,# randomly shift images vertically (fraction of total height)
	    shear_range                   = 0.,  # set range for random shear
	    zoom_range                    = 0.,  # set range for random zoom
	    channel_shift_range           = 0.,  # set range for random channel shifts
	    fill_mode                     = 'nearest',# set mode for filling points outside the input boundaries
	    cval                          = 0.,  # value used for fill_mode = "constant"
	    horizontal_flip               = True,  # randomly flip images
	    vertical_flip                 = False,  # randomly flip images
	    rescale                       = None,# set rescaling factor (applied before any other transformation)
	    data_format                   = None,# image data format, either "channels_first" or "channels_last"
	    validation_split              = 0.0# fraction of images reserved for validation (strictly between 0 and 1)
	    )

	# Compute quantities required for feature-wise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(x_train)

	model.fit_generator(
	    datagen.flow(x_train, y_train, batch_size=batch_size),
	    epochs=epochs,
	    validation_data=(x_test, y_test),
	    steps_per_epoch=int(1500/4),
	    workers=4
	    callbacks=[reduce_lr, checkpoint, early_stop],
	)
	"""