
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






# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#######################################
####        Data on the fly        ####
#######################################

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)








#######################################
####   Data on the fly(flexible)   ####
#######################################

class DataGenerator(tensorflow.keras.utils.Sequence):
    
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32, image_size=256, shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        if filename in self.pneumonia_locations:
            # loop through pneumonia
            for location in self.pneumonia_locations[filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h = location
                msk[y:y+h, x:x+w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)




import numpy as np

# DefineParameters
params = {}

# Datasets
partition = # IDs
labels = # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

model.fit_generator(
	generator=training_generator,
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=6
)