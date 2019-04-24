if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
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

    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    early_stop = EarlyStopping( monitor='val_loss', 
                                min_delta=0.001, 
                                patience=8, 
                                mode='min', 
                                verbose=1)

    checkpoint = ModelCheckpoint('cifar10_baseline.h5', 
                                  monitor='val_loss', 
                                  verbose=1, 
                                  save_best_only=True, 
                                  mode='min', 
                                  period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.2, 
                                  patience=4, 
                                  min_lr=1e-6) 
    
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        steps_per_epoch=1500,
                        validation_data=(x_test, y_test),
                        callbacks=[reduce_lr, checkpoint, early_stop],
                        workers=4)







## 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


def lr_schedule(epoch):
    """Learning Rate Schedule
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 140:
        lr *= 1e-3
    elif epoch > 100:
        lr *= 1e-2
    elif epoch > 60:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



import tensorflow as tf
from keras.utils import multi_gpu_model
import numpy as np

with tf.device('/cpu:0'):
    model = resnet_v2(input_shape=input_shape, depth=depth)


# Replicates the model on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_attention_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True,
    validation_split=0.0)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

# Fit the model on the batches generated by datagen.flow().
resnet_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    epochs=epochs, verbose=1, workers=4,steps_per_epoch=int(1500/4),
                    callbacks=callbacks)







