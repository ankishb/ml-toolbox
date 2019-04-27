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











import keras
from sklearn.metrics import roc_auc_score
 
class Histories(keras.callbacks.Callback):
    def __init__(self, validation_data=()):
        super(Histories, self).__init__()
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        self.complete_losses = []
        self.complete_accs = []
        
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred)
        self.aucs.append(roc_val)
        print('\r =====roc-auc_val: %s =====' % (str(round(roc_val,4))),end=100*' '+'\n')

        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        self.complete_losses.append(logs.get('loss'))
        self.complete_accs.append(logs.get('acc'))
        return




from keras.callbacks import *

class CyclicLR(Callback):
    """
    # Arguments
        base_lr: initial learning rate which is the lower boundary in the cycle.
        max_lr: upper boundary in the cycle.
            
        step_size: number of training iterations per half cycle. 
            Authors suggest setting step_size 2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}. (Default 'triangular')
            "triangular": A basic triangular cycle w/ no amplitude scaling.
            "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
            "exp_range": A cycle that scales initial amplitude by gamma**(cycle iterations) at each cycle iteration. 
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function: gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0. mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}. 
        The amplitude of the cycle can be scaled on a per-iteration or per-cycle basis.
            Defines whether scale_fn is evaluated on cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    # Example
        clr = CyclicLR(base_lr=0.001, max_lr=0.006,step_size=2000., mode='triangular')
        model.fit(X_train, Y_train, callbacks=[clr])
        
    Class also supports custom scaling functions:
        # sinusoidal learning rate
        clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
        # exp sinusoidal behaviour
        clr_fn = lambda x: 1/(5**(x*0.0001))

        clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000., scale_fn=clr_fn, scale_mode='cycle')
        model.fit(X_train, Y_train, callbacks=[clr])
    
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())



