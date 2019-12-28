
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

early_stop = EarlyStopping( 
    monitor   = 'val_loss', 
    min_delta = 0.0001, 
    patience  = 5, 
    mode      = 'min', 
    verbose   = 1
)


# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'tracker.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(
    filepath, 
    monitor = 'val_loss', 
    verbose = 1, 
    mode    = 'min', 
    period  = 2
    save_best_only=True
)

lr_reducer = ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor  = np.sqrt(0.1), 
    patience= 2, 
    min_lr  = 1e-6
) 

def lr_scheduling(epoch):
    """ Learning Rate Schedule
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 100:
        lr *= 1e-3
    elif epoch > 75:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_scheduling)




# import keras
from tensorflow.keras.callbacks import *

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
        
    # Class also supports custom scaling functions:
        # sinusoidal learning rate
        clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
        # exp sinusoidal behaviour
        clr_fn = lambda x: 1/(5**(x*0.0001))

        clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000., scale_fn=clr_fn, scale_mode='cycle')
        model.fit(X_train, Y_train, callbacks=[clr])
    
    # Referance:
        https://github.com/bckenstler/CLR
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=4., gamma=1.,
                mode='triangular', scale_fn=None, scale_mode='cycle'):
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

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1-x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1-x)) * self.scale_fn(self.clr_iterations)
        
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



from sklearn.metrics import roc_auc_score
 
# class Histories(keras.callbacks.Callback):
class Histories(Callback):
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
        print(f"\r ===== roc-auc_val: {round(roc_val,4)} ========")
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        self.complete_losses.append(logs.get('loss'))
        self.complete_accs.append(logs.get('acc'))
        return
