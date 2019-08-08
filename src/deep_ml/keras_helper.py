
from sklearn.utils import class_weight

def get_class_weights(y):
    """ 
    Example:
        model.fit(X_t, y, batch_size=10, epochs=2,validation_split=0.1,sample_weight=sample_wts)
    
    """
    return class_weight.compute_sample_weight('balanced', y)
