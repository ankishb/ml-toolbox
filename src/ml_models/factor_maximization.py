
class FM_classifier(BaseEstimator, ClassifierMixin):
    """
    Args:
        n_iter: no of iteration used for training
        n_latents: dims of latent factor
        lr_rate: learning rate for sgd algorithm
        reg_w: regularization for weight matrix
        reg_v: regularization for feature's latent factors

    References:
        http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """
    def __init__(self, n_iters, n_factors, n_features, lr_rate, reg_w, reg_v):
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.lr_rate = lr_rate
        self.n_latents = n_factors
        self.n_features = n_features
        self.n_ietrs = n_iters

    def fit(self, X, Y, features_list_per_sample):
        n_samples, n_features = X.shape
        self.feature_indices = features_list_per_sample
        self.w = np.zeros(n_features)
        self.w0 = 0.0

        np.random.seed(1234)
        self.w = np.random.normal(
            scale=1/np.sqrt(self.n_features),
            size=(n_features)
        )
        self.v = np.random.normal(
            scale=1/np.sqrt(self.n_latents),
            size=(self.n_latents, n_features)
        )

        Y = Y.astype(np.int32)
        Y[Y == 0] = -1 # helpful for pred and grad calculation
        
        self.history = []
        for i in range(self.n_iters):
            loss = sgd_update(X, Y, i, self.w0, self.w, self.v, self.feature_indices, 
                              self.n_feature, self.n_latent, lr_rate=self.lr_rate, 
                              reg_w=self.reg_w, sreg_v=elf.reg_v)
            self.history.append(loss)

        return self

    def _predict(self, X):
        """Similar to predict_single_sample, but in vectorize form"""
        linear_output = X * self.w
        v = self.v.T
        term = (X * v) ** 2 - (X.power(2) * (v ** 2))
        factor_output = 0.5 * np.sum(term, axis = 1)
        return self.w0 + linear_output + factor_output

    def predict_proba(self, X):
        """ predicting probabilities for given input data X
        Args:
            X : dense or sparse matrix of shape [n_samples, n_features]

        Returns:
            probs : 2d ndarray of shape [n_samples, n_classes]
        """
        preds = self._predict(X)
        pred_probs = 1.0/(1.0 + np.exp(-pred))
        probs = np.vstack((1 - pred_probs, pred_probs)).T
        return probs


    def loos_and_grad(y_hat, y):
    """Calculate logloss and its grad"""
    loss = np.log(1.0 + np.exp(- y * y_hat))
    grad = y / (1.0 + np.exp(y * y_hat))
    return loss, grad


def predict_single_sample(data, idx, w0, w, v, features_idx, n_feature, n_latent):
    """
    Args:
        Data: X
        idx: index
        w0: bias
        w: weight vector of shape [n_feature]
        v: weight matrix of shape [n_feature, n_latent]
        features_idx: features-index list for the current sample
        n_latent: latent feature for v
        n_feature: no of features

    Expression: 
        y_hat(xi) = wo + (wi * xi) + 1/2 sum_k [(vi_k * xi)^2 - (vi_k)^2 * (xi^2)]

    Return:
        pred: Prediction of sample using the above expression
        vixi_store: will be use for gradient calculation
    """
    vixi_store = np.zeros(n_latent)
    pred = w0 + data[idx] * w[features_idx]

    for factor in range(n_latent):
        vixi = data[idx] * v[features_idx, factor]
        vi2 = v[features_idx, factor] * v[features_idx, factor]
        xi2 = data[idx] * data[idx]
        vi2xi2 = vi2 * xi2
        vixi2 = vixi * vixi
        
        vixi_store[factor] = vixi
        pred += 0.5 * (vixi2 - vi2xi2)

    return pred, vixi_store


def sgd_update(X, Y, idx, w0, w, v, feature_indices, n_feature, n_latent,
              lr_rate=0.001, reg_w=0.001, reg_v=0.001):
    loss = 0.0
    n_samples = data.shape[0]
    for i in range(n_samples):
        features_idx = feature_indices[i]
        pred, vixi_store = predict_single_sample(X[i], i, w0, w, v, features_idx, n_feature, n_latent)

        loss, grad = log_loss(pred, Y[i])
        w0 -= lr_rate * grad
        w[features_idx] -= lr_rate * (grad * X[i] + 2 * reg_w * w[features_idx])

        for factor in range(n_latent):
            term = vixi_store[factor] - X[i] * v[feature_idx, factor]
            v_grad = grad * X[i] * term
            v[features_idx, factor] -= lr_rate * (v_grad + 2 * reg_v * v[features_idx, factor])

    loss = loss/n_samples
    return loss


