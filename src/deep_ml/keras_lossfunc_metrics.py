


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)


def logcosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.
    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.
    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
    # Returns
        Tensor with one scalar loss entry per sample.
    """
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    return K.mean(_logcosh(y_pred - y_true), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def sparse_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)









def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)









class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}


class MaxNorm(Constraint):
    """MaxNorm weight constraint.
    Constrains the weights incident to each hidden unit
    to have a norm less than or equal to a desired value.
    # Arguments
        max_value: the maximum norm for the incoming weights.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    def __init__(self, max_value=2, axis=0):
        self.max_value = max_value
        self.axis = axis

    def __call__(self, w):
        norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
        desired = K.clip(norms, 0, self.max_value)
        w *= (desired / (K.epsilon() + norms))
        return w

    def get_config(self):
        return {'max_value': self.max_value,
                'axis': self.axis}


class NonNeg(Constraint):
    """Constrains the weights to be non-negative.
    """

    def __call__(self, w):
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        return w


class UnitNorm(Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.
    # Arguments
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        return w / (K.epsilon() + K.sqrt(K.sum(K.square(w),
                                               axis=self.axis,
                                               keepdims=True)))

    def get_config(self):
        return {'axis': self.axis}


class MinMaxNorm(Constraint):
    """MinMaxNorm weight constraint.
    Constrains the weights incident to each hidden unit
    to have the norm between a lower bound and an upper bound.
    # Arguments
        min_value: the minimum norm for the incoming weights.
        max_value: the maximum norm for the incoming weights.
        rate: rate for enforcing the constraint: weights will be
            rescaled to yield
            `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
            Effectively, this means that rate=1.0 stands for strict
            enforcement of the constraint, while rate<1.0 means that
            weights will be rescaled at each step to slowly move
            towards a value inside the desired interval.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis

    def __call__(self, w):
        norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
        desired = (self.rate * K.clip(norms, self.min_value, self.max_value) +
                   (1 - self.rate) * norms)
        w *= (desired / (K.epsilon() + norms))
        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value,
                'rate': self.rate,
                'axis': self.axis}


# Aliases.

max_norm = MaxNorm
non_neg = NonNeg
unit_norm = UnitNorm
min_max_norm = MinMaxNorm


# Legacy aliases.
maxnorm = max_norm
nonneg = non_neg
unitnorm = unit_norm










def softmax(x, axis=-1):
    """Softmax activation function.
    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)


def elu(x, alpha=1.0):
    """Exponential linear unit.
    # Arguments
        x: Input tensor.
        alpha: A scalar, slope of negative section.
    # Returns
        The exponential linear activation: `x` if `x > 0` and
        `alpha * (exp(x)-1)` if `x < 0`.
    # References
        - [Fast and Accurate Deep Network Learning by Exponential
           Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    """
    return K.elu(x, alpha)


def selu(x):
    """Scaled Exponential Linear Unit (SELU).
    SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
    are predefined constants. The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `lecun_normal` initialization) and the number of inputs
    is "large enough" (see references for more information).
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # Returns
       The scaled exponential unit activation: `scale * elu(x, alpha)`.
    # Note
        - To be used together with the initialization "lecun_normal".
        - To be used together with the dropout variant "AlphaDropout".
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * K.elu(x, alpha)


def softplus(x):
    """Softplus activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The softplus activation: `log(exp(x) + 1)`.
    """
    return K.softplus(x)


def softsign(x):
    """Softsign activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The softsign activation: `x / (abs(x) + 1)`.
    """
    return K.softsign(x)


def relu(x, alpha=0., max_value=None, threshold=0.):
    """Rectified Linear Unit.
    With default values, it returns element-wise `max(x, 0)`.
    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.
    # Arguments
        x: Input tensor.
        alpha: float. Slope of the negative part. Defaults to zero.
        max_value: float. Saturation threshold.
        threshold: float. Threshold value for thresholded activation.
    # Returns
        A tensor.
    """
    return K.relu(x, alpha=alpha, max_value=max_value, threshold=threshold)


def tanh(x):
    """Hyperbolic tangent activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The hyperbolic activation:
        `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
    """
    return K.tanh(x)


def sigmoid(x):
    """Sigmoid activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The sigmoid activation: `1 / (1 + exp(-x))`.
    """
    return K.sigmoid(x)


def hard_sigmoid(x):
    """Hard sigmoid activation function.
    Faster to compute than sigmoid activation.
    # Arguments
        x: Input tensor.
    # Returns
        Hard sigmoid activation:
        - `0` if `x < -2.5`
        - `1` if `x > 2.5`
        - `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.
    """
    return K.hard_sigmoid(x)


def exponential(x):
    """Exponential (base e) activation function.
    # Arguments
        x: Input tensor.
    # Returns
        Exponential activation: `exp(x)`.
    """
    return K.exp(x)


def linear(x):
    """Linear (i.e. identity) activation function.
    # Arguments
        x: Input tensor.
    # Returns
        Input tensor, unchanged.
    """
	return x



softmax
elu
selu
softplus
softsign
relu
tanh
sigmoid
hard_sigmoid
exponential
linear







def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def sparse_categorical_accuracy(y_true, y_pred):
    # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    # convert dense predictions to labels
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred_labels = K.cast(y_pred_labels, K.floatx())
    return K.cast(K.equal(y_true, y_pred_labels), K.floatx())


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.cast(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), K.floatx())


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    # If the shape of y_true is (num_samples, 1), flatten to (num_samples,)
    return K.cast(K.in_top_k(y_pred, K.cast(K.flatten(y_true), 'int32'), k),
                  K.floatx())


# Aliases

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity






sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam
adamax = Adamax
nadam = Nadam





class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.
    """

    def __call__(self, shape, dtype=None):
        return K.constant(0, shape=shape, dtype=dtype)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.
    """

    def __call__(self, shape, dtype=None):
        return K.constant(1, shape=shape, dtype=dtype)


class Constant(Initializer):
    """Initializer that generates tensors initialized to a constant value.
    # Arguments
        value: float; the value of the generator tensors.
    """

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=None):
        return K.constant(self.value, shape=shape, dtype=dtype)

    def get_config(self):
        return {'value': self.value}


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.
    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, mean=0., stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=None):
        return K.random_normal(shape, self.mean, self.stddev,
                               dtype=dtype, seed=self.seed)

    def get_config(self):
        return {
            'mean': self.mean,
            'stddev': self.stddev,
            'seed': self.seed
        }


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.
    # Arguments
        minval: A python scalar or a scalar tensor. Lower bound of the range
          of random values to generate.
        maxval: A python scalar or a scalar tensor. Upper bound of the range
          of random values to generate.  Defaults to 1 for float types.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=None):
        return K.random_uniform(shape, self.minval, self.maxval,
                                dtype=dtype, seed=self.seed)

    def get_config(self):
        return {
            'minval': self.minval,
            'maxval': self.maxval,
            'seed': self.seed,
        }


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.
    These values are similar to values from a `RandomNormal`
    except that values more than two standard deviations from the mean
    are discarded and redrawn. This is the recommended initializer for
    neural network weights and filters.
    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, mean=0., stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=None):
        return K.truncated_normal(shape, self.mean, self.stddev,
                                  dtype=dtype, seed=self.seed)

    def get_config(self):
        return {
            'mean': self.mean,
            'stddev': self.stddev,
            'seed': self.seed
        }


class VarianceScaling(Initializer):
    """Initializer capable of adapting its scale to the shape of weights.
    With `distribution="normal"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:
        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"
    With `distribution="uniform"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
    # Arguments
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "normal", "uniform".
        seed: A Python integer. Used to seed the random generator.
    # Raises
        ValueError: In case of an invalid value for the "scale", mode" or
          "distribution" arguments.
    """

    def __init__(self, scale=1.0,
                 mode='fan_in',
                 distribution='normal',
                 seed=None):
        if scale <= 0.:
            raise ValueError('`scale` must be a positive float. Got:', scale)
        mode = mode.lower()
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument: '
                             'expected on of {"fan_in", "fan_out", "fan_avg"} '
                             'but got', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype=None):
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self.distribution == 'normal':
            # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(scale) / .87962566103423978
            return K.truncated_normal(shape, 0., stddev,
                                      dtype=dtype, seed=self.seed)
        else:
            limit = np.sqrt(3. * scale)
            return K.random_uniform(shape, -limit, limit,
                                    dtype=dtype, seed=self.seed)

    def get_config(self):
        return {
            'scale': self.scale,
            'mode': self.mode,
            'distribution': self.distribution,
            'seed': self.seed
        }


class Orthogonal(Initializer):
    """Initializer that generates a random orthogonal matrix.
    # Arguments
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to seed the random generator.
    # References
        - [Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks](http://arxiv.org/abs/1312.6120)
    """

    def __init__(self, gain=1., seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype=None):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        rng = np.random
        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
        a = rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # Pick the one with the correct shape.
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return self.gain * q[:shape[0], :shape[1]]

    def get_config(self):
        return {
            'gain': self.gain,
            'seed': self.seed
        }


class Identity(Initializer):
    """Initializer that generates the identity matrix.
    Only use for 2D matrices.
    If the desired matrix is not square, it pads with zeros on the
    additional rows/columns
    # Arguments
        gain: Multiplicative factor to apply to the identity matrix.
    """

    def __init__(self, gain=1.):
        self.gain = gain

    def __call__(self, shape, dtype=None):
        if len(shape) != 2:
            raise ValueError(
                'Identity matrix initializer can only be used for 2D matrices.')

        return self.gain * K.eye((shape[0], shape[1]), dtype=dtype)

    def get_config(self):
        return {
            'gain': self.gain
        }


def lecun_uniform(seed=None):
    """LeCun uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(3 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        - [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """
    return VarianceScaling(scale=1.,
                           mode='fan_in',
                           distribution='uniform',
                           seed=seed)


def glorot_normal(seed=None):
    """Glorot normal initializer, also called Xavier normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    return VarianceScaling(scale=1.,
                           mode='fan_avg',
                           distribution='normal',
                           seed=seed)


def glorot_uniform(seed=None):
    """Glorot uniform initializer, also called Xavier uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    return VarianceScaling(scale=1.,
                           mode='fan_avg',
                           distribution='uniform',
                           seed=seed)


def he_normal(seed=None):
    """He normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
           ImageNet Classification](http://arxiv.org/abs/1502.01852)
    """
    return VarianceScaling(scale=2.,
                           mode='fan_in',
                           distribution='normal',
                           seed=seed)


def lecun_normal(seed=None):
    """LeCun normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(1 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
        - [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """
    return VarianceScaling(scale=1.,
                           mode='fan_in',
                           distribution='normal',
                           seed=seed)


def he_uniform(seed=None):
    """He uniform variance scaling initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
           ImageNet Classification](http://arxiv.org/abs/1502.01852)
    """
    return VarianceScaling(scale=2.,
                           mode='fan_in',
                           distribution='uniform',
                           seed=seed)


# Compatibility aliases

zero = zeros = Zeros
one = ones = Ones
constant = Constant
uniform = random_uniform = RandomUniform
normal = random_normal = RandomNormal
truncated_normal = TruncatedNormal
identity = Identity
orthogonal = Orthogonal


