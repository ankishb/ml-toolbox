

from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])


    
Parameter Expressions

The stochastic expressions currently recognized by hyperopt's optimization algorithms are:

    hp.choice(label, options)
        Returns one of the options, which should be a list or tuple. The elements of options can themselves be [nested] stochastic expressions. In this case, the stochastic choices that only appear in some of the options become conditional parameters.

    hp.randint(label, upper)
        Returns a random integer in the range [0, upper). The semantics of this distribution is that there is no more correlation in the loss function between nearby integer values, as compared with more distant integer values. This is an appropriate distribution for describing random seeds for example. If the loss function is probably more correlated for nearby integer values, then you should probably use one of the "quantized" continuous distributions, such as either quniform, qloguniform, qnormal or qlognormal.

    hp.uniform(label, low, high)
        Returns a value uniformly between low and high.
        When optimizing, this variable is constrained to a two-sided interval.

    hp.quniform(label, low, high, q)
        Returns a value like round(uniform(low, high) / q) * q
        Suitable for a discrete value with respect to which the objective is still somewhat "smooth", but which should be bounded both above and below.

    hp.loguniform(label, low, high)
        Returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed.
        When optimizing, this variable is constrained to the interval [exp(low), exp(high)].

    hp.qloguniform(label, low, high, q)
        Returns a value like round(exp(uniform(low, high)) / q) * q
        Suitable for a discrete variable with respect to which the objective is "smooth" and gets smoother with the size of the value, but which should be bounded both above and below.

    hp.normal(label, mu, sigma)
        Returns a real value that's normally-distributed with mean mu and standard deviation sigma. When optimizing, this is an unconstrained variable.

    hp.qnormal(label, mu, sigma, q)
        Returns a value like round(normal(mu, sigma) / q) * q
        Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.

    hp.lognormal(label, mu, sigma)
        Returns a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed. When optimizing, this variable is constrained to be positive.

    hp.qlognormal(label, mu, sigma, q)
        Returns a value like round(exp(normal(mu, sigma)) / q) * q
        Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is bounded from one side.

2.2 A Search Space Example: scikit-learn

To see all these possibilities in action, let's look at how one might go about describing the space of hyperparameters of classification algorithms in scikit-learn. (This idea is being developed in hyperopt-sklearn)

from hyperopt import hp
space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
    },
    {
        'type': 'svm',
        'C': hp.lognormal('svm_C', 0, 1),
        'kernel': hp.choice('svm_kernel', [
            {'ktype': 'linear'},
            {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
            ]),
    },
    {
        'type': 'dtree',
        'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('dtree_max_depth',
            [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
        'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
    },
    ])