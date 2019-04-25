
# Generalized Linear Model

The main component of GLM are family and link function.

#### Family:
	
It represent the nature of the linear model, that we are going to fit on the observation, in the form of `f(X;theta)` In GLM, `exponential family` are chosen, it has wide variety of function, which can be used, some of them are represent follows:

- gaussian: (Linear Regression) The response must be numeric (Real or Int). This is the default family.
- binomial: (Logistic Regression). The response must be categorical 2 levels/classes or binary (Enum or Int).
- ordinal: (Logistic Ordinal Regression) Requires a categorical response with at least 3 levels. (For 2-class problems, use family=”binomial”.)
- quasibinomial: (Pseudo-Logistic Regression) The response must be numeric.
- multinomial: (Multiclass Classification) The response can be categorical with more than two levels/classes (Enum).
- poisson: (Poisson Models). The response must be numeric and non-negative (Int).
- gamma: (Gamma Models) The response must be numeric and continuous and positive (Real or Int).
- tweedie: (Tweedie Models) The response must be numeric and continuous (Real) and non-negative.
	Tweedie distributions are especially useful for modeling positive continuous variables with exact zeros. The variance of the Tweedie distribution is proportional to the pth power of the mean i.e. var(y).This is parametrized by variance power `p`. It is defined for all p values except in the (0,1) interval and has the following distributions as special cases:

	p=0 : Normal
	p=1 : Poisson
	p∈(1,2) : Compound Poisson, non-negative with mass at zero
	p=2 : Gamma
	p=3 : Inverse-Gaussian
	p>2 : Stable, with support on the positive reals

- negativebinomial: (Negative Binomial Models). The response must be numeric and non-negative (Int).




#### Link:

Link function represent the expected value of the response y as E(y). Choosing right link function is must based on problem, you must consider the `distribution of target`, while doing so. 
	H2O's GLM supports the following link functions: 
	Family_Default, Identity, Logit, Log, Inverse, Tweedie, Ologit, Oprobit, and Ologlog


#### Regularization:

- alpha: L1(lasso)
- lambda: L2 (ridge)
- both: L1 + L2 (Elastic net)