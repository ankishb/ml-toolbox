
## Regression and classification:
In all the following model, only difference comes in the distribution. Therefore it comes under generalized linear model. 
First, we see some of these model, their modelling distribution and loss function derivation. In the next section we generalize it to all these model.
1. Linear Regression:
	- gaussian: `p(y/x,w) = exp(-(y - w x)^2)`
	- loss: `log(p(y/x,w)) = (y - w x)^2`
2. Binary classification/Logistics Regression:
	- Binomial: `p(y/x,w) = mu^y (1 - mu)^(1-y)`, `mu = 1/(1+exp(-wx))`
	- loss: `log(p(y/x,w) = ylog(mu) + (1-y)log(1-mu)`
3. Pseudo Logisitic Regression:
	- quasibinomial: `binomial dist can have 0/1 value, if we use two diff value -4/7, then quasibinomial comes handy`
4. Multiclass classification:
	- Multinomial: `p(y/x,w) = mu1^y1 mu2^y2 .... muk^yk`
	- `p(y/x,w) = exp(w1 x1) / (exp(w1 x1) + exp(w2 x2) + ... + exp(wk xk)`
	- loss: `softmax-loss `
5. Poisson regression:
	- categorical data
	- poisson: `p(y/x,w) = lambda^y exp(-lambda) / (y!)`
	- `log(lambda) = w x` -> `lambda = exp(w x)`
	- loss: `log(p(y/x,w) = ylog(lambda) - lambda = y w x - exp(w x)`

> Note: n trail of berniulli distribution creates binomial distribution


## Generalized Linear Model
There are `3` main components in GLM, which are
1. p(y/w): that is distribution of model, also called `family function` 
2. linear componenet: `f(x) -> w1 x + wo`
3. `E(y)` modelling of response variable, also called `link function`. For exp: we have an skewed distribution, we then transform the output in another form, using `log`, `power`, `gaussian` etc.

#### Family:
It represent the nature of the linear model, that we are going to fit on the observation, in the form of `f(X;theta)` In GLM, `exponential family` are chosen, it has wide variety of function, which can be used, some of them are represent follows:
1. gaussian: (Linear Regression) 
	- For numeric (Real or Int).
2. binomial: (Logistic Regression). 
	- For categorical 2 levels/classes or binary (Enum or Int).
3. ordinal: (Logistic Ordinal Regression) 
	- Requires a categorical response with at least 3 levels. (For 2-class problems, use family=”binomial”.)
4. quasibinomial: (Pseudo-Logistic Regression) 
	- For numeric.
5. multinomial: (Multiclass Classification) 
	- For categorical with more than two levels/classes (Enum).
6. poisson: (Poisson Models). 
	- For numeric and non-negative (Int).
7. gamma: (Gamma Models) 
	- For numeric and continuous and positive (Real or Int).
8. tweedie: (Tweedie Models) 
	- Tweedie distributions are a family of distributions that include gamma, normal, Poisson, and their combinations.
	- For numeric and continuous (Real) and non-negative.
	Tweedie distributions are especially useful for modeling positive continuous variables with exact zeros. The variance of the Tweedie distribution is proportional to the pth power of the mean i.e. var(y).This is parametrized by variance power `p`. It is defined for all p values except in the (0,1) interval and has the following distributions as special cases:

	p=0 : Normal
	p=1 : Poisson
	p∈(1,2) : Compound Poisson, non-negative with mass at zero
	p=2 : Gamma
	p=3 : Inverse-Gaussian
	p>2 : Stable, with support on the positive reals

9. negativebinomial: (Negative Binomial Models). 
	- For numeric and non-negative (Int).




#### Link:
Link function represent the expected value of the response y as `E(y)`. Choosing right link function is must based on problem, you must consider the `distribution of target`, while doing so. 
H2O's GLM supports the following link functions: 
1. Family_Default
2. Identity
3. Logit
4. Log
5. Inverse
6. Tweedie
7. Ologit
8. Oprobit
9. Ologlog

#### Regularization:
- alpha: L1(lasso)
- lambda: L2 (ridge)
- both: L1 + L2 (Elastic net)
