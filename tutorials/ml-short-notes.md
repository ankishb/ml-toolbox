

# perceptron
- Online Algo (check one example at a time)
- Error driven (update weights only if prediction is wrong)
- **Linear Boundary Seprator**
- Descision rule to update weights `y_hat*y <= 0` as `y C [-1,1]` and weight update rule is `w += y*x` and `b += y`.
- Hyperplane will be perpendicular to weight vector, because `W*X = 0`, when `bias` is zero.
- Direction of weights is towards `+ve` samples.
- `W*X` is nothing but the `projection` of sample `x` on the `W` vector. So it tells the distance of how far it is from origin of hyperplane, when bias is zero.
- For `X` to be `D` dims space, hyperplane will always be `D-1` dims. For example for `2D` data points, `decision boundary` is `linear`.
- **Interpretation**: Scale the data in `[0-1]` range, now weights will represent the `sensitivity` of classification prediction on the features. `More weights of feature, more sensitive to output`.
- For `+ve` class, features with higher +ve weights are more responsible and for `-ve` class, feature with higher -ve weights are more sensitive. If features have some noise, then it will be more appropriate to remove such feature, if it is more sentive for boundary or prediction.
- Issue with perceptron that it consider `later points more than former points`.
- **Average Perceptron** And **Voting Perceptron**, **What????**
- Researcher worked on XOR problem for decades using perceptron, which leads to AI-Winter.

Note: `kernel` is here to saved for non-linear boundary.


Cat var: Qualitive variable
Num var: Quantitative Var

t-statistics:
Final the coeeficient of feature in model and also find the std dev error and t-stat = (coeff/std-dev error)


# logistic regeression
- Linear regression try to put a linear line on samples, **Just imagine**, Can a linear line give a solution which seprate **[0,1]**. **Well, it can, Look at following exp**, because we need a threshold to decide which side the samples lie. But for biased/imbalanced dataset, the linear line will be biased along one side and solution may not be good, whereas logistic regression helps to put a sigmoid like curve on the samples, which seems good.

Fit a line though the following sample and analyze the threshold of 0.5 to detect the test samples.

||=========================||=========================|| 
||1         ******         ||         ******      *   || 
||                         ||                         || 
||0 ******                 ||    *****                || 
||=========================||=========================||
||   Linear fit well       ||     Linear line biased  || 
||      GOOD               ||          BAD            ||
||=========================||=========================||

# Log-reg Conti.
- log-reg, output is always between **0-1**, where in linear reg, output can be anything **>1 & <0**.
- log-reg computes probability of being 1.
- optimization problem.

# Ridge Regression:
- L2 regularization
- same solution as least square with **lambda I in inverse**, it also make inverse possible, if matrix is low rank.
- early stopping (not a regularization, but it helps in same way)


# optimization
- Gradient decent:
    need to pick learning rate
- conjugate gd:
    very fast
    no hyperparameter
    more complex

# Convex function
- if second derivative is always **positive**.
- if hessian is positive definite.

---



## Unbalanced Dataset:
There seems to be some confusion about calculating class weights, as this dataset is very imbalanced. This dataset has ~0.247% 1s, and the rest are 0s (~99.753%). scale_pos_weight of LightGBM is "weight of positive class in binary classification task" according to LightGBM documentation. I think that translates into a multiplication factor that has to be applied to number of 1s in order to get the same sample number as in 0s. So: 99.753 / 0.247 = ~ 403.8

```python
import pandas as pd
from collections import Counter

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}

train = pd.read('train.csv')
class_weights = get_class_weights(train.is_attributed.values)
print(class_weights)

Out: {0: 1.0, 1: 403.74}
```

## Percentile calculation:
- find the percentile score in exam or test.
1. Sort the data values from low to high
2. Multiply the percentile with total number of value. For 25 student and 99percentile, it will be 25*0.99.
3. Round to nearest whole number. 1.5-->2, 2.7-->3, 1.2-->1, 5-->5
4. Pick the value at that index, given by step 3

## Find Inverse of matrix:
- use GAUSS ELIMINATION METHOD
- Procede like following: **a11, a21, a31, -- a22, a32, a33, -- a23, a13, a12**
- In brief, proceed through each columns, but lower left triangle, and then move upward and take left turn, it reach to a12.
```c++
h := 1 /* Initialization of the pivot row */
 k := 1 /* Initialization of the pivot column */
 while h ≤ m and k ≤ n
   /* Find the k-th pivot: */
   i_max := argmax (i = h ... m, abs(A[i, k]))
   if A[i_max, k] = 0
     /* No pivot in this column, pass to next column */
     k := k+1
   else
      swap rows(h, i_max)
      /* Do for all rows below pivot: */
      for i = h + 1 ... m:
         f := A[i, k] / A[h, k]
         /* Fill with zeros the lower part of pivot column: */
         A[i, k]  := 0
         /* Do for all remaining elements in current row: */
         for j = k + 1 ... n:
            A[i, j] := A[i, j] - A[h, j] * f
      /* Increase pivot row and column */
      h := h+1 
      k := k+1
```

## Optimization (Part-1):
- A function is convex, when **f(y) >= f(x)+dy/dx (y-x)**, which is nothing but the value at function is always greater than its tangent.
- Convex function has its second derivative(hessian) as semi-definite.
```proof
At critical point, f'(x) = 0;
using taylor series: f(x+h) = f(x) + h*f'(x) + 1/2 h*f''(x)*h
f(x+h) = f(x) + 1/2 h*f''(x)*h, as f'(x) = 0
f(x+h) - f(x) = 1/2 h*f''(x)*h = hessian
For local minima to exist, h should be >=0,
So hessian >=0, which is positive semidefinte
```

## gradient Desent:
- can optimize any function, **convex/non-convex**
- the idea is to take gradient and move in the opposite direction of it. **As grad tells us the direction in which slope is increasing**.
- we also use **learning rate** to give less signifies to update using current gradient.

## learning rate:
- constant
- momentum based **1/t, 1/sqrt(t), 1/(1-t), RMSProp< Nestrov Momentum**
- adaptive lr **Adam, AdaGrad,...**
- cyclic learning rate **very effective**, the idea is insetead of finding one local minima, it find many many extrema and build the ensemble using all those optima pts.

## Gradient Based Optimization:
- GD: update as per the gradient of all data
    - GD converge to optimal at the rate of **(1/k)**, which means if you need an accuracy of **1e-4**, then it need something on the order of one thousand steps.
- SGD: Take one random sample and upadte acc to that, it will have **high variance**
- Batch GD: **Reduce Variance**
- Subgradient: At non-differential point, we get a range by looking at grad value on the right and left, then pick a value and pretend it like a differential function
- Constrained opt (Lagrangian Based && Projected Grad)
- Coordinate Descent Algo: Where we update one dimension feature from D-dim space
- Alternative Optimization: **Exp: EM**
- Newton Method(Second Order Method, another is L-BFGS):
    1. It tells about its curvature, shape, etc
    2. Each step is finding the minima of **quadratic function** in local space.
    3. No need of learning rate
    4. As f(y) = f(x) + (y-x) df/dx + (y-x)^2 d^2f/dx^2
    f(wt+1) =f(wt)+df/dwt (w-wt) + 1/2 (w-wt)^2 d^2f/dw^2
    **w = argmax_w f(wt+1)**
    **wt+1 = wt + (hessian)^-1 grad**
    5. Expensive Because of hessian
    6. Very Fast, if f(w) is convex

> double derivative tell about curvature, which decide the learning rate. As if the surface is getting less steeper, then the learning step is decreased.

## Deep Learning Famously Optimizer: [Best](http://ruder.io/optimizing-gradient-descent/)
- SGD, RMSProp, AdaGrad, AdaDelta, Adam, Nadam
- For time series ==> prefereably RMSPRop
- Best to use ==> SGD+Nestrov or Adam
- Nestrov ==> First jump and then correct its step(or jump)
- sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
- Batch normalization additionally acts as a regularizer, reducing the need for Dropout.
- For sparse data (tf-idf) ==> AdaGrad, AdaDelta(improved over AdaGrad)
    Best, if we are using sparse data such as tf-idf features for words.

- Momentum:
**Another way to think about momentum, suppose our update as oscilating while moving to its local minima, so when we taking average of many points value, its oscillation in verticle direction suppress, but in horizontal direction as all grad effect sums up, so it will have nice big grad value.**

```
V_w = beta V_w + (1-beta) dw
V_b = beta V_b + (1-beta) db

w = w - alpha V_w
b = b - alpha V_b
```
The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation.



- AdaGrad: weakness: lr_rate decaying
Notice that the weights that receive high gradients will have their effective learning rate reduced, while weights that receive small or infrequent updates will have their effective learning rate increased. This agressive behaviour, stops Deep-NN to learn very early.

- RMSProp:
The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate. In particular, it uses a moving average of squared gradients instead. Hence, RMSProp still modulates the learning rate of each weight based on the magnitudes of its gradients, which has a beneficial equalizing effect, but unlike Adagrad the updates do not get monotonically smaller

- AdaDelta: Removed weakness of adadelta
Best, if we are using sparse data such as tf-idf features for words.

> If our data is sparse and our features have very different frequencies, we might not want to update all of them to the same extent, but perform a larger update for rarely occurring features.


## lagrangian Method:
- Primal & Dual method: Both gives same answer if the constrained function is convex, i.e. **g(w) <= 0**
- **w = argmin_w{f(w) + argmax_a{a . g(w)}}**
- For dual solution, α*g(w) = 0 (complimentary slackness/Karush-Kuhn-Tucker (KKT) condition)

## Projected Grad Method:
1. Each step of projected GD works as follows
2. Do the usual GD update: z(t+1) = w(t) − ηtg(t)
3. Check z(t+1) for the constraints
    If z(t+1) 2 C, w(t+1) = z(t+1)
    If z(t+1) 2 C = , project on the constraint set: w(t+1) = ΠC[z(t+1)]

- Example: let's suppose our z(t+1), lie outside the unit circle, but just consider it to be 1, if our constrained of **w belong to unit circle**.


## Loss function (**Convex Surrogate Losses Function**):
- Surrogate losses, because it define the upper bound on the real losses and minimize that. So minimze the surrogate losses function make sure pushing down real losses too.
- **0/1 loss**: 1[ywx <= 0], this is NP hard, no efficient solution. For example for wx = -0.0000001, with increase of 00000009 it will still be -1, but 000000011 increase will take it to right side that is +1.
- **Perceptron**: max(0, -ywx)
- **Hinge Loss**: max(0, 1-ywx)
- **logistic**: log(1+exp(-ywx))/log2
- **Exponential Loss**: exp(-ywx)
- **Absolute loss**: |y - wx| **Robust to outliers**
- **epsilon-insensitive loss**: |y-wx| + epsilon

## Hyperplane:
- **W x + b = 0**, a hyperplane
    1. **b==0**, then plane is passing through origin
    2. **b>0**, hyperplane moves parallel in direction of w
    3. **b<0**, hyperplane moves opposite direction


# regularization:
- avoid overfitting
- better generalization

### Types of regularization
- **L_p Norm**:
    - p < 1, it shrink from diamond to shape, which stretches towards axes, it has sparsity nature, but also non-convex, so non preferable generally.
    - p = 1 ==> diamond, this is least value of p, with convexity properties, Also have sparse nature.
    - p = 2 ==> euclidean loss, helps in generalize better, by reducing weights so much that boundriess becomes smooth
    - **Why small weights are prefered, because by changing the input x by epsilon, its prediction should not change, which is possible with smalle weights only.**
    - p > 2, it move from circle to square as p = INF
    - p = INF, also called max norm.
- **L0**:
    - point shape
    - NP hard
    - Count of non-negative feature value.
- **L1**:
    - **lasso Regression**
    - Diamond shape(if the linear predictor or line of solution meet the corner point, it creats sparsity in solution)
    - sparse feaure
    - deal with multicolinearity problem
- **L2**:
    - **Ridge** regression
    - circular shape
    - small weights **if there is any noise in feature, then having small weights generally doesn't effect too much**
- **Dropout** Reg (in deep learning)
- for unsupervised learning:
    - sum_i w_i*(x_i - x_j)^2
- As p decrease in **L_p**, the shape of l_p shrink from edge(imagine dianmond and shrink its edges.) and tends to have sharp corners.

# Impotant Points
- Non-Linear Boundary using linear predictors, for circular dataset, using linear predictor, such as **f(w0 + w1*x1 + w2*x2)** will not work, But if we use **f(w0 + w1*x1^2 + w2*x2^2)**, it can find a circular boundary. **w0** helps in finding the threshold, the radius of circle.


## Time complexity of matrix multiplication:
- one is [m X n] and other is [n X p], the time complexity will be **[mnp]**

## Linear Regression:
- closed form solution is **[w = (X' X + lambda I_d)^(-1) X' Y]**, the time analysis is as 
    1. X'X : DND
    2. X' X + lambda I_d : D D
    3. Inverse of D X D matrix is D^3
    4. then [DXD][DXN][NX1] will be DDN
    5. [DXN][NX1] will be DN 
    - Overall O(DDD + DDN) == O(DDN)
- With GD, it will be O(KND), where k is number of iteration.


## Inductive Bias: 
- Inductive bias is the set of assumptions a learner uses to predict results given inputs it has not yet encountered.



## SVM (Maximum margin hyperplane): 
- **ywx > gamma**, it add a pre-specified margin.
- For gamma=0, it becomes perceptron
- Reason behind the name \Support Vector Machine"?
    1. SVM optimization discovers the most important examples (called \support vectors") in training data
    2. These examples act as \balancing" the margin boundaries (hence called \support")
- margin(gamma) is distance as **(wx+b) / ||w||**
- to maximize the margin, we minimize the **||w||**

## soft-margin SVM:
- Very small C: Large margin but also large training error.
- Very large C: Small training error but also small margin.
- C controls the trade-off between large margin and small training error

- Dual formaulation, where we have argmax_a (a.1 - a G a)
- The dual formulation is nice due to two primary reasons:
Allows conveniently handling the margin based constraint (via Lagrangians)
Important: Allows learning nonlinear separators by replacing inner products (e.g., Gmn = ym yn xm xn)
by kernelized similarities (kernelized SVMs)

> One Biggest Advantage of Soft Margin SVM over hard margin, is that there is always going to be some solution, whether it is linearly seperable or not. **For example, lets suppose data is not linearly seperable, so hard margin will give up, but in soft margin, it use slacks to incorporate some mistakes/error, and will come up with a solution anyway.**

> What if we have solution, w and b, but not slacks, can we find it, **YES**, slacks's value is nothing but the hinge loss on the corresponding example.
slack = |0                   if yn (w xn + b) >= 1
        |1 - yn (w xn + b)   otherwise

> Soft Margin loss, is same as regularized hinge loss.
min_{w,b} = ||w||^2 + C sum_n hingle_loss(yn, (w xn + b))
1. first term --> large margin
2. 2nd term   --> small slack


## Multiclass SVM:
- It use K weight vector, one for each class
- maximum margin problem will be same except the fact that now, loss will include all weight vector
- want score w.r.t correct class to be at least 1 more than score w.r.t all other classes
- **W_y X >= 1 + w_yhat X** yhat --> incorrect class and y --> correct class
- Same as Multiclass Hinge loss as **max{0, 1+max{k=yhat}(W_k X - W_y X)}**

1. One vs All:
    - For k classes, we make k binary classifier
2. All pairs:
    - Choose pairs of 2 from k, so in total we will have **K(k-1)/2** classifiers, Not very practicals for large number of classes.


## One Class Clf (Outlier/Novelty detection): 
1. Support Vector Data Description:
    - Assume positives lie within a ball with smallest possible radius (and allow slacks)
2. One-Class SVM:
    - Find a max-marg hyperplane separating positives from origin (representing negatives)

## Support vector Regression:
- epsilon insentive loss
- nonlinear regression can be made my kernel trick.

---

## Nonlinearity handling (kernel trick):
1. Separte 1-D data of 2 classes, 
    - Exp: .......****.......
    - Project data into 2 dimension of shape parabola or upper triangle as 
    1. x --> [x, x^2]
    2. x --> [x, |x|]
2. Seperate circular data
    - [x1, x2] --> [x1^2, x1*x2, x2^2]
3. Possible Mapping:
    - x --> x^2, x^3, x^n
    - cos(x), log(n)
    - any function


## kernel:
- Implicit mapping for data, which means we operate an function on input, which automatically compute operation on higher dimesional space. 
- Exp: K(X,Y) = (X*Y)^2, where X and Y are 2 dim feature. With this operation, it implicity have **dot([x1^2, x1*x2, x2^2], [y1^2, y1*y2, y2^2])**, ie **phi(X).phi(Y)**
> Note: We didn't need to define the phi operation explicitly.

Imp Points:
1. Any kernel function, which map X-->F, with mapping as **phi**, should satisfy **mercer condition**

> F need to be a vector space, with dot product operaion defined on it, F space is also called hilbert space.

## Mercer’s Condition (Kernel Functions)
For k to be a kernel function
1. k must define a dot product for some Hilbert Space F
2. Above is true if k is symmetric and positive semi-definite (p.s.d.) function (though there are
exceptions; there are also "indefinite" kernels).
    - The function k is p.s.d. if the following holds
    **integ integ f (x)k(x; z)f (z)dxdz ≥ 0**

## properties of kernel function:
Let k1, k2 be two kernel functions then the following are as well:
1. k(x; z) = k1(x; z) + k2(x; z): direct sum
2. k(x; z) = αk1(x; z): scalar product
3. k(x; z) = k1(x; z)k2(x; z): direct product
Kernels can also be constructed by composing these rules

## Types of kernel:
1. Linear kernel: X*Y (identity mapping)
2. Quadratic kernel: (X*Y)^2 or (1 + X*Y)^2
3. Polynomial kernel: (1 + X*Y)^d
4. Radial Basis Function: (Gaussian kernel) : exp(-gamma |X-Y|^2)
    - infinite dim basis function (implicitly)
    - Also called stationary kernel, as distance between X and Y is constant

## Kernel Matrix:
- nXn matrix, which is pairwise similarity between n samples
- K is a symmetric and positive definite matrix
- For a P.D. matrix: z>Kz > 0; (also, all eigenvalues positive)
- Also known as the Gram Matrix


## Kernel tricks
1.  Any learning model in which, during training and test, inputs only appear as dot products (xi. xj) can be kernelized (i.e., non-linearlized), by replacing the xi.xj terms by φ(xi).φ(xj) = k(xi, xj)
2. Most learning algorithms can be easily kernelized
    - Distance based methods, Perceptron, SVM, linear regression, etc.
    - Many of the unsupervised learning algorithms too can be kernelized (e.g., K-means clustering, Principal Component Analysis, etc. - will see later)



## Recurrent Neural Network:
- weight sharing across the sequence
    - It create dependency across the data in sequence, where as with different weights across sequence, will cause independet decision in final prediction.
- Two main problem: 
    1. vanishing Grad (Most Imp) (cured by LSTM, but expensive)
    2. Exploding Grad (can handled by grad clipping)

## LSTM:
1. forget Gate (using sigmoid)
2. Input Gate (using sigmoid)
3. **ct_hat** Current cell state (using tanh)
4. Update Cell state (ft*c{t-1} + it*ct_hat)
5. Output gate (using sigmoid)
6. ht hidden state = ot*tanh(ct)

## LSTM varuiant:
1. LSTM with PeepHole:
2. LSTM with cooupling between forget and input gate
3. GRU
    1. reset gate
    2. update gate
    3. no cell state

    **The GRU unit controls the flow of information like the LSTM unit, but without having to use a memory unit. It just exposes the full hidden content without any control.**
    A GRU has two gates, an LSTM has three gates.
    GRUs don’t possess and internal memory (c_t) that is different from the exposed hidden state. They don’t have the output gate that is present in LSTMs.
    The input and forget gates are coupled by an update gate z and the reset gate r is applied directly to the previous hidden state. Thus, the responsibility of the reset gate in a LSTM is really split up into both r and z.
    We don’t apply a second nonlinearity when computing the output.

---

## Word Embedding: [Best Blog](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html)
- count based:
    1. Unsupervised method
    2. Use raw co-occurance count matrix
    3. Run factor model/PCA/SVD etc to find the comprssed representation of word, with the assumption that word with similar context share same symatics meaning
- context based
    1. Skip Gram
    - This is predictive model, find the probabiluity of neighbours words given the target word, which is middle word in sliding window
    Each context-target pair is treated as a new observation in the data. For example, the target word “swing” in the above case produces four training samples: (“swing”, “sentence”), (“swing”, “should”), (“swing”, “the”), and (“swing”, “sword”).
    2. Cont Bag of Words
    The Continuous Bag-of-Words (CBOW) is another similar model for learning word vectors. It predicts the target word (i.e. “swing”) from source context words (i.e., “sentence should the sword”).
    In The CBOW model. Word vectors of multiple context words are averaged to get a fixed-length vector as in the hidden layer. Other symbols have the same meanings as in Fig 1.

## Loss function to train word embedding:
1. Full Softmax
2. Hierarchichal Softmax (didn't understood properly). The idea here is to create a tree, where each node represent relative probability  of children nodes and leaf nodes are words. So to reach at one word, we follow a unique path, so on...
3. Cross Entropy
4. Noise Contrastive Analysis
    - The idea here is to differentiate target word from noise sample usinf a logistic regression classifier.
5. Negative Sampling
    - The idea is very simple, just replace the probabilty with sigmoid, now relation of noise contrastive analysis become very simple.


---

## Hyperparameter Tuning:
1. Random Search
2. Grid Search
3. Bayesian opt

## Bayesian Optimization (hyperparameter tuning):
    
    Bayesian optimization works by constructing a posterior distribution of functions (gaussian process) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not

    As you iterate over and over, the algorithm balances its needs of exploration and exploitation taking into account what it knows about the target function. At each step a Gaussian Process is fitted to the known samples (points previously explored), and the posterior distribution, combined with a exploration strategy (such as UCB (Upper Confidence Bound), or EI (Expected Improvement)), are used to determine the next point that should be explored.


---

## Null Hypothesis:
- Accepted Fact
- which can be nullify/Invalidate
- For example: Earth is oval shaped, this is **null Nypothesis**, and earth is flat, is **alternative hypothesis**.

## Type-I and type-II error:
    Truth about the population
Decision based on sample    H0 is true  H0 is false
Fail to reject H0   Correct Decision (probability = 1 - α)  Type II Error - fail to reject H0 when it is false (probability = β)
Reject H0   Type I Error - rejecting H0 when it is true (probability = α)   Correct Decision (probability = 1 - β)
Example of type I and type II error

To understand the interrelationship between type I and type II error, and to determine which error has more severe consequences for your situation, consider the following example.
A medical researcher wants to compare the effectiveness of two medications. The null and alternative hypotheses are:

    Null hypothesis (H0): μ1= μ2

    The two medications are equally effective.

    Alternative hypothesis (H1): μ1≠ μ2

    The two medications are not equally effective.

A type I error occurs if the researcher rejects the null hypothesis and concludes that the two medications are different when, in fact, they are not. If the medications have the same effectiveness, the researcher may not consider this error too severe because the patients still benefit from the same level of effectiveness regardless of which medicine they take. However, if a type II error occurs, the researcher fails to reject the null hypothesis when it should be rejected. That is, the researcher concludes that the medications are the same when, in fact, they are different. This error is potentially life-threatening if the less-effective medication is sold to the public instead of the more effective one.

## Precision & recall:
- Precision: Of the positive predicted output, what percentage is actually positive.
- Recall: Of the positive actual output, what percentage our model predict it positive 

                | Predicted +ve | predicted -ve
--------------------------------------------------
Actual Postive  |     TP        |  FN (Type-II)
Actual Negative |  FP(Type-I)   |      TN

precision | TP/(TP+FP) 
Recall    | TP/(TP+FN)
F1        | 2PR/(P+R)

- False Negative has more severe effect than False Positive
- True positive rate: Fraction of actual positive predicted as positives: TP/(TP+FN) **Ist Row**
- False positive Rate: Fraction of actual negatives wrongly predicted as positive FP/(TN+FP)**2nd Row**
- ROC: Plot of TPR vs FPR for all possible value of threshold, also called area under operaring point
- AUC of 0.5, means close to random.

## regression Metrics: Page 15 lec23 CS771

---





## Imp:
1. Hadamard : element wise multiplication
2. Image Processing Algo: https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-3-greyscale-conversion/