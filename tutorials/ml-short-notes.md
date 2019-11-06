

# perceptron
- Online Algo (check one example at a time)
- Error driven (update weights only if prediction is wrong)
- **Linear Boundary Seprator**
- Descision rule to update weights `y_hat*y <= 0` as `y C [-1,1]` and weight update rule is `w += y*x` and `b += y`
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


## Regression Analysis:
regression analysis estimates the relationship between two or more variables.
- It indicates the significant relationships between dependent variable and independent variable.
- It indicates the strength of impact of multiple independent variables on a dependent variable.


## types of Regressions?
1. Linear Regression
2. Logistic Regression
3. Polynomial Regression
4. Stepwise Regression
5. Ridge Regression
6. Lasso Regression
7. ElasticNet Regression


# logistic regeression
- Linear regression try to put a linear line on samples, **Just imagine**, Can a linear line give a solution which seprate **[0,1]**. **Well, it can, Look at following exp**, because we need a threshold to decide which side the samples lie. But for biased/imbalanced dataset, the linear line will be biased along one side and solution may not be good, whereas logistic regression helps to put a sigmoid like curve on the samples, which seems good.

Fit a line though the following sample and analyze the threshold of 0.5 to detect the test samples.

```
||=========================||=========================|| 
||1         ******         ||         ******      *   || 
||                         ||                         || 
||0 ******                 ||    *****                || 
||=========================||=========================||
||   Linear fit well       ||     Linear line biased  || 
||      GOOD               ||          BAD            ||
||=========================||=========================||
```
# Log-reg Conti.
- log-reg, output is always between **0-1**, where in linear reg, output can be anything **>1 & <0**.
- log-reg computes probability of being 1.
- optimization problem.

# Ridge Regression:
- L2 regularization
- same solution as least square with `lambda I in inverse`, it also make inverse possible, if matrix is low rank.
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


## Model Flexibility vs model interpretability:
- Flexibility increase from left to right
- Interpretation decrease from left to right
  `subset-selection/lasso, linear regression, Decision tree/general additive model, SVM, Bagging, boosting`

> More flexible means more complex in stat, because they are more prone to overfitting

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
4. If the value obtained at step `2` is not whole number:
      Pick the value at that index, given by step `3`
    Else
      Ans will be the average of value at (index, index+1)
```
or example, suppose you have 25 test scores, and in order from lowest to highest they look like this: 43, 54, 56, 61, 62, 66, 68, 69, 69, 70, 71, 72, 77, 78, 79, 85, 87, 88, 89, 93, 95, 96, 98, 99, 99. To find the 90th percentile for these (ordered) scores, start by multiplying 90% times the total number of scores, which gives 90% ∗ 25 = 0.90 ∗ 25 = 22.5 (the index). Rounding up to the nearest whole number, you get 23
```

## Find Inverse of matrix:
- use GAUSS ELIMINATION METHOD
- Procede like following: **a11, a21, a31, -- a22, a32, a33, -- a23, a13, a12**
- In brief, proceed through each columns, but lower left triangle, and then move upward and take left turn, it reach to a12.
```c++
1. Init A X = B as [A | B]
2. Find pivot and rearrange row such that, diagnol represent the big number from the following rows
  For exp: 
  [1,  2, 4]    [49, 2, 2]
  [2, 10, 1] => [2, 10, 1]
  [49, 2, 2]    [1,  2, 4]
3. start with first pivot (49) and compute factor f as (2/49) and subtract the entire row from f*pivot that will be (2/49)*(49) as A[i][j] = A[i][j] - f*pivot
4. repeat for each pivot and iterate downward for each row
5. In the end, we will get matrix in row echlon form, which give as coeffiencient of X.
  [1, x12, x13, x14]
  [0,  1 , x23, x24]
  [0,  0 ,  1 , x34]

Note: there is a catch, if matrix A is [m X n] dimension:
  1. m == n , then we have unique solution (x33 = B3)
  2. m > n , no solution
  3. m < n , many solution (choose any value for x34, and then x33 = B3 - x34)
```

#### Solving linear equation using gaussian elimination method
```c++
//Gauss Elimination
#include<iostream>
#include<iomanip>
using namespace std;
int main()
{
    int n,i,j,k;
    cout.precision(4);        //set precision
    cout.setf(ios::fixed);
    cout<<"\nEnter the no. of equations\n";        
    cin>>n;                //input the no. of equations
    float a[n][n+1],x[n];        //declare an array to store the elements of augmented-matrix    
    cout<<"\nEnter the elements of the augmented-matrix row-wise:\n";
    for (i=0;i<n;i++)
        for (j=0;j<=n;j++)    
            cin>>a[i][j];    //input the elements of array
    for (i=0;i<n;i++)                    //Pivotisation
        for (k=i+1;k<n;k++)
            if (abs(a[i][i])<abs(a[k][i]))
                for (j=0;j<=n;j++)
                {
                    double temp=a[i][j];
                    a[i][j]=a[k][j];
                    a[k][j]=temp;
                }
    cout<<"\nThe matrix after Pivotisation is:\n";
    for (i=0;i<n;i++)            //print the new matrix
    {
        for (j=0;j<=n;j++)
            cout<<a[i][j]<<setw(16);
        cout<<"\n";
    }    
    for (i=0;i<n-1;i++)            //loop to perform the gauss elimination
        for (k=i+1;k<n;k++)
            {
                double t=a[k][i]/a[i][i];
                for (j=0;j<=n;j++)
                    a[k][j]=a[k][j]-t*a[i][j];    //make the elements below the pivot elements equal to zero or elimnate the variables
            }
    
    cout<<"\n\nThe matrix after gauss-elimination is as follows:\n";
    for (i=0;i<n;i++)            //print the new matrix
    {
        for (j=0;j<=n;j++)
            cout<<a[i][j]<<setw(16);
        cout<<"\n";
    }
    for (i=n-1;i>=0;i--)                //back-substitution
    {                        //x is an array whose values correspond to the values of x,y,z..
        x[i]=a[i][n];                //make the variable to be calculated equal to the rhs of the last equation
        for (j=i+1;j<n;j++)
            if (j!=i)            //then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
                x[i]=x[i]-a[i][j]*x[j];
        x[i]=x[i]/a[i][i];            //now finally divide the rhs by the coefficient of the variable to be calculated
    }
    cout<<"\nThe values of the variables are as follows:\n";
    for (i=0;i<n;i++)
        cout<<x[i]<<endl;            // Print the values of x, y,z,....    
    return 0;
}
```
## Optimization (Part-1):
- A function is convex, when `f(y) >= f(x)+dy/dx (y-x)`, which is nothing but the value at function is always greater than its tangent.
- Convex function has its second derivative(hessian) as semi-definite.
```
At critical point, f'(x) = 0;
using taylor series: f(x+h) = f(x) + h*f'(x) + 1/2 h*f''(x)*h
f(x+h) = f(x) + 1/2 h*f''(x)*h, as f'(x) = 0
f(x+h) - f(x) = 1/2 h*f''(x)*h = hessian
For local minima to exist, h should be >=0,
So hessian >=0, which is positive semidefinte
```

## gradient Desent:
- can optimize any function, `convex/non-convex`
- the idea is to take gradient and move in the opposite direction of it. **As grad tells us the direction in which slope is increasing**.
- we also use `learning rate` to give less signifies to update using current gradient.

## learning rate:
- constant
- momentum based `1/t`, `1/sqrt(t)`, `1/(1-t)`, RMSProp< Nestrov Momentum**
- `adaptive lr` **Adam, AdaGrad,...**
- `cyclic learning rate` very effective, the idea is insetead of finding one local minima, it find many many extrema and build the ensemble using all those optima pts.

## Gradient Based Optimization:
- `GD`: update as per the gradient of all data
    - GD converge to optimal at the rate of **(1/k)**, which means if you need an accuracy of **1e-4**, then it need something on the order of one thousand steps.
- `SGD`: Take one random sample and upadte acc to that, it will have **high variance**
- `Batch GD`: **Reduce Variance**
- `Subgradient`: At non-differential point, we get a range by looking at grad value on the right and left, then pick a value and pretend it like a differential function
- `Constrained opt` (Lagrangian Based && Projected Grad)
- Coordinate Descent Algo: Where we update one dimension feature from D-dim space
- `Alternative Optimization`: **Exp: EM**
- `Newton Method`(Second Order Method, another is `L-BFGS`):
    1. It tells about its curvature, shape, etc
    2. Each step is finding the minima of **quadratic function** in local space.
    3. No need of learning rate
    4. As `f(y) = f(x) + (y-x) df/dx + (y-x)^2 d^2f/dx^2`
    `f(wt+1) =f(wt)+df/dwt (w-wt) + 1/2 (w-wt)^2 d^2f/dw^2`
    `w = argmax_w f(wt+1)`
    `wt+1 = wt + (hessian)^-1 grad`
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

1. Momentum:
**Another way to think about momentum, suppose our update as oscilating while moving to its local minima, so when we taking average of many points value, its oscillation in verticle direction suppress, but in horizontal direction as all grad effect sums up, so it will have nice big grad value.**
    V_w = beta V_w + (1-beta) dw
    V_b = beta V_b + (1-beta) db

    w = w - alpha V_w
    b = b - alpha V_b

The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation.
2. `AdaGrad`: weakness: lr_rate decaying
Notice that the weights that receive high gradients will have their effective learning rate reduced, while weights that receive small or infrequent updates will have their effective learning rate increased. This agressive behaviour, stops Deep-NN to learn very early.
3. RMSProp:
The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate. In particular, it uses a moving average of squared gradients instead. Hence, RMSProp still modulates the learning rate of each weight based on the magnitudes of its gradients, which has a beneficial equalizing effect, but unlike Adagrad the updates do not get monotonically smaller
4. AdaDelta: Removed weakness of adadelta
Best, if we are using sparse data such as tf-idf features for words.

> If our data is sparse and our features have very different frequencies, we might not want to update all of them to the same extent, but perform a larger update for rarely occurring features.


## lagrangian Method:
- Primal & Dual method: Both gives same answer if the constrained function is convex, i.e. **g(w) <= 0**
- **`w = argmin_w{f(w) + argmax_a{a . g(w)}}`**
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
- `Perceptron`: `max(0, -ywx)`
- `Hinge Loss`: `max(0, 1-ywx)`
- `logistic`: `log(1+exp(-ywx))/log2`
- `Exponential Loss`: `exp(-ywx)`
- `Absolute loss`: `|y - wx|` **Robust to outliers**
- `epsilon-insensitive loss`: `|y-wx| + epsilon`

## Hyperplane:
- `W x + b = 0`, a hyperplane
    1. `b==0`, then plane is passing through origin
    2. `b>0`, hyperplane moves parallel in direction of w
    3. `b<0`, hyperplane moves opposite direction


## regularization:
- `shrinking of coefficient`
- avoid overfitting
- better generalization

### Types of regularization
1. `L_p Norm`:
    - p < 1, it shrink from diamond to shape, which stretches towards axes, it has sparsity nature, but also non-convex, so non preferable generally.
    - p = 1 ==> diamond, this is least value of p, with convexity properties, Also have sparse nature.
    - p = 2 ==> euclidean loss, helps in generalize better, by reducing weights so much that boundriess becomes smooth
    - **Why small weights are prefered, because by changing the input x by epsilon, its prediction should not change, which is possible with smalle weights only.**
    - p > 2, it move from circle to square as p = INF
    - p = INF, also called max norm.
2. `L0`:
    - point shape
    - NP hard
    - Count of non-negative feature value.
3. `L1`:
    - **lasso Regression**
    - Diamond shape(if the linear predictor or line of solution meet the corner point, it creats sparsity in solution)
    - sparse feaure
    - deal with multicolinearity problem
4. `L2`:
    - `higher lambda, more shrinkage(approx 0)`
    - `importance`: we don't want our model to be unstable with the addition of any noise. Note that, if there is noise in input feature, the coefficient for that feature will change the direction of hyperplane.
    - **Ridge** regression
    - circular shape
    - small weights **if there is any noise in feature, then having small weights generally doesn't effect too much**
5. `Dropout` Reg (in deep learning)
- for unsupervised learning:
    - `sum_i w_i*(x_i - x_j)^2`
- As p decrease in **L_p**, the shape of l_p shrink from edge(imagine dianmond and shrink its edges.) and tends to have sharp corners.

> Note: Least square method is scale invariant as y = beta0 + x1*beta1 + x2*beta2. So LS method can learn appropriate coeff betas with the scaled features, but with regularization, it can creates some problem, it `increase the panality for those features`. Better method to standarize the data before using it in ridge regression.

## Impotant Points
- Non-Linear Boundary using linear predictors, for circular dataset, using linear predictor, such as **f(w0 + w1*x1 + w2*x2)** will not work, But if we use **f(w0 + w1*x1^2 + w2*x2^2)**, it can find a circular boundary. **w0** helps in finding the threshold, the radius of circle.


## Time complexity of matrix multiplication:
- one is [m X n] and other is [n X p], the time complexity will be **[mnp]**

## Linear Regression:
- closed form solution is `[w = (X' X + lambda I_d)^(-1) X' Y]`, the time analysis is as 
    1. `X'X : DND`
    2. `X' X + lambda I_d : D D`
    3. Inverse of D X D matrix is D^3
    4. then [DXD][DXN][NX1] will be DDN
    5. [DXN][NX1] will be DN 
    - Overall O(DDD + DDN) == O(DDN)
- With GD, it will be O(KND), where k is number of iteration.


## Inductive Bias: 
- Inductive bias is the set of assumptions a learner uses to predict results given inputs it has not yet encountered.

---

## SVM (Maximum margin hyperplane): 
- `ywx > gamma`, it add a pre-specified margin.
- For gamma=0, it becomes perceptron
- Reason behind the name "Support Vector Machine"?
    1. SVM optimization discovers the most important examples (called `support vectors`) in training data
    2. These examples act as `balancing` the margin boundaries (hence called `support`)
- margin(gamma) is distance as `(wx+b) / ||w||`
- to maximize the margin, we minimize the `||w||`

## soft-margin SVM:
- `Very small C`: Large margin but also large training error.
- `Very large C`: Small training error but also small margin.
- C controls the trade-off between large margin and small training error

- Dual formaulation, where we have `argmax_a (a.1 - a G a)`
- The dual formulation is nice due to two primary reasons:
  1. Allows conveniently handling the margin based constraint (via Lagrangians)
  2. Important: Allows learning nonlinear separators by replacing inner products (e.g., `Gmn = ym yn xm xn`) by kernelized similarities (kernelized SVMs)

- What if we have solution, w and b, but not slacks, can we find it, **YES**, slacks's value is nothing but the hinge loss on the corresponding example.
  - slack = `0, when  yn (w xn + b) >= 1`
    and `(1 - yn (w xn + b))`   otherwise

- Soft Margin loss, is same as `regularized hinge loss`. `min_{w,b} = ||w||^2 + C sum_n hingle_loss(yn, (w xn + b))`
  1. first term --> large margin
  2. 2nd term   --> small slack

> One Biggest Advantage of Soft Margin SVM over hard margin, is that there is always going to be some solution, whether it is linearly seperable or not. **For example, lets suppose data is not linearly seperable, so hard margin will give up, but in soft margin, it use slacks to incorporate some mistakes/error, and will come up with a solution anyway.**


## Multiclass SVM:
- It use K weight vector, one for each class
- maximum margin problem will be same except the fact that now, loss will include all weight vector
- want score w.r.t correct class to be at least 1 more than score w.r.t all other classes
- `W*y X >= 1 + w*yhat X` yhat --> incorrect class and y --> correct class
- Same as Multiclass Hinge loss as `max{0, 1+max{k=yhat}(W_k X - W_y X)}`

1. One vs All:
    - For k classes, we make k binary classifier
2. All pairs:
    - Choose pairs of 2 from k, so in total we will have `K(k-1)/2` classifiers, Not very practicals for large number of classes.


## One Class Clf (Outlier/Novelty detection): 
1. Support Vector Data Description:
    - Assume positives lie within a ball with smallest possible radius (and allow slacks)
2. One-Class SVM:
    - Find a max-margin hyperplane separating positives from origin (representing negatives)

## Support vector Regression:
- `epsilon insentive loss`
- nonlinear regression can be made my kernel trick.

---

## Nonlinearity handling (kernel trick):
1. Separte 1-D data of 2 classes, 
    - Exp: `.......****.......`
    - Project data into 2 dimension of shape parabola or upper triangle as 
    1. `x --> [x, x^2]`
    2. `x --> [x, |x|]`
2. Seperate circular data
    - `[x1, x2] --> [x1^2, x1*x2, x2^2]`
3. Possible Mapping:
    - `x --> x^2, x^3, x^n`
    - `cos(x)`, `log(n)`
    - any function


## kernel:
- `Implicit mapping` for data, which means we operate an function on input, which automatically compute operation on higher dimesional space. 
- Exp: `K(X,Y) = (X*Y)^2`, where X and Y are 2 dim feature. With this operation, it implicity have `dot([x1^2, x1*x2, x2^2], [y1^2, y1*y2, y2^2])`, ie `phi(X).phi(Y)`

> Note: We didn't need to define the phi operation explicitly.

### Imp Points:
1. Any kernel function, which map `X-->F`, with mapping as **phi**, should satisfy **mercer condition**

> F need to be a vector space, with dot product operaion defined on it, F space is also called hilbert space.

#### Mercer’s Condition (Kernel Functions)
For k to be a kernel function
1. k must define a dot product for some Hilbert Space F
2. Above is true if k is symmetric and positive semi-definite (p.s.d.) function (though there are
exceptions; there are also "indefinite" kernels).
    - The function k is p.s.d. if the following holds
    `integ integ f (x)k(x; z)f (z)dxdz ≥ 0`

## properties of kernel function:
Let k1, k2 be two kernel functions then the following are as well:
1. `k(x; z) = k1(x; z) + k2(x; z)`: direct sum
2. `k(x; z) = αk1(x; z)`: scalar product
3. `k(x; z) = k1(x; z)k2(x; z)`: direct product
Kernels can also be constructed by composing these rules

## Types of kernel:
1. Linear kernel: `X*Y` (identity mapping)
2. Quadratic kernel: `(X*Y)^2 or (1 + X*Y)^2`
3. Polynomial kernel: `(1 + X*Y)^d`
4. Radial Basis Function: (Gaussian kernel) : `exp(-gamma |X-Y|^2)`
    - infinite dim basis function (implicitly)
    - Also called `stationary kernel`, as distance between X and Y is constant

## Kernel Matrix:
- nXn matrix, which is `pairwise similarity between n samples`
- K is a `symmetric and positive definite matrix`
- For a P.D. matrix: `z'Kz > 0`; (also, all eigenvalues positive)
- Also known as the `Gram Matrix`


## Kernel tricks
1.  Any learning model in which, during training and test, inputs only appear as dot products (xi. xj) can be kernelized (i.e., non-linearlized), by replacing the `xi.xj` terms by `φ(xi).φ(xj) = k(xi, xj)`
2. Most learning algorithms can be easily kernelized
    - Distance based methods, Perceptron, SVM, linear regression, etc.
    - Many of the unsupervised learning algorithms too can be kernelized (e.g., K-means clustering, Principal Component Analysis, etc. - will see later)

## Speeding up kernel Method:
- slow at training and testing samples
- Like in ridge regression, we need to store the entire training sample to make a prediction **or** the basis vector of original data, which can be huge to store. **But there are method, which can helps us in computing low dim features**
- Landmarks and Random feature
- The idea here is that instead of computing high dim `phi(x)`, we can replaced it with low-dims `psi(x)`, with following property fulfilled i.e. `psi(x) psi(y) ~~ phi(x) phi(y)`
- computation become cheaper with `dual form`, if `D > N`

1. Landmarks:
    - Select L training data [z1, z2, ..., zL] as landmarks
    - `psi(xn) = [k(z1, xn), k(z2, xn), ..., k(zL, xn)]`
    - `k(xn, xm) = psi(xn) psi(xm)`
    - fast both on training and testing
2. Random feature:
    - `k(xn, xm) = E_{w~p(w)} [t_w(xn) t_w(xm)]`
    - use monte carlo to compute the kernel matrix.
    1. Sample w from distirbition
    2. For that w, compute `t_w`, which is a `L` dims vector
    3. For Exp: RBF kernel: `k(xn, xm) = E_{w~p(w)} [cos(w*xn) cos(w*xm)]`
3. Another method to speed up SVM kernel method
    - cluster the support vector (alpha_n)
    - Low rank approximation of kernel matrix

---

## Fisher information score
- used in feature selection
```matlab
mu = mean(feature);

n_1 = sum(label == 1);
mu_1 = mean(feature(label == 1));
var_1 = var(feature(label == 1));

n_2 = sum(label == 2);
mu_2 = mean(feature(label == 2));
var_2 = var(feature(label == 2));

inter_class = n_1*(mu_1-mu)^2 + n_2*(mu_2-mu)^2; 
intra_class = (n_1-1)*var_1 + (n_2-1)*var_2;

score = inter_class / intra_class;
```

---

## TF-IDF:
Computes the (query, document) similarity. It has two parts.
1. `TF Score (Term Frequency)`
    - Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:
    - `TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)`
2. `Inverse Document Frequency`
    - IDF which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:
    - `IDF(t) = log_e(Total number of documents / Number of documents with term t in it)`
- we usually consider log transformation of tf score
- `tfidf = (1+log(1 + freq_of_word_in_doc/total_word_in_doc)) * log10(total_docs/no_of_docs_in_which_word_appear)`


---
## STATISTICS:
## Ordinary Least square
- Assumption: `there is no error in observation`
- So we minimize the residual error **vertical distance between data sample and slope(line)**
- **It is scale invariant**

### Assumptions(OLS):
1. The regression model is linear in the coefficients and the error term
2. The error term has a population mean of zero
3. All independent variables are uncorrelated with the error term
4. Observations of the error term are uncorrelated with each other
5. The error term has a constant variance (`no heteroscedasticity`)
    - The assumption of `homoscedasticity` (meaning “same variance”) is central to linear regression models. 
    - `Heteroscedasticity` (the violation of homoscedasticity) is present when the size of the error term differs across values of an independent variable.
6. No independent variable is a perfect linear function of other explanatory variables
7. The error term is normally distributed (optional)



## Total Least square
- Assumption: `there is error in observation` **More Practical**
- So we minimize the residual error **diagnoal distance between data sample and slope(line)**, which consider the error in both varaible along x as well as in y direction.
- **It is not scale invariant**

## Imp points about regression (Scale Invariancy Property):
- We can translate features, any way we want, without changing the model. 
- With scaling you need to be a little more careful when using a regularized model – these models are `not scale invariant`. If the scales of predictors vary wildly, models like the Lasso will shrink out the scaled down predictors. To put all predictors on an equal footing, you should be rescaling the columns. 
- Typically this involves forcing the columns to have unit variance.

1. OLS is scale invariant. If you have a model `y = w0 + w1 x1 + w2 x2` and you replace `x1` with `x1'=x1/2` and re-estimate the model, you’ll get a new model `y = w0 + 2w1 x1' + w2 x2` which gives exactly the same preditions. The new `x1'` is half as big, so its coefficient is now twice as big.
2. `Ridge and L1-penalized regression (and hence elastic net) are not scale invariant`. 
    - Ridge shrinks the big weights more than the small ones
3. `L0 regression is scale invarian`t; the feature is in or out of the model, so the size doesn’t matter.
4. `PCA is not scale invariant`. People therefore often rescale the data (standardize it) before they do PCA. 


---

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

#### Table of error types:
- `H0` is Null Hypothesis

 _ | H0 is true | H0 is false
--- | --- | ---
Reject H0 |  Type-I error, (false positive) (prob = α) | Correct inference (true positive) (prob = 1 - β)
Fail to Reject H0 |  Correct inference (true negative) (prob = 1 - α) | Type II error (false negative) (prob = β)

1. The type I error rate or significance level is the probability of rejecting the null hypothesis given that it is true. 
  - Often, the significance level is set to 0.05.
  - It implies that it is acceptable to have a 5% probability of incorrectly rejecting the null hypothesis.
2. Type II error occurs when the null hypothesis is false, but erroneously fails to be rejected. 
  - A type II error is often called a false negative (where an actual hit was disregarded by the test and is seen as a miss) in a test checking for a single condition with a definitive result of true or false.

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
### R2 (R-squared): [Best](https://statisticsbyjim.com/regression/interpret-r-squared-regression/)
- It is an accuracy statistics in order to access a regression model
- It is the percentage of variance in Y explained by model
- Higher is R2, better the model is.
- **R2 = 1 - (RSS / TSS)**, where **RSS = sum (y - yhat)^2** and **TSS = sum (y - mean(y))^2**
- R squared is also known as:
    1. the fraction of variance explained.
    2. the sum of squares explained.
    3. Coefficient of determination
- R2 can be negative
- R2 = 0, means that model is just predicting the average of output


> Note: R2 cann't tells us about bias in the model, it is possible that higher R2 score is due to biased model. **So Never conclude that lower R2 model are not good, they can be good, Always look at residual plot to determine that**.

---

## Types of Regression model:
- Linear Regresion
- Logistic Regression
- Polynomial Regression
- Ridge
- Lasso
- Elastic-Net
- Robust Regression
- QuasiBionomail regression (Where target varaible's distribution assumed as skewed)
- least square model (ordinal Least square)
- Total Least square

## Difference between least square estimation and maximum likelihood estimation:
- LSE is to minimize the least square error
- MLE is to maximize the log likelihood as the loss function and minimze it with respect to the parameter.
- They are not equivalent untill, we assume gaussian distribution as probability density function in MLE.

## assumption in Oordinary Least Square:
1. The regression model is linear in the coefficients, as well in the error term
2. The error term has a population mean of zero
3. All independent variables are uncorrelated with the error term
4. Observations of the error term are uncorrelated with each other
    - One observation of the error term should not predict the next observation. For instance, if the error for one observation is positive and that systematically increases the probability that the following error is positive, that is a positive correlation. If the subsequent error is more likely to have the opposite sign, that is a negative correlation. This problem is known both as serial correlation and autocorrelation
5. The error term has a constant variance (no heteroscedasticity)
  - `Homoscedasticity` describes a situation in which the error term (that is, the “noise” or random disturbance in the relationship between the independent variables and the dependent variable) is the same across all values of the independent variables
7. The error term is normally distributed (optional)
    - OLS does not require that the error term follows a normal distribution to produce unbiased estimates with the minimum variance. However, satisfying this assumption allows you to perform statistical hypothesis testing and generate reliable confidence intervals and prediction intervals.


## Q. Do least square and linear regression same thing?
- Linear regression assumes a linear relationship between the independent and dependent variable. It doesn't tell you how the model is fitted. Least square fitting is simply one of the possibilities. Other methods for training a linear model is in the comment.

- Non-linear least squares is common (https://en.wikipedia.org/wiki/Non-linear_least_squares). For example, the popular **Levenberg–Marquardt algorithm** solves something like:

`β^=argminβS(β)≡argminβ∑i=1m[yi−f(xi,β)]2`

It is a least squares optimization but the model is not linear.
**They are not the same thing.**
- In summary: linear regression is optimization problem, with the intent to find best possible parameter for the linear line, where as least square method is potential loss function for an optimization problem. Which means, loss is (y - f(x))^2, where f(x) can be any function, linear/non-linear.

> f linear regression is  "low bias/high variance", there must be some alternative method that is biased but that has lower variance. Various forms of regularized regression exist, including lasso regression and ridge regression. These methods essentially shrink the estimated coefficient(s) towards zero, and they correspond to priors on the coefficient values. I believe that regularizing a regression using lasso or ridge will always decrease the variance in the estimator, and it will always introduce bias if the true value of the coefficient is not zero.

---

## Bias And Varaince:
Generally to measure the model, we draw a `chart of performance error vs model complexity`, this will leads us to the better decision.
  - with more complexity in model(higher order polynomial features), we will have `low-bias and higher-variance`
  - with less model complexity, we will have `high-bias and low-variance`.

The prediction error for any machine learning algorithm can be broken down into three parts:
1. Bias Error
2. Variance Error
3. Irreducible Error

### Mathematical expression: [Do derivation by urself, its confusing]
```python
y = f(x) + epsilon
err(x) = E[(y - yhat)^2]
err(x) = [(f(x) + epsilon - yhat)^2]
err(x) = [(f(x) + epsilon - yhat + E[yhat] - E[yhat])^2]
err(x) = (f(x) - E[yhat])^2 + E[f(x)^2] - E[yhat]^2 + epsilon^2
err(x) = Bias^2 + variance + Irreducible-error
```
The irreducible error cannot be reduced regardless of what algorithm is used. It is the error introduced from the chosen framing of the problem and may be caused by factors like unknown variables that influence the mapping of the input variables to the output variable

### Bias:
1. `Low Bias`: Suggests less assumptions about the form of the target function.
  - `Decision Trees, k-Nearest Neighbors and Support Vector Machines` are low bias ML Algo.
2. `High-Bias`: Suggests more assumptions about the form of the target function.
  - `Linear Regression, Linear Discriminant Analysis and Logistic Regression` are high bias ML Algo

### Variance:
Machine learning algorithms that have a high variance are strongly influenced by the specifics of the training data. This means that the specifics of the training have influences the number and types of parameters used to characterize the mapping function.
1. `Low Variance`: Suggests small changes to the estimate of the target function with changes to the training dataset.
  - `Linear Regression, Linear Discriminant Analysis and Logistic Regression`
2. `High Variance`: Suggests large changes to the estimate of the target function with changes to the training dataset.
  - `Decision Trees, k-Nearest Neighbors and Support Vector Machines`

> Generally, nonparametric machine learning algorithms that have a lot of flexibility have a high variance. For example, decision trees have a high variance, that is even higher if the trees are not pruned before use.

#### To reduce the variance further:
1. Ensemble of different models
2. (Not much imp)Ensemble of different parameters of same model (As, while solving an optimization, there can be many optima points)
3. Increase the dataset size
4. Increase diversity in features
5. Another
    - set random field
    - Early stopping
    - pruning of trees


## Cross-validation:
We took `5000` predictor to build a model for our 100 samples, we want to select best `100` predictors

### The right way:
- We build all such model, and select `top 100` based on cv error

### In wrong way:
- we use `correlation` for `each predictor with the target/label` and choose the `best 100 correlated predictor`.
- now `cv error` will be very low, as it has already seen the label. 
- test error will be very high in this case, because cv gets wrong due to step 1.


---

## Covariance:
- `cov(x,y) = sum_i (xi - E[x]) ( yi - E[y])`
- it is used to measure the direction of linearity(relationship) in x and y

## Correlation: [best](https://www.analyticsvidhya.com/blog/2015/06/correlation-common-questions/)
- `coerr = cov(x,y)/(var(x) var(y))`
- `cov(x,y) = sum_i (xi - E[x]) ( yi - E[y]) / sqrt(sum_i (xi - E[x])) sqrt(sum_i ( yi - E[y]))`
- it measure the strength as well as direction of colinearity.
- **Correlation** is normalized **Covariance**
- range is [-1, 1]
- `helps in feature selection(filter method)`
- pearson correlation has range of [-1, 1], **with string assumption of Linearity**
- Pearson correlation is **very sensitive to outliers**

> It is not of tranitive nature, which means if A and B are correlated && B and C are correlated, it doesn't tell about correlation of A and C.



### Spearmann coefficient
- `sigma(x, y) = 1 - (6* sum_i (d_i)^2) / (n*(n-1))`, where d is distance between corresponding rank between two variable
    1. Rank the data points of x and y, as highest value with rank of 1.
    2. Take manhattan distance between 2 variable, square them and add them all.
    3. Put in formaula above.

## Why colinearity is bad in linear model?
- If the model is consider all variable, to learn the characteristics, then those correlated feature will confuses it in making good decsion
- For linear regression, our `assumption of independent variable` get violated. 
  - Let's understand this problem specifically: Our objective is to model the dependent/response variable based on independent variable. This means that each independent variable has its own coeeficient, independent of other feature. But with multicolinearity, a minute change in one feature, changes other, but coefficent can't have this behaviour, because of our assumption
- example of multicolinearity: two feature are `x` and other is `x+10`
- Multicollinearity can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable.
- Tree Based Model(specifically `boosting tree`) are free from this problem, because it split a node based on only one feature at a time.
- `bagging methods` can have very small effect, but usually `unobserved`.

> corrleation: we talk between two variable, `multicolinearity` is used, when correlation occurs in multiple feature. For exp: `x`, `x*2 + 3` and `x/10`



##  dimensionality reduction algorithms:
- PCA (linear)
- t-SNE (stochastic neighbourhood embedding)(non-parametric/ nonlinear)
- Isomap (nonlinear)
- LLE (Local Likelihood Embedding) (nonlinear)
- SNE (nonlinear)
- Laplacian Eigenmaps (nonlinear)

### t-sne:
t-SNE is based on probability distributions with random walk on neighborhood graphs to find the structure within the data. 
- Local approaches seek to map nearby points on the manifold to nearby points in the low-dimensional representation. Global approaches on the other hand attempt to preserve geometry at all scales, i.e mapping nearby points to nearby points and far away points to far away points




## Generalized Clustering Setting:
- Objective: Learn model to cluster the unlabel data
- We need `cluster-id` and `cluster properties such as mean or variance`
- If we know `cluster-id`, then we can easily find `cluster properties` same as in `generative classification`
- First guess comes in find to solve it is `alternative optimization`
  - find cluster id (Z)
  - update parameter

### K-Mean Clustering Algorithm:
- Objective: reduce the `within-cluster variance`
- loss function: `sum_n sum_k z_{nk} ||x_n - z_k||^2`
- `(X - Z U)`, where `Z: n X k`, `U: k X d`, `X: n X d`
- assume cluster to be `equiprobable` or `convex-shaped`
- Also called `hard-clustering`
- Guaranteed to converge to local optimal (with proof)
- It can be `kernelized` :)
- If euclidean distance is replaced by absolute distance, it will be `k-Median` Algorithm
  - robust to outliers
- It just learn the mean of cluster, but it can be modified to `GMM` to capture variance
- Our `z` is one-hot vector, if we use `probability` vector, then it will be called probabilistic clustering or `soft-clustering`

```python
@kmean algo (hard-clustering)
1. Find optimal clustering id (labels)
    C_k = {n: k argmin_k ||x_j - mu_k||^2 }
2. Find Optimal centroid
    Mean_k = 1/(|C_K|) sum_{n belongs to C_k} x_n 
```

```python
@kmean algo (soft-clustering) used in fuzzy
1. Find optimal clustering id (labels)
    gamma_{nk} = softmax( ||x_j - mu_k||^2 )
2. Find Optimal centroid
    Mean_k = 1/(|gamma_{nk}|) sum ( gamma_{nk} x_n )
```

#### kernelized k-mean:
- `L(x, mean, c) = ||x - c||^2`, which become with kernelized trick as `||phi(x) - phi(c)||^2 = phi(x,x) + phi(c,c) + 2*phi(x, c)`

#### probalistic setting
- Objective: Cluster id that is `P(z/x,theta)`
- `P(z/x,theta) = constant p(z/theta) p(x/z,theta)`
- If we know z, then `p(x/z,theta)` is `generative classification` problem setting.
- `P(z/theta)` is multinomial distribution
- `P(z_n = k/x_n,theta) ~ pi_k N(x_n| mu_k, sigma_k)`

#### Mixture of experts:
- Learn several linear model for non-linear distribution.
```python
1. Init random cluster
2. Learn linear model on each cluster
3. update clusters properties (mean)
4. go back to 2., untill it converges
```


## probabilty Distribution:
- Discrete Distribution:
  1. Bernoulli
    - distribution over {0,1} e.g coin toss problem
    - `p^x (1-p)^(1-x)`
  2. Binomail
    - distribution over number of suceess m over n trial
    - `NCm p^m (1-p)^(N-m)`
  3. Multinoulli:
    - categorical distribution (multiclass classification)
    - `prod p_k^x_k`
  4. Multinomial:
    - repeat mutinolli N times
    - models the bin allocation via discrete vector x of size k
  5. poisson distribution
    - model a non-negative integer (count)
    - example : `number of words in a doc` or `number of events in fixed inteval of time`
    - `lambda^k exp(-lambda)/ k!`
- Continuous distribution
  1. Uniform
    - `1/(b-a)`
  2. Beta Distribution
    - model rv between [0,1]
    - Often used to model the probability parameter of a Bernoulli or Binomial (also conjugate to these distributions)
  3. Gamma Distribution
    - used to model positve real valued rv
    - Often used to model the rate parameter of Poisson or exponential distribution (conjugate to both), or to model the inverse variance (precision) of a Gaussian (conjuate to Gaussian if mean known)
  4. Dirichlet Distribution
    - Dirichlet is conjugate to Multinoulli/Multinomial
    - Note: Dirichlet can be seen as a generalization of the Beta distribution. Normalizing a bunch of Gamma r.v.’s gives an r.v. that is Dirichlet distributed.
- Gaussian `N(x|m1,s1)`
  - transformation of space `y = Ax+b` . It becomes `N(y|A*m1+b, A*s1*A')`
  - product of two gaussian will be `1/z N(x|m,s)`, where `m = (m1*s2 + m2*s1)/(s1+s2)` and `s = (s1*s2)/(s1+s2)`
    - It is unnormalized, where `z = N(m1|m2,s1+s2) = N(m2|m1,s1+s2)`
  - diag covariance : 
    - with equal variance across each dim is circular
    - with unequal will be elipse in horizinal or vertical direction
    - full cov: ellipical shape in any order(axis)



---
## Evaluation Metric:

### Confusion Metrics:
  | Predicted +ve | Predicted -ve
--- | --- | ---
Actual +ve | TP | FN
Actual -ve | FP | TN

#### Precision: 
- How good is model on predicting positive class, which is `TP/(TP + FP)`
- More Important, when we False positive id  


```
ax = articles['publication'].value_counts().sort_index().plot(kind='bar', fontsize=14, figsize=(12,10))
ax.set_title('Article Count\n', fontsize=20)
ax.set_xlabel('Publication', fontsize=18)
ax.set_ylabel('Count', fontsize=18);
```


---
## Decision Tree:
Our objective is to find the regions, which can uniquely predict the value of input feature. 
- In case of `regression`, objective function becomes: `sum_j{1:J} sum_{i belong to R_j} (y_i - yhat_{R_j}^2`
- This is most important to undertand. To compute error, we proceed as follows: For each region `j`, we collect true labels of each such sample which lie in that region and then take average of their prediction(value from that region j), and then compute `(y - yhat)^2`.
- The tree are built in Greedy fashion. At each node, we split on the basis of higher information gain of feature.
- Although each split of the tree makes two partitions, it can fit interactions, while the additive model cannot. So tree model are capable of fitting a richer class of functions:

### Pruning (weakest link pruning):
- use greedy approach, to prune tree from bottom to up approach
- In case of `regression`, objective function becomes: `sum_m{1:|T|} sum_{xi belong to Rm} (y_i - yhat_{Rm}^2 + alpha |T|`, where `|T|` is the number of `terminal nodes/leaf nodes`.

### Algorithm:
```python
1. Use recurive binary splitting to grow a larger tree on training data, stopping only when each terminal node has fewer than some minimum number of observation. For exp, we split till we have 5 observation in each region
2. apply cost complexity pruning to larger tree in order to obtain a sequence of best subtree as a function of alpha.
3. Use k-fold cross validation to choose alpha
4. return best subtree from step 2 that corresponds to the chosen value of alpha
```



- The `Information Gain` (IG) can be defined as follows:
  `IG(Dp) = I(Dparent) − Nleft/Np I(Dleft) − Nright/Np I(Dright)` where I could be `entropy`, `Gini index`, or `classification error`, Dparent, Dleft, and Dright are the dataset of the parent, left and right child node.
- consider an following example for understanding why classification error is not a good metrics to rule based method like decision tree.
         A                      B
      (40,40)                (40,40)
      /     \                /     \
  (30,10)  (10,30)       (20,40)  (20,0)

In A and B, we can clearly see that B is better, because of right child have homogenity, whereas in A, there is no such case.
- But as per the classification error method, both A and B have same error-rate of `0.25`. 
- Gini criteria: `0.17`
- Entropy      : `0.31`

- `Higher the information gain, lesser will be the entropy, better it will be`. It tell as that there is less uniformatity, which is what we desire. **A rule should involve more homogenity**

### Gini index:
- `G = sum_{k = 1:K} pmk * (1 - pmk)`
- `measure of total variance across K classes`
- also known for `purity measure`
- smaller is better. For exp: if pmk is 0 or 1, then G = 0
- alternative to cross entropy, `D = - (sum_{k = 1:K} pmk log(pmk))`
- You have a bag of marbles with 64 red marbles and 36 blue marbles. What is the value of the Gini Index for that bag?  Ans: `Gini Index = .64*(1-.64) + .36*(1-.36) = .4608`

#### Gini: 
- `1 - sum_{j: Classes} p_j^2`
- `Higher the value of Gini higher the homogeneity.`

#### Entropy: 
  `- sum_{j: Classes} p_j log(p_j)`
- **why gini is preferred over entropy?**
  1. First of all both are pretty much same (You can draw both metric on graph, entripy is parabolic, where as gini's curve follow same nature, but curve is little below of entropy)
  2. Gini has computational advantage. `No need of expensive logrithm`

---

## Bagging:
Bagging is very powerful method. It is generally the combination of many `decorrelated`(**very desirable**) models. 
- reduce the variance as `(sig1 + sig2 +... sign) / n`
- `out of bag` error, which is error computes on leave out observation. As in bootstrap sampling, it choose `68%` of observation to fit the model, the other `32%` will be used for validation, which is called as `out of bag error`.

## Random forest:
- it use a simple but powerful idea to reduce variance, by choose `m` predictors out of `p` feature. for exp: `m = sqrt(p)`
- use less number of predictors/features for each bag, it can reduce the variance, beacuse of no/less correlation between each bag.

### `limitations of Random forest are` :
  1. Correlated features will be given equal or similar importance, but overall reduced importance compared to the same tree built without correlated counterparts.
  2. Random Forests and decision trees, in general, give preference to features with high cardinality ( `Trees are biased to these type of variables` ).

> Note: In stat, we use predictor terms instead of features.

---

> An interseting method for feature selection is we can select only those predictors, which have high variance. Note that there is no cheating, because it doesn't use label to do that. We can just build a tree and find the variance of each feature, from the splitting decision.


## Boosting:
- build tree sequentially on residuals
1. start with equal weight `Dt(n) = 1/n` for each sample.
2. find error `et = sum_n Dt(n) |yt == yhatt|`
3. Compute importance `alphat = 1/2 log((1 - et)/et)`
4. update weight `D{t+1}(n) = Dt(n) exp(beta)`, where beta = 
    `-alpha for correct prediction` and 
    `alpha for incorrect`
5. Normalize `D{t+1}(n)`
6. Go to step 2. untill converge


## important stuff on bagging and boosting
- To perform Random Forests we need to select 2 parameters: number of samples B and m= number of variables sampled at each split.
- In order to perform Boosting, we need to select 3 parameters: number of samples B, tree depth d, and step size 
---

## Text Normalization:

#### Stemming:
- "Stemming is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language."
- stemming is just stemming of prefix and suffix such as `(-ed,-ize, -s,-de,mis)`, even the rest of the word doesn't have meaning
  1. Porter-stem: fast, set of 5 rule
  2. Lanchester-stemming: slow, set of 120 rule, iterative approach, check character by chracter
  3. SnowballStemmer: Non-english work stemmer

#### Lemmantization:
- Lemmatization, unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language. In Lemmatization root word is called Lemma. A lemma (plural lemmas or lemmata) is the canonical form, dictionary form, or citation form of a set of words.
  1. WordNet-lemmantization


```python
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word = "multiplying" 
lem.lemmatize(word, "v")
>> "multiply" 
stem.stem(word)
>> "multipli"
```

---

## LDA
- `supervised`
- linear transformation techniques
- LDA attempts to find a feature subspace that maximizes class separability
- Remember that LDA makes `assumptions` about normally distributed classes and equal class covariances.


#### What is the difference between LDA and PCA for dimensionality reduction?
1. Both LDA and PCA are linear transformation techniques: LDA is a supervised whereas PCA is unsupervised – PCA ignores class labels.
2. PCA helps to find the directions of maximal variance, while LDA attempts to find a feature subspace that maximizes class separability 

---

## Random Forest
- Random Forest is an ensemble model of decision trees(`bagging` which helps in `reduce variance`).
- time complexity: to building a complete unpruned decision tree is O( N * v * n log(n) ), where `n` is the number of observations, `v` is the number of variables/attributes and `N` is number of estimators.









---

## Hypothesis:
- A premise or claim that we want to test
- `Null-Hypothesis`: `H0` currently  accepted value for a parameter
- `Alternative Hypothesis`: `Ha` research hypothesis, The claim to be tested.
- These are oppositve of each other `mathematical`


## Chi-square test:
Problem

A public opinion poll surveyed a simple random sample of 1000 voters. Respondents were classified by gender (male or female) and by voting preference (Republican, Democrat, or Independent). Results are shown in the contingency table below.

  | Voting Preferences | Row total

    | Rep  | Dem  | Ind
---|---|---|---
Male | 200  | 150 |  50 | 400
Female | 250 |  300 |  50 | 600
Column total | 450  | 450 | 100 | 1000

Is there a gender gap? Do the men's voting preferences differ significantly from the women's preferences? Use a 0.05 level of significance.

Solution

The solution to this problem takes four steps: (1) state the hypotheses, (2) formulate an analysis plan, (3) analyze sample data, and (4) interpret results. We work through those steps below:

    State the hypotheses. The first step is to state the null hypothesis and an alternative hypothesis.

    Ho: Gender and voting preferences are independent.

    Ha: Gender and voting preferences are not independent.
    Formulate an analysis plan. For this analysis, the significance level is 0.05. Using sample data, we will conduct a chi-square test for independence.
    Analyze sample data. Applying the chi-square test for independence to sample data, we compute the degrees of freedom, the expected frequency counts, and the chi-square test statistic. Based on the chi-square statistic and the degrees of freedom, we determine the P-value.

    DF = (r - 1) * (c - 1) = (2 - 1) * (3 - 1) = 2

    Er,c = (nr * nc) / n
    E1,1 = (400 * 450) / 1000 = 180000/1000 = 180
    E1,2 = (400 * 450) / 1000 = 180000/1000 = 180
    E1,3 = (400 * 100) / 1000 = 40000/1000 = 40
    E2,1 = (600 * 450) / 1000 = 270000/1000 = 270
    E2,2 = (600 * 450) / 1000 = 270000/1000 = 270
    E2,3 = (600 * 100) / 1000 = 60000/1000 = 60

    Χ2 = Σ [ (Or,c - Er,c)2 / Er,c ]
    Χ2 = (200 - 180)2/180 + (150 - 180)2/180 + (50 - 40)2/40
        + (250 - 270)2/270 + (300 - 270)2/270 + (50 - 60)2/60
    Χ2 = 400/180 + 900/180 + 100/40 + 400/270 + 900/270 + 100/60
    Χ2 = 2.22 + 5.00 + 2.50 + 1.48 + 3.33 + 1.67 = 16.2

    where DF is the degrees of freedom, r is the number of levels of gender, c is the number of levels of the voting preference, nr is the number of observations from level r of gender, nc is the number of observations from level c of voting preference, n is the number of observations in the sample, Er,c is the expected frequency count when gender is level r and voting preference is level c, and Or,c is the observed frequency count when gender is level r voting preference is level c.
    The P-value is the probability that a chi-square statistic having 2 degrees of freedom is more extreme than 16.2.

    We use the Chi-Square Distribution Calculator to find P(Χ2 > 16.2) = 0.0003.
    Interpret results. Since the P-value (0.0003) is less than the significance level (0.05), we cannot accept the null hypothesis. Thus, we conclude that there is a relationship between gender and voting preference.

Note: If you use this approach on an exam, you may also want to mention why this approach is appropriate. Specifically, the approach is appropriate because the sampling method was simple random sampling, the variables under study were categorical, and the expected frequency count was at least 5 in each cell of the contingency table.


#### Interpret Results
- Reject the hypothesis, if `p-value` is less than significance level(eg. 0.05 that is 95% confidence interval)
- `A P-value measures the strength of evidence in support of a null hypothesis.`
- P-value is the probability for the null hypothesis to be True.

---

## Pandas Aggregation:
```python
# Group the data frame by month and item and extract a number of stats from each group
data.groupby(['month', 'item']).agg(aggregation)
aggregation = {
        # find the min, max, and sum of the duration column
        'duration': [min, max, sum],
         # find the number of network type entries
        'network_type': "count",
        # min, first, and number of unique dates per group
        'date': [min, 'first', 'nunique']
    }
```

- Another way to aggregate with new columns names:
```python
# Group the data frame by month and item and extract a number of stats from each group
data.groupby(['month', 'item']).agg(aggregation)
aggregation = {
        # find the min, max, and sum of the duration column
        'duration': {
            'total_duration'  : 'sum',
            'average_duration': 'mean',
            'num_calls'       : 'count',
        },
         # find the number of network type entries
        'network_type': {
            'count_networks' : 'count',
            'num_days'       : lambda x: max(x)-min(x),
        },
        # min, first, and number of unique dates per group
        'date': [min, 'first', 'nunique']
    }
```

```python
# Load the required packages
import time
import psutil
import numpy as np
import pandas as pd
import multiprocessing as mp

# Check the number of cores and memory usage
num_cores = mp.cpu_count()
print("This kernel has ",num_cores,"cores and you can find the information regarding the memory usage:",psutil.virtual_memory())


# Writing as a function
def process_user_log(chunk):
    grouped_object = chunk.groupby(chunk.index,sort = False) # not sorting results in a minor speedup
    func = {
      'date'   : ['min','max','count'],
      'num_25' : ['sum'],
      'num_50' : ['sum'], 
      'num_75' : ['sum'],
      'num_unq': ['sum'],
      'totSec' : ['sum']
    }
    answer = grouped_object.agg(func)
    return answer

# Number of rows for each chunk
size = 4e7 # 40 Millions
reader = pd.read_csv('user_logs.csv', chunksize = size, index_col = ['msno'])
start_time = time.time()

for i in range(10):
    user_log_chunk = next(reader)
    if(i==0):
        result = process_user_log(user_log_chunk)
        print("Number of rows ",result.shape[0])
        print("Loop ",i,"took %s seconds" % (time.time() - start_time))
    else:
        result = result.append(process_user_log(user_log_chunk))
        print("Number of rows ",result.shape[0])
        print("Loop ",i,"took %s seconds" % (time.time() - start_time))
    del(user_log_chunk)    

# Unique users vs Number of rows after the first computation    
print("size of result:", len(result))
check = result.index.unique()
print("unique user in result:", len(check))

result.columns = ['_'.join(col).strip() for col in result.columns.values]
```
---

## Imp:
1. Hadamard : element wise multiplication
2. Image Processing Algo: https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-3-greyscale-conversion/
3. [Data science prep by amazon](https://www.quora.com/What-is-the-best-site-for-preparing-data-science-interview)
4. [Various Question for all section](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-data-science-machine-learning-interview-guide/)
5. importance of positve definite matrix
  - In optimization problem, positve definiteness `guarantedd for existance for optima`
  - It is a class of symetric matrix, which has huge application in algebra. For example to find the inverse, it is O(n^3) and even impossible to compute for very high dimensional space because of memory constarint. `But using symetrical property, we can decoompose the matrix (using Cholesky decomposition) as A = L.L', where L is lower trainagular matrix`
6. [amazon interview question](https://medium.com/acing-ai/amazon-ai-interview-questions-acing-the-ai-interview-3ed4e671920f)