

# perceptron
- Online Algo (check one example at a time)
- Error driven (update weights only if prediction is wrong)
- **Linear Boundary Seprator** (`can be kernelized to obtain non-linear boundary`)
- Descision rule to update weights `y_hat * y <= 0` as `y belongs to [-1,1]`
    - weight update rule is `w = w + y * x`
    - `b = b + y`
- **Hyperplane will be perpendicular to weight vector**, because `W * X = 0`, when `bias = 0`.
- Direction of weights is towards `+ve` samples.
- `W * X` is nothing but the `projection` of sample `x` on the `W` vector. So it tells the distance of how far it is from origin of hyperplane, when bias is zero.
- For `X` to be `D` dims space, hyperplane will always be `D-1` dims. For example for `2D` data points, `decision boundary` is `linear`.
- **Interpretation**: Scale the data in range of `[0-1]`, now weights will represent the `sensitivity of classification prediction on the features`. `More weights of feature, more sensitive to output`.
- For `+ve` class, features with higher +ve weights are more responsible and for `-ve` class, feature with higher -ve weights are more sensitive. If features have some noise, then it will be more appropriate to remove such feature, if it is more sentive for boundary or prediction.
- Issue with perceptron that it consider `later points more than former points`.
- **Average Perceptron** And **Voting Perceptron**, **What????**
- Researcher worked on XOR problem for decades using perceptron, which leads to AI-Winter.


> In statistics, categorical varaible are known as `qualitative variable` and for numerical feature, `quantitative variable` term is used.


## Regression Analysis:
regression analysis estimates the relationship between two or more variables.
- It indicates the significant relationships between dependent variable and independent variable.
- It indicates the strength of impact of multiple independent variables on a dependent variable.


## types of Regressions
1. Linear Regression
2. Logistic Regression
3. Polynomial Regression
4. Stepwise Regression
5. Ridge Regression
6. Lasso Regression
7. ElasticNet Regression


# logistic regeression
- Linear regression try to put a linear line on samples, **Just imagine**, Can a linear line give a solution which seprate **[0,1]**. **Well, it can, Look at following exp**, because we need a threshold to decide which side the samples lie. But for biased/imbalanced dataset, the linear line will be biased along one side and solution may not be good, whereas logistic regression helps to put a sigmoid like curve on the samples, which seems good.
- In log-reg, output is always between `[0-1]`, where in linear reg, output can be anything `> 1 or < 0`.
- it computes probability of being 1.
- log-reg is an optimization problem.

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

### Advantage of ols (in term of assumptions)
1. It does not require a linear relationship between the dependent and independent variables. 
2. The error terms (residuals) do not need to be normally distributed.
3. homoscedasticity is not required.
4. the dependent variable in logistic regression is not measured on an interval or ratio scale.

### Assumption in log-reg
1. First, binary logistic regression requires the dependent variable to be binary and ordinal logistic regression requires the dependent variable to be ordinal.

2. Second, logistic regression requires the observations to be independent of each other.  In other words, the observations should not come from repeated measurements or matched data.

3. Third, logistic regression requires there to be little or no multicollinearity among the independent variables.  This means that the independent variables should not be too highly correlated with each other.

4. Fourth, logistic regression assumes linearity of independent variables and log odds.  although this analysis does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds.

5. Finally, logistic regression typically requires a large sample size.  A general guideline is that you need at minimum of 10 cases with the least frequent outcome for each independent variable in your model. For example, if you have 5 independent variables and the expected probability of your least frequent outcome is .10, then you would need a minimum sample size of 500 (10*5 / .10).





# Ridge Regression:
- L2 regularization
- same solution as least square with `lambda I in inverse`, it also make inverse possible, if matrix is low rank.
- early stopping (not a regularization, but it helps in same way)


# optimization
- Gradient decent:
    need to pick learning rate
- conjugate gradient:
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

## Generative model vs discriminative model:

### Generative classifiers
‌1. Naive Bayes
2. Bayesian networks
3. Markov random fields
‌4. Hidden Markov Models (HMM)

### Discriminative Classifiers
‌1. Logistic regression
2. Support Vector Machine
‌3. Traditional neural networks
‌4. Nearest neighbour
5. Conditional Random Fields (CRF)s
    

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
2. Multiply the percentile with total number of value. For 25 student and 99 percentile, it will be 25*0.99.
3. Round to nearest whole number. 1.5-->2, 2.7-->3, 1.2-->1, 5-->5
4.  If the value obtained at step `2` is not whole number:
        Pick the value at that index, given by step `3`
    Else
        Ans will be the average of value at (index, index+1)
```
For example, suppose you have 25 test scores, and in order from lowest to highest they look like this: 43, 54, 56, 61, 62, 66, 68, 69, 69, 70, 71, 72, 77, 78, 79, 85, 87, 88, 89, 93, 95, 96, 98, 99, 99. To find the 90th percentile for these (ordered) scores, start by multiplying 90% times the total number of scores, which gives 90% ∗ 25 = 0.90 ∗ 25 = 22.5 (the index). Rounding up to the nearest whole number, you get 23th term. Ans is 98
```



## Find Inverse of matrix:
- use GAUSS ELIMINATION METHOD
- Procede like following: 
1. `a11, a21, a31`
2. `a22, a32, a33`
3. `a23, a13, a12`
```c++
1. Init A X = B as [A | B]
2. Find pivot and rearrange row such that, diagnol represent the big number from the following rows
  For exp: 
  [1,  2, 4]    [49, 2, 2]
  [2, 10, 1] => [2, 10, 1]
  [49, 2, 2]    [1,  2, 4]
3. start with first pivot (49) and compute factor f as (2/49) and subtract the entire row from f * pivot that will be (2/49)*(49) as A[i][j] = A[i][j] - f*pivot
4. repeat for each pivot and iterate downward for each row
5. In the end, we will get matrix in row echlon form, which give as coefficient of X.
  [1, x12, x13, x14]
  [0,  1 , x23, x24]
  [0,  0 ,  1 , x34]

Note: there is a catch, if matrix A is [m X n] dimension:
  1. m == n , then we have unique solution (x33 = B3)
  2. m > n , no solution
  3. m < n , many solution (choose any value for x34, and then x33 = B3 - x34)
```


#### Solving linear equation using gaussian elimination method
Steps
• Eliminate x1 from second equaধon
Row2 Row2 - a21 / a11 * Row1
• Eliminate x1 from third equaধon
Row3 Row3 - a31 / a11 * Row1
• Repeat above procedure to eliminate x1 from n-th row

Back Subsধtuধon
• Solve for xn
xn =
b( nn-1)
a
(n-1)
nn
• Back subsধtute in the upper triangular system
1. Subsধtute xn in (n - 1)-th equaধon to solve for xn-1
2. Subsধtute xn-1 in (n - 2)-th equaধon to solve for xn-2
3. Repeat
• Floaধng point operaধons (flops)
Number of flops = 2n3
3 + O(n2)
| {z }
Forward Eliminaধon


```c
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

```c++
//Gauss Elimination
#include<iostream>
#include<iomanip>
using namespace std;
int main()
{
    int n,i,j,k;
    cout.precision(4);     //set precision
    cout.setf(ios::fixed);
    cout<<"\nEnter the no. of equations\n";        
    cin>>n;                //input the no. of equations
    float a[n][n+1],x[n];  //declare an array to store the elements of augmented-matrix    
    cout<<"\nEnter the elements of the augmented-matrix row-wise:\n";
    for (i=0;i<n;i++)
        for (j=0;j<=n;j++)    
            cin>>a[i][j];    //input the elements of array
    //Pivotisation
    for (i=0; i<n; i++){
        for (k=i+1; k<n; k++){
            if(abs(a[i][i]) >= abs(a[k][i])) continue;
            for(j=0; j<=n; j++){
                swap(a[i][j], a[k][j]);
            }
        }
    }
    cout<<"\nThe matrix after Pivotisation is:\n";
    for (i=0;i<n;i++){
        for (j=0;j<=n;j++)
            cout<<a[i][j]<<setw(16);
        cout<<"\n";
    }
    // loop to perform the gauss elimination
    for(i=0; i<n-1; i++){
        for(k=i+1; k<n; k++){
            double factor = a[k][i] / a[i][i];
            for(j=0; j<=n; j++){
                //make the elements below the pivot elements equal to zero or elimnate the variables
                a[k][j] = a[k][j] - factor * a[i][j];
            }
        }
    }
    cout<<"\n\nThe matrix after gauss-elimination is as follows:\n";
    for (i=0; i<n; i++){
        for (j=0; j<=n; j++)
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
    for(i=0; i<n; i++)
        cout << x[i] << endl;
    return 0;
}
```

## Difference between gauss-jordan and gauss-elimination method:
Both methods are used to find solutions for linear systems by pivoting and elimination like as Ax⃗ =b⃗ 

. Gauss method end the matrix as a superior-triangular matrix and you find the solutions of a linear system by applying a regressive substitution.

Gauss-Jordan do the same as an additional: turn the desired current matrix A

a identity matrix. Thus, Gauss-Jordan means Gauss method plus doing the operation sufficient to make the matrix triangular inferior as well, which ends in a identity matrix.

With Gauss-Jordan method you have the exactly solution that you want without the need of regressive/progressive substitution. The desired solution is directly encoded on the vectorb⃗ .

## Fastest method to solve for inverse of matrix (or find the solution for set of linear equation in general)
1. Fast method would be the Gauss-Jordan method. 
2. Next would be LU Decomposition method. 
3. And last would be using Gaussian elimination. 
4. (cramer rule)Using the co-factor method would be the least efficient method - just see how much time it takes to find determinant of a matrix.

## Optimization (Part-1):
- A function is convex, when `f(y) >= f(x) + dy/dx (y-x)`, which is nothing but the value at function is always greater than its tangent.
- In other words, pick any two points on curve, the line passing through two points, will always be above the curve, if curve follows convexity.
- Convex function has its second derivative(hessian) as semi-definite.
```
At critical point, f'(x) = 0;
using taylor series: f(x+h) = f(x) + h * f'(x) + 1/2 h* f"(x) * h
f(x+h) = f(x) + 1/2 h * f"(x) * h, as f'(x) = 0
f(x+h) - f(x) = 1/2 h * f"(x) * h
For local minima to exist, hessian(h) should be >=0, which is positive semidefinte
```

### Concave, Convex and Non-Convex function:
1.  A function is non-convex if the function is not a convex function.
2. A function `g` is concave if `−g`is a convex function.
3. A function is non-concave if the function is not a concave function.
4. A non-convex function need not be a concave function. For example, the function f(x)=x(x−1)(x+1)
defined on [−1,1].

> Notice that a function can be both convex and concave at the same time, a straight line is both convex and concave.


## gradient Desent:
- can optimize any function, `convex/non-convex`
- the idea is to take gradient and move in the opposite direction of it. `As grad tells us the direction in which slope is increasing`.
- we use `learning rate` to control the significance of update using current gradient.

## learning rate:
- constant
- momentum based `1/t`, `1/sqrt(t)`, `1/(1-t)`, RMSProp < Nestrov Momentum
- `adaptive lr` **Adam, AdaGrad,...**
- `cyclic learning rate` very effective, the idea is insetead of finding one local minima, it find many-many extrema and build the ensemble using all those optima pts.

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
        `f(wt+1) = f(wt) + df/dwt (w-wt) + 1/2 (w-wt)^2 d^2f/dw^2`
        `w = argmax_w f(wt+1)`
        `wt+1 = wt + (hessian)^-1 * grad`
    5. Expensive Because of hessian
    6. Very Fast, if f(w) is convex

> double derivative tell about curvature, which decide the learning rate. As if the surface is getting less steeper, then the learning step is decreased.

## Deep Learning Famously Optimizer: [Best](http://ruder.io/optimizing-gradient-descent/)
- SGD, RMSProp, AdaGrad, AdaDelta, Adam, Nadam
- For time series ==> prefereably RMSPRop
- Best to use ==> SGD + Nestrov or Adam
- Nestrov ==> First jump and then correct its step(or jump)
- sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
- Batch normalization additionally acts as a regularizer, reducing the need for Dropout.
- For sparse data (e.g. tf-idf) ==> AdaGrad, AdaDelta(improved over AdaGrad)
    Best, if we are using sparse data such as tf-idf features for words.

1. Momentum:
**Another way to think about momentum, suppose our update as oscilating while moving towards its local minima, so when we taking average of many points value, its oscillation in verticle direction suppress, but in horizontal direction as all grad effect sums up, so it will have nice big grad value.**
    - `V_w = beta V_w + (1-beta) dw`
    - `V_b = beta V_b + (1-beta) db`
    - `w = w - alpha V_w`
    - `b = b - alpha V_b`
The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation.

2. `AdaGrad`: weakness: lr_rate decaying
Notice that the weights that receive high gradients will have their effective learning rate reduced, while weights that receive small or infrequent updates will have their effective learning rate increased. This agressive behaviour, stops Deep-NN to learn very early.

3. `RMSProp`:
The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate. In particular, it uses a moving average of squared gradients instead. Hence, RMSProp still modulates the learning rate of each weight based on the magnitudes of its gradients, which has a beneficial equalizing effect, but unlike Adagrad the updates do not get monotonically smaller

4. `AdaDelta`: Removed weakness of adadelta
Best, if we are using sparse data such as tf-idf features for words.

> If our data is sparse and our features have very different frequencies, we might not want to update all of them to the same extent, but perform a larger update for rarely occurring features.


## lagrangian Method:
- Primal & Dual method: Both gives same answer if the constrained function is convex, i.e. `g(w) <= 0`
- `w = argmin_w {f(w) + argmax_a {a.g(w)}}`
- `For dual solution, a.g(w) = 0` (complimentary slackness/Karush-Kuhn-Tucker `(KKT) condition`)

## Projected Grad Method:
1. Each step of projected GD works as follows
2. Do the usual GD update: `z(t+1) = w(t) − ηt g(t)`
3. Check z(t+1) for the constraints
    If z(t+1) <= C, `w(t+1) = z(t+1)`
    If z(t+1) > C , project on the constraint set: `w(t+1) = ΠC[z(t+1)]`

- Example: let's suppose our z(t+1), lie outside the unit circle, but just consider it to be 1, if our constrained of **w belong to unit circle**.


## Loss function (`Convex Surrogate Losses Function`):
- Surrogate losses, because it define the upper bound on the real losses and minimize that. So minimze the surrogate losses function make sure pushing down real losses too.
- **0/1 loss**: 1[ywx <= 0], this is NP hard, no efficient solution. For example for wx = -0.0000001, with increase of 00000009 it will still be -1, but 000000011 increase will take it to right side that is +1.
- `Perceptron`: `max(0, -y w x)`
- `Hinge Loss`: `max(0, 1 - y w x)`
- `logistic`: `y log(p) + (1 - y) log(1 - p)`, where `p = 1 / (1 + exp(-y w x))` is prob of being `y = +1`
- `Exponential Loss`: `exp(-y w x)`
- `Absolute loss`: `|y - w x|` **Robust to outliers**
- `epsilon-insensitive loss`: `|y - w x| + epsilon`

## Hyperplane:
- `W x + b = 0`, a hyperplane
    1. `b = 0`, then plane is passing through origin
    2. `b > 0`, hyperplane moves parallel in direction of w
    3. `b < 0`, hyperplane moves opposite direction


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
2. (1.) is true if k is symmetric and positive semi-definite (p.s.d.) function (though there are exceptions; there are also "indefinite" kernels).
    - The function k is p.s.d. if the following holds
    `integ integ f(x) k(x; z) f(z) dx dz ≥ 0`

## properties of kernel function:
Let k1, k2 be two kernel functions then the following are as well:
1. `k(x; z) = k1(x; z) + k2(x; z)`: direct sum, if `k1` is valid kernel and `k2` is valid
2. `k(x; z) = αk1(x; z)`: scalar product
3. `k(x; z) = k1(x; z)k2(x; z)`: direct product
Kernels can also be constructed by composing these rules

> `For kernel to be valid, it should be positive semi definite.` for exp: In `1.2 k1(x,y) - 0.2 k2(x,y)` , `- 0.2 k2(x,y)` doesn't follows the property

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

## Recurrent Neural Network:
- weight sharing across the sequence
    - It create dependency across the data in sequence, but with different weights across sequence. This will cause in independent decision at the final prediction.
- Two main problem: 
    1. vanishing Grad (Most Imp) (cured by LSTM, but expensive)
    2. Exploding Grad (can handled by grad clipping)

## LSTM:
1. forget Gate (using sigmoid)
2. Input Gate (using sigmoid)
3. **ct_hat** Current cell state (using tanh)
4. Update Cell state (`ft * c{t-1} + it * ct_hat`)
5. Output gate (using sigmoid)
6. ht hidden state = `ot * tanh(ct)`

## LSTM variant:
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
### count based:
1. Unsupervised method
2. Use raw co-occurance count matrix
3. Run factor model/PCA/SVD etc to find the comprssed representation of word, with the assumption that word with similar context share same symantics (meaning)

### context based
1. Skip Gram
    - This is predictive model, which find the probability of neighbour words given the target word, which is middle word in sliding window
    - Each context-target pair is treated as a new observation in the data. 
    - For example, the target word “swing” in the `“sentence should the sword”` produces four training samples: (“swing”, “sentence”), (“swing”, “should”), (“swing”, “the”), and (“swing”, “sword”).
2. Continuous Bag of Words
    - The Continuous Bag-of-Words (CBOW) predicts the target word (i.e. “swing”) from source context words (i.e., `“sentence should the sword”`).
    - In The CBOW model, Word vectors of multiple context words are averaged to get a fixed-length vector as in the hidden layer.

## Loss function to train word embedding:
1. Full Softmax
2. Hierarchichal Softmax (didn't understood properly). The idea here is to create a tree, where each node represent relative probability  of children nodes and leaf nodes are words. So to reach at one word, we follow a unique path, so on...
3. Cross Entropy
4. Noise Contrastive Analysis
    - The idea here is to differentiate target word from noise sample using a logistic regression classifier.
5. Negative Sampling
    - The idea is very simple, just replace the probabilty with sigmoid, now relation of noise contrastive analysis become very simple.


---

## Hyperparameter Tuning:
1. Random Search
2. Grid Search
3. Bayesian opt

## Bayesian Optimization (hyperparameter tuning):
1. Bayesian optimization works by constructing a posterior distribution of functions (gaussian process) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not
2. As you iterate over and over, the algorithm balances its needs of exploration and exploitation taking into account what it knows about the target function. At each step a Gaussian Process is fitted to the known samples (points previously explored), and the posterior distribution, combined with a exploration strategy (such as UCB (Upper Confidence Bound), or EI (Expected Improvement)), are used to determine the next point that should be explored.


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

## Null Hypothesis:
- Accepted Fact
- which can be nullify/Invalidate
- For example: Earth is oval shaped, this is **null Nypothesis**, and earth is flat, is **alternative hypothesis**.

## Z-test:
`Std-dev = sigma/sqrt(N)`
`Z = (X - mu) / std-dev`
`Z critical value for α = 0.05 (one tailed) would be 1.65 as seen from the z table.`
- Therefore, if we get Z greater than the Z critical value, we can reject the null hypothesis.
- We know that confidence interval depends on the standard deviation of the data. If we introduce outliers into the data, the standard deviation increases, and hence the confidence interval also increases.
- The standard error of the mean is the standard deviation by the square root of the number of values.
- To check, if we have sufficient evidence to reject the null, we use `two tailed test`. It states that, the `z critical value for a 2 tailed test would be ±2.58`. If calculated z value is less than z_critical, we cann't reject null hypothesis.

## Type-I and type-II error:
A medical researcher wants to compare the effectiveness of two medications. The null and alternative hypotheses are:
1. Null hypothesis (H0): μ1= μ2
    The two medications are equally effective.
2. Alternative hypothesis (H1): μ1≠ μ2
    The two medications are not equally effective.

- A `type-I error` occurs if the researcher rejects the null hypothesis and concludes that the two medications are different when, in fact, they are not. If the medications have the same effectiveness, the researcher may not consider this error too severe because the patients still benefit from the same level of effectiveness regardless of which medicine they take. 
- However, if a `type-II error` occurs, the researcher fails to reject the null hypothesis when it should be rejected. That is, the researcher concludes that the medications are the same when, in fact, they are different. This error is potentially life-threatening if the less-effective medication is sold to the public instead of the more effective one.

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
- Another definition:
    `Precision`: What proportion of positive identifications was actually correct?
    `Recall`: What proportion of actual positives was identified correctly?

In statistical hypothesis testing a `type-I error` is the rejection of a true null hypothesis (also known as a "false positive" finding or conclusion), while a `type-II error` is the non-rejection of a false null hypothesis (also known as a "false negative" finding or conclusion)

                | Predicted +ve | predicted -ve
--------------------------------------------------
Actual Postive  |     TP        |  FN (Type-II)
Actual Negative |  FP(Type-I)   |      TN

precision | TP / (TP + FP) 
Recall    | TP / (TP + FN)
F1        | 2 P*R / (P + R)

- False Negative has more severe effect than False Positive
- True positive rate: Fraction of actual positive predicted as positives: TP/(TP+FN) **Ist Row**
- False positive Rate: Fraction of actual negatives wrongly predicted as positive FP/(TN+FP)**2nd Row**
- ROC: Plot of TPR vs FPR for all possible value of threshold, also called area under operaring point
- AUC of 0.5, means close to random.

## which is more important among type-1 and type-2
A Type I error, on the other hand, is an error in every sense of the word. A conclusion is drawn that the null hypothesis is false when, in fact, it is true. Therefore, Type I errors are generally considered more serious than Type II errors. The probability of a Type I error (α) is called the significance level and is set by the experimenter. There is a tradeoff between Type I and Type II errors. The more an experimenter protects himself or herself against Type I errors by choosing a low level, the greater the chance of a Type II error. Requiring very strong evidence to reject the null hypothesis makes it very unlikely that a true null hypothesis will be rejected. However, it increases the chance that a false null hypothesis will not be rejected, thus lowering power. The Type I error rate is almost always set at .05 or at .01, the latter being more conservative since it requires stronger evidence to reject the null hypothesis at the .01 level then at the .05 level. 

### Look at the Potential Consequences

Since there's not a clear rule of thumb about whether Type 1 or Type 2 errors are worse, our best option when using data to test a hypothesis is to look very carefully at the fallout that might follow both kinds of errors. Several experts suggest using a table like the one below to detail the consequences for a Type 1 and a Type 2 error in your particular analysis. 
Null    Type 1 Error: H0 true, but rejected     Type 2 Error: H0 false, but not rejected
Medicine A does not relieve Condition B.    Medicine A does not relieve Condition B, but is not eliminated as a treatment option.   Medicine A relieves Condition B, but is eliminated as a treatment option.
Consequences    Patients with Condition B who receive Medicine A get no relief. They may experience worsening condition and/or side effects, up to and including death. Litigation possible.    A viable treatment remains unavailable to patients with Condition B. Development costs are lost. Profit potential is eliminated.

Whatever your analysis involves, understanding the difference between Type 1 and Type 2 errors, and considering and mitigating their respective risks as appropriate, is always wise. For each type of error, make sure you've answered this question: "What's the worst that could happen?"


## imp stuff
- `Sensitivity`: In simple terms, the proportion of patients that were identified correctly to have the disease (i.e. True Positive) upon the total number of patients who actually have the disease is called as Sensitivity or Recall.
- `Specificity`: Similarly, the proportion of patients that were identified correctly to not have the disease (i.e. True Negative) upon the total number of patients who do not have the disease is called as Specificity.

### Trade-off between Sensitivity and Specificity
- **When we decrease the threshold, we get more positive values thus increasing the sensitivity. Meanwhile, this will decrease the specificity.**
- **Similarly, when we increase the threshold, we get more negative values thus increasing the specificity and decreasing sensitivity.**
- `As Sensitivity ⬇️ Specificity ⬆️`
- `As Specificity ⬇️ Sensitivity ⬆️`

### Trade off between Sensitivity & Specificity
- To plot ROC curve, instead of Specificity we use (1 — Specificity). So now, when the sensitivity increases, (1 — specificity) will also increase. This curve is known as the ROC curve.

## regression Metrics: Page 15 lec23 CS771
### R2 (R-squared): [Best](https://statisticsbyjim.com/regression/interpret-r-squared-regression/)
- It is an accuracy statistics in order to access a regression model
- It is the `percentage of variance in Y` explained by model
- `Higher is R2, better the model is`.
- R squared is also known as:
    1. the fraction of variance explained.
    2. the sum of squares explained.
    3. Coefficient of determination
- R2 can be negative
- R2 = 0, means that model is just predicting the average of output
- **R2 = 1 - (RSS / TSS)**, where **RSS = sum_i (yi - yhati)^2** and **TSS = sum_i (yi - mean(y))^2**
- ss_tot, `total sum of squares`: `ss_total = sum_i (yi - mean(y))^2` (proportion to variance of data)
- ss_reg, `regression sum of squares`: `ss_total = sum_i (yhati - mean(y))^2` 
- ss_res, `residual sum of squares`: `ss_total = sum_i (yi - yhati)^2`
- R^2 of `0.49`, tells us that `49%` of variablity of depenedent variable has been accounted for and the remaining `51%` of variability is still unaccounted for.
- `ss_tot = ss_reg + ss_res`
- `R^2 = ss_reg / ss_tot`

## F1 score:
- it is harmonic mean of `precision` and `recall`
- `F1 = (2 * precision * recall) / (precision + recall)`
- Generally, we consider weighted `F1` score, where we can prioritize recall or precision
- `F_beta = (1 + beta^2) . (precision * recall) / (beta^2 . precision + recall)`
- `F_beta = (1 + beta^2) . TP / ((1 + beta^2) TP + beta^2 . FN + FP)`

> Note: R2 cann't tells us about bias in the model, it is possible that higher R2 score is due to biased model. **So Never conclude that lower R2 model are not good, they can be good, Always look at residual plot to determine that**.

---

## Types of Regression model:
- Linear Regresion
- Logistic Regression
- Ridge
- Lasso
- Elastic-Net
- Robust Regression
- least square model (ordinal Least square)
- Polynomial Regression
- QuasiBionomail regression (Where target varaible's distribution assumed as skewed)
- Total Least square


## Total Least square
- Assumption: `there is error in observation` **More Practical**
- So we minimize the residual error **diagnoal distance between data sample and slope(line)**, which consider the error in both varaible along x as well as in y direction.
- **It is not scale invariant**


## Ordinary Least square
- Assumption: `there is no error in observation`
- minimize the residual error **vertical distance between data sample and slope(line)**
- **It is scale invariant**


## Assumption in Oordinary Least Square:
1. The regression model is linear in the coefficients, as well in the error term
2. The error term has a population mean of zero
3. All independent variables are uncorrelated with the error term
4. Observations of the error term are uncorrelated with each other
    - One observation of the error term should not predict the next observation. For instance, if the error for one observation is positive and that systematically increases the probability that the following error is positive, that is a positive correlation. If the subsequent error is more likely to have the opposite sign, that is a negative correlation. This problem is known both as serial correlation and autocorrelation
5. The error term has a constant variance (`no heteroscedasticity`)
    - `Homoscedasticity`(meaning “same variance”) describes a situation in which the error term (that is, the “noise” or random disturbance in the relationship between the independent variables and the dependent variable) is the same across all values of the independent variables
    - `Heteroscedasticity` (the violation of homoscedasticity) is present when the size of the error term differs across values of an independent variable.
6. The error term is normally distributed (optional)
    - OLS does not require that the error term follows a normal distribution to produce unbiased estimates with the minimum variance. However, satisfying this assumption allows you to perform statistical hypothesis testing and generate reliable confidence intervals and prediction intervals.

## Imp points about regression (Scale Invariancy Property):
- We can translate features, any way we want, without changing the model. 
- With scaling you need to be a little more careful when using a regularized model – these models are `not scale invariant`. If the scales of predictors vary wildly, models like the Lasso will shrink out the scaled down predictors. To put all predictors on an equal footing, you should be rescaling the columns. 
- Typically this involves forcing the columns to have unit variance.

1. OLS is scale invariant. If you have a model `y = w0 + w1 x1 + w2 x2` and you replace `x1` with `x1'=x1/2` and re-estimate the model, you’ll get a new model `y = w0 + 2w1 x1' + w2 x2` which gives exactly the same preditions. The new `x1'` is half as big, so its coefficient is now twice as big.
2. `Ridge and L1-penalized regression (and hence elastic net) are not scale invariant`. 
    - Ridge shrinks the big weights more than the small ones
3. `L0 regression is scale invariant`; the feature is in or out of the model, so the size doesn’t matter.
4. `PCA is not scale invariant`. People therefore often rescale the data (standardize it) before they do PCA. 


## Q. Difference between least square estimation and maximum likelihood estimation:
- LSE is to minimize the least square error
- MLE is to maximize the log likelihood as the loss function and minimze it with respect to the parameter.
- They are not equivalent untill, we assume gaussian distribution as probability density function in MLE.


## Q. Do least square and linear regression same thing?
1. **They are not the same thing.**
2. Linear regression assumes a linear relationship between the independent and dependent variable. It doesn't tell you how the model is fitted. Least square fitting is simply one of the possibilities.
3. In summary: linear regression is optimization problem, with the intent to find best possible parameter for the linear line, where as least square method is potential loss function for an optimization problem. Which means, loss is (y - f(x, w))^2, where f(x,w) can be any function, linear/non-linear, parameterized by `w`.
4. In least square: `w =  argmin_w [y − f(x, w)]^2`

> f linear regression is  "low bias/high variance", there must be some alternative method that is biased but that has lower variance. Various forms of regularized regression exist, including lasso regression and ridge regression. These methods essentially shrink the estimated coefficient(s) towards zero, and they correspond to priors on the coefficient values. Also note that regularizing a regression using lasso or ridge will always decrease the variance in the estimator, and it will always introduce bias if the true value of the coefficient is not zero.

---

## Bias And Varaince:
Generally to measure the model, we draw a `chart of performance error vs model complexity`, this will leads us to the better decision.
  1. with more complexity in model(higher order polynomial features), we will have `low-bias and higher-variance`
  2. with less model complexity, we will have `high-bias and low-variance`.

The prediction error for any machine learning algorithm can be broken down into three parts:
1. Bias Error
2. Variance Error
3. Irreducible Error
    - The `irreducible error` cannot be reduced regardless of what algorithm is used. 
    - This error is introduced by the problem formulation and may be caused by factors like unknown variables that influence the mapping of the input variables to the output variable

### Mathematical expression: [Do derivation by urself, its confusing]
```python
y = f(x) + epsilon
err(x) = E[(y - yhat)^2]
err(x) = E[(f(x) + epsilon - yhat)^2]
err(x) = E[(f(x) + epsilon - yhat + E[yhat] - E[yhat])^2]
err(x) = (f(x) - E[yhat])^2 + (E[f(x)^2] - E[yhat]^2) + epsilon^2
err(x) = Bias^2 + variance + Irreducible-error
```


### Bias:
1. `Low Bias`: Suggests less assumptions about the form of the target function.
  - `Decision Trees, k-Nearest Neighbors and Support Vector Machines` are low bias ML Algo.
2. `High-Bias`: Suggests more assumptions about the form of the target function.
  - `Linear Regression, Linear Discriminant Analysis and Logistic Regression` are high bias ML Algo

### Variance:
Machine learning algorithms that have a high variance are strongly influenced by the specifics of the training data. This means that the specifics of the training have influences the number and types of parameters used to characterize the mapping function.
1. `Low Variance`: Suggests `small changes` to the estimate of the target function with changes to the training dataset.
  - `Linear Regression, Logistic Regression and Linear Discriminant Analysis`
2. `High Variance`: Suggests `large changes` to the estimate of the target function with changes to the training dataset.
  - `Decision Trees, k-Nearest Neighbors and Support Vector Machines`

> Generally, nonparametric machine learning algorithms that have a lot of flexibility have a high variance. For example, decision trees have a high variance, that is even higher if the trees are not pruned before use.

### To reduce the variance further:
1. Ensemble of different models
2. (Not much imp) Ensemble of different parameters of same model (As, while solving an optimization, there can be many optima points)
3. Increase the dataset size
4. Increase diversity in features(imp)
5. Another
    - set different random seed
    - Early stopping
    - pruning of trees(imp)


## Q. Given `5000` predictor to build a model, how to select best `100` predictors (`Cross-validation`):
1. The right way:
    - We build all such model, and select `top 100` based on cv error
2. In wrong way:
    - we use `correlation` for `each predictor w.r.t target` and choose the `best 100 correlated predictor`.
    - now `cv error` will be very low, as it has already seen the target variable. 
    - test error will be very high in this case, because cv gets wrong due to step 1.


---

## Covariance:
- `cov(x,y) = sum_i (xi - E[x]) ( yi - E[y])`
- it is used to measure the direction of linearity(relationship) in x and y

## Correlation: [best](https://www.analyticsvidhya.com/blog/2015/06/correlation-common-questions/)
### pearson correlation:
- **Correlation** is normalized **Covariance**
- `coerr = cov(x,y)/(var(x) var(y))`
- `cov(x,y) = sum_i (xi - E[x]) ( yi - E[y]) / sqrt(sum_i (xi - E[x])) sqrt(sum_i ( yi - E[y]))`
- it measure the strength as well as direction of colinearity.
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

> If the nonlinear relationship had been monotonic rank correlation (Spearman's rho) would be appropriate.

## Why colinearity is bad in linear model?
- If the model is consider all variable, to learn the characteristics, then those correlated feature will confuses it in making good decsion
- For linear regression, our `assumption of independent variable` get violated. 
    - Let's understand this problem in depth: Our objective is to model the dependent/response variable based on independent variable. This means that each independent variable has its own coeeficient, independent of other feature. But with multicolinearity, a minute change in one feature, changes other feature, but coefficent can't have this behaviour, because of our assumption
- example of multicolinearity: two feature are `x` and other is `x+10`
- Multicollinearity can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable.
- Tree Based Model(specifically `boosting tree`) are free from this problem, because it split a node based only on one feature at a time.
- `bagging methods` can have very small effect, but usually `unobserved`.

> corrleation: we talk between two variable, `multicolinearity` is used, when correlation occurs in multiple feature. For exp: `x`, `x*2 + 3` and `x/10`

---

##  dimensionality reduction algorithms:
- PCA (linear)
- t-SNE (stochastic neighbourhood embedding)(non-parametric/ nonlinear)
- Isomap (nonlinear)
- LLE (Local Likelihood Embedding) (nonlinear)
- SNE (nonlinear)
- Laplacian Eigenmaps (nonlinear)

### t-sne:
- t-SNE is based on probability distributions with random walk on neighborhood graphs to find the structure within the data. 
- Local approaches seek to map nearby points on the manifold to nearby points in the low-dims representation. Global approaches on the other hand attempt to preserve geometry at all scales, i.e mapping nearby points to nearby points and far away points to far away point.


---


## Types of Clustering
1. Centroid-based Clustering
- distance based metrics
- efficient but sensitive to initial conditions and outliers.

2. Density-based Clustering
- Density-based clustering connects areas of high example density into clusters. 
- This allows for arbitrary-shaped distributions as long as dense areas can be connected. 
- These algorithms have difficulty with data of varying densities and high dimensions. 
- Further, by design, these algorithms do not assign outliers to clusters.

3. Distribution-based Clustering
- work good only, when we aware the distribution of dataset, such as Gaussian distributions.

4. Hierarchical Clustering
- Hierarchical clustering creates a tree of clusters. 
- Hierarchical clustering, not surprisingly, is well suited to hierarchical data, such as taxonomies. 
- In addition, another advantage is that any number of clusters can be chosen by cutting the tree at the right level.



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
- `X = Z U`, where `Z: n X k`, `U: k X d`, `X: n X d`
- assume cluster to be `equiprobable` or `convex-shaped`
- Also called `hard-clustering`
- Guaranteed to converge to local optimal (with proof)
- It can be `kernelized` (wow)
- If euclidean distance is replaced by absolute distance, it will be `k-Median` Algorithm
    - robust to outliers
- It just learn the mean of cluster, but it can be modified to `GMM` to capture variance
- Our `z` is one-hot vector, if we use `probability` vector, then it will be called probabilistic clustering or `soft-clustering`

> 1. Minimize the intra-cluster distances  2. Maximize the inter-cluster distances  3. However, K-Means is implicitly based on pairwise Euclidean distances b/w data points. 4. SStotal = SSwithin + SSbetween. So, if SSwithin is minimized then SSbetween is maximized.



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
- Objective: Cluster id is defined as `P(z/x,theta)`
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

---

## Probability Distribution:

### Discrete Distribution:
1. Bernoulli
    - distribution over {0,1} e.g coin toss problem
    - can applied only for binary event, `yes or no`, `success or failure`
    - `p^x (1-p)^(1-x)`
    - `mean: p`
    - `var: p (1 - p)`
2. Binomial
    - This distribution describes the `behavior of outputs of n random experiments`, each having a Bernoulli distribution with probability p.
    - distribution over `number of suceess m over n trial`
    - `NCm p^m (1-p)^(N-m)`
    - `mean : m * p`
    - `var: m * p * (1- p)`
3. Multinoulli:
    - similar to bernoulli except `multi-dimensional`
    - categorical distribution (multiclass classification)
    - `p1^x1 p2^x2 p3^x3 .... pk^xk`
4. Multinomial:
    - similar to binomial except `multi-dimensional`
    - repeat mutinolli N times
    - models the bin allocation via discrete vector x of size k
5. poisson distribution
    - model a non-negative integer (count)
    - example : `number of words in a doc` or `number of events in fixed inteval of time`
    - `lambda^k exp(-lambda)/ k!`
6. Geometrical distribution
    - It represents the number of failures before you get a success in a series of Bernoulli trials.
    - If X = n, it means you succeeded on the nth try and failed for n-1 tries.
    - `f(x) = (1 − p)^(x-1) p`
    - There are three assumptions of Geometric Distribution:
        1. There are two possible outcomes for each trial (success or failure).
        2. The trials are independent.
        3. The probability of success is the same for each trial.


### Continuous distribution
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
5. Gaussian `N(x|m1,s1)`
    - transformation of space `y = Ax+b` . It becomes `N(y|A*m1+b, A*s1*A')`
    - product of two gaussian will be `1/z N(x|m,s)`, where `m = (m1*s2 + m2*s1)/(s1+s2)` and `s = (s1*s2)/(s1+s2)`
        - It is unnormalized, where `z = N(m1|m2,s1+s2) = N(m2|m1,s1+s2)`
    - diag covariance : 
        - with equal variance across each dim is circular
        - with unequal will be elipse in horizinal or vertical direction
        - full cov: ellipical shape in any order(axis)


---

## Decision Tree:
Our objective is to find the regions, which can uniquely predict the value of input feature. 
- In case of `regression`, objective function becomes: `sum_j{1:J} sum_{i belong to R_j} (y_i - yhat_{R_j}^2`
- This is most important to undertand. To compute error, we proceed as follows: 
    1. For each region `j`, we collect true labels of each such sample which lie in that region.
    2. take average of their prediction(value from that region j)
    3. compute `(y - yhat)^2`.
- The tree are built in `Greedy fashion`. At each node, we split on the basis of higher information gain of feature.
- Although each split of the tree makes two partitions, it can fit interactions, while the additive model cannot. So tree model are capable of fitting a richer class of functions.


### Information Gain:
- The `Information Gain` (IG) can be defined as follows:
    `IG(Dp) = I(Dparent) − Nleft/Np I(Dleft) − Nright/Np I(Dright)` where I could be `entropy`, `Gini index`, or `classification error`, Dparent, Dleft, and Dright are the dataset of the parent, left and right child node.
- consider an following example for understanding why classification error is not a good metrics to rule based method like decision tree.
```
         A                      B
      (40,40)                (40,40)
      /     \                /     \
  (30,10)  (10,30)       (20,40)  (20,0)
```

In A and B, we can clearly see that B is better, because of right child have homogenity, whereas in A, there is no such case.
- But as per the classification error method, both A and B have same error-rate of `0.25`. 
- Gini criteria: `0.17`
- Entropy      : `0.31`

- `Higher the information gain, lesser will be the entropy, better it will be`. It tell as that there is less uniformatity, which is what we desire. **A rule should involve more homogenity**

### Gini index:
- `G = sum_{k = 1:K} pmk * (1 - pmk)`
- `measure of total variance across K classes`
- also known for `purity measure`
- `smaller is better`. For exp: if pmk is 0 or 1, then G = 0
- alternative to cross entropy, `D = - (sum_{k = 1:K} pmk log(pmk))`
- For exp: Let's say, we have a bag of marbles with 64 red marbles and 36 blue marbles. What is the value of the Gini Index of marbels in that bag?  Ans: `Gini Index = .64*(1-.64) + .36*(1-.36) = .4608`

#### Gini: 
- `1 - sum_{j: Classes} p_j^2`
- `Higher the value of Gini higher the homogeneity.`

#### Entropy: 
- `- sum_{j: Classes} p_j log(p_j)`
- **why gini is preferred over entropy?**
  1. First of all both are pretty much same (We can draw both metric on graph, we see that entropy is parabolic, where as gini's curve almost follows the same nature, but curve is little below of entropy in magnitude)
  2. Gini has computational advantage. `No need of expensive logrithm`

> Given a histogram of freq of occurance of X, 1. if it is uniform, then entropy is higher(highly uncertain and boring) 2. It hist has nice peak(one or two), it is more certain, so entropy is low(good)
> reference: http://www.cs.cmu.edu/~cga/ai-course/dtree.pdf

### Most important point of decision tree:
- With high cardinality, `information gain` perform much worse, they caused bias in solution
- to handle that, `gain-ration` is preferable, which include their `occurance` as well in the final split
- `intrinsic information = - (Nleft/Np) log(Nleft/Np) - (Nright/Np) log(Nright/Np)`
- `gain-ratio = Information-gain / intrinsic-information`
- For certain types of attributes with a large number of distinct values, splitting on that attribute would cause overfitting. Think about a decision tree choosing to split on the SSNs of your training set. Information gain will be huge, as IG is biased towards splitting on a large set of attributes. Splitting on SSNs will not be useful however as it leads to learning something specific about training set. It will not generalize.
- Gain ratio overcomes the problem with information gain by taking into account the number of branches that would result before making the split.
- `info-gain = H(class) - H(class/feature)`, and `intrinsic-info = H(feature)`, so `gain-ratio = info-gain / intrinsic-info` which will be in the range of `[0-1]`

> To put it more precisely, the information gain (mutual information) is always biased upward. It is severely biased when two conditions meet: you have small sample, and you have many variable levels. This stems from the fact that in general practice one calculates naive information gain, or in other words, a sample estimate (point estimate). This estimate will almost surely be affected by deviations of observed probability estimates from theoretical. More variable levels multiplied by lower number of observations will exaggerate the observed probability deviations. It is called the limited sampling bias.


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

### Advantages of Decision Tree.
1. easy to explain
2. follows same strategy as human takes decision
3. easy to interpret 

### Drawbacks of Decision Tree.
1. prone to overfitting(need extra care)
2. greedy approach(each split is locally optimal, so this is NP-complete problem)
3. Information gain in a decision tree with categorical variables gives a `biased response` for attributes with greater no. of categories.
4. computationally expensive with many classes.


---

## Bagging:
- bagging is very powerful method. It is generally the combination of many `decorrelated`(**very desirable**) models.
- reduce the variance as `(sig1 + sig2 +... sign) / n`
- `out of bag` error, which is error computes on leave out observation. As in bootstrap sampling, it choose `68%` of observation to fit the model, the other `32%` will be used for validation, which is called as `out of bag error`.

## Random forest:
- Random Forest is an ensemble model of decision trees(`bagging` which helps in `reduce variance`).
- it is based upon a simple but powerful idea to reduce variance, by choose `m` predictors out of `p` feature. for exp: `m can be chosen as sqrt(p)`
- use less number of predictors/features for each bag, it can reduce the variance, beacuse of no/less correlation between each bag.
- `time complexity`: to building a complete unpruned decision tree is `O( N * v * n log(n))`, where `n` is the number of observations, `v` is the number of variables/attributes and `N` is number of estimators.


### `limitations of Random forest are` :
1. Correlated features will be given equal or similar importance, but overall reduced importance compared to the same tree built without correlated counterparts.
2. Random Forests and decision trees, in general, give preference to features with high cardinality ( `Trees are biased to these type of variables` ). **For reference look at gain-ratio** to handle this

> Note: In statistics, the terms predictor is used instead of features.

---

> An interseting method for feature selection is we can select only those predictors, which have high variance. Note that there is no cheating, because it doesn't use label to do that. We can just build a tree and find the variance of each feature, from the splitting decision.


## Boosting:
- build tree sequentially on residuals
1. start with equal weight `Dt(n) = 1/n` for each sample.
2. find error `et = sum_n Dt(n) |yt == yhatt|`
3. Compute importance `alphat = 1/2 log((1 - et)/et)`
4. update weight `D{t+1}(n) = Dt(n) exp(beta)`, where beta = 
    `-alpha for correct prediction` 
    `alpha for incorrect`
5. Normalize `D{t+1}(n)`
6. Go to step 2. untill converge


## important stuff on bagging and boosting
- In order to run Random Forests we need to select 2 parameters: 
    1. number of samples `B`
    2. `m` = number of variables sampled at each split.
- In order to run Boosting Algorithm, we need to select 3 parameters: 
    1. number of samples `B`
    2. tree depth `d`
    3. step size (learning rate)


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

### Effect of idf on ranking
- idf has no effect on ranking one term queries like iPhone
– idf affects the ranking of documents for queries **with atleast two terms** 
– For the query `capricious person`, idf weighting makes occurrences of `capricious` count for much more in the final document ranking than occurrences of `person`. 

## Text Normalization:
### Stemming:
- "Stemming is the process of `reducing inflection in words to their root forms` such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language."
- stemming is just `stemming of prefix and suffix` such as `(-ed, -ize, -s, -de, -mis)`
  1. `Porter-stem`: fast, set of 5 rule
  2. `Lanchester-stemming`: slow, set of 120 rule, iterative approach, check character by chracter
  3. `SnowballStemmer`: Non-english work stemmer

### Lemmantization:
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

---

## p-value:
- In statistics, the p-value is the probability of obtaining the observed results of a test, assuming that the null hypothesis is correct. 
- **The P-test statistic typically follows a standard normal distribution when large sample sizes are used.**
- It is the level of marginal significance within a statistical hypothesis test representing the probability of the occurrence of a given event. 
- The p-value is used as an alternative to rejection points to provide the smallest level of significance at which the null hypothesis would be rejected.

#### Interpret Results
- Reject the null hypothesis, if `p-value` is less than significance level(eg. 0.05 that is 95% confidence interval)
- `A P-value measures the strength of evidence in support of a null hypothesis.`
- P-value is the probability for the null hypothesis to be True.


1. A small p-value (typically ≤ 0.05) indicates `strong evidence against the null hypothesis`, so you `reject the null hypothesis`.
2. A large p-value (> 0.05) indicates `weak evidence against the null hypothesis`, so you `fail to reject the null hypothesis`.

```python
lm = LinearRegression()
lm.fit(X,y)
params = np.append(lm.intercept_,lm.coef_)
predictions = lm.predict(X)

newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
# newX = np.append(np.ones((len(X),1)), X, axis=1)
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
print(myDF3)
```

---

## Activation function in deep learning
1. sigmoid function: ``1 / (1 + exp(-x))
2. Tanh : `2 / (1 + exp(-2x))  -1`
3. Relu: `x if x > 0, 0 otherwise`
4. Leaky Relu: 
    - `x if x > 0, `
    - `a*x if x < 0, a is constant`
5. Parameterized Relu:
    - `x if x > 0, `
    - `-w*x if x < 0, w is learnable parameter`
6. Exponential linear unit:
    - `x if x > 0, `
    - `a*(exp(x) - 1) if x < 0`
7. Softplus: `log(1 + exp(x))`

### Why Relu is better:
1. computationaly cheaper
2. `No vanishing Gradient` problem, even with very deep network
3. `sparsity` (add more non-linearity and better generalization, also follows the human mind)
4. It converge faster
6. `Relu6` : clipping gradient to avoid exploding gradient problem.

---

## Gradient Checking:
### Why do we need Gradient Checking?

Back prop as an algorithm has a lot of details and can be a little bit tricky to implement. And one unfortunate property is that there are many ways to have subtle bugs in back prop. So that if you run it with gradient descent or some other optimizational algorithm, it could actually look like it's working. And your cost function, J of theta may end up decreasing on every iteration of gradient descent. But this could prove true even though there might be some bug in your implementation of back prop. So that it looks J of theta is decreasing, but you might just wind up with a neural network that has a higher level of error than you would with a bug free implementation. And you might just not know that there was this subtle bug that was giving you worse performance. So, what can we do about this? There's an idea called gradient checking that eliminates almost all of these problems.
### What is Gradient Checking?

We describe a method for numerically checking the derivatives computed by your code to make sure that your implementation is correct. Carrying out the derivative checking procedure significantly increase your confidence in the correctness of your code.

If I have to say in short than Gradient Checking is kind of debugging your back prop algorithm. Gradient Checking basically carry out the derivative checking procedure. 

---

## Weight Init in deep learning:
- A too-large initialization leads to exploding gradients
- A too-small initialization leads to vanishing gradients
III   How to find appropriate initialization values
- To prevent the gradients of the network’s activations from vanishing or exploding, we will stick to the following rules of thumb:
    1. The mean of the activations should be zero.
    2. The variance of the activations should stay the same across every layer.
    Under these two assumptions, the backpropagated gradient signal should not be multiplied by values too small or too large in any layer. It should travel to the input layer without exploding or vanishing.
- More concretely, consider a layer l. Its forward propagation is:
    - `z[l] = W[l]a[l−1] + b[l]`
    - `a[l] = g[l](z[l])`
- We would like the following to hold:
    1. `E[a[l−1]] = E[a[l]]`
    2. `Var(a[l−1]) = Var(a[l])`
- Final result:
    1. `w ~ N(0, 1/n_prev` where `n_prev` is the number of neuron in previous layer
    2. `b = 0`

### Xavier Init:
- widely used, with the activation function like `tanh`
- Using `forward pass` relation, we come up at
    1. `Var(a[l]) = n[l−1] * Var(W[l]) * Var(a[l−1])`
    2.  So `Var(W[l]) = 1 / n[l-1]`, to ensure the above property to avoid exploding and vanishing graient problem
- Using `backward pass` relation, we come up at, `Var(a[l]) = n[l−1] * Var(W[l]) * Var(a[l−1])`
    1. `Var(a[l-1]) = n[l] * Var(W[l]) * Var(a[l])`
- In practice, we consider `harmonic mean of both variance`, which is `2 / (n[l] + n[l-1])`

- In general, `Var(a[L]) = [prod{l=1:L} ​n[l−1] Var(W[l])] Var(x)​`
    Depending on how we initialize our weights, the
        [prod{l=1:L} ​n[l−1] Var(W[l])] < 1 ⟹ Vanishing Signal
        [prod{l=1:L} ​n[l−1] Var(W[l])] = 1 ⟹ Var(a[L]) = Var(x)
        [prod{l=1:L} ​n[l−1] Var(W[l])] > 1 ⟹ Exploding Signal

### He Init:
- work great with `relu`
- `Var(a[l]) = sqrt(2 / n_prev)`, `n_prev = fan_in`

---

## Cross Validation:
- K-fold
- startified k-fold
- leave one out
- leave-p out
- group-k fold
- leave-1 group out
- leave-p group out
- time-series split
- all above with shuffling

---

## Batch Normalization:
- batch-norm solve the problem of **local-covariate-shift**
- compute mean and variance of current batch
- normalize the dataset with that
- scale and shift the output, to add more non-linearity
    1. When we normalize, it will be `0-1`, while using `sigmoid` at each layer, output will be linear, beacuse, most of the time it lies around `0.5`, where curve is almost linear.
    2. To avoid that, we learn a scaling and shifting factor too, which is applied at output of batch-norm layer.

### Advantage of batch-norm
1. Faster training
    1. We can use higher learning rates because batch normalization makes sure that there’s no activation that’s gone really high or really low. And by that, things that previously couldn’t get to train, it will start to train.

2. It reduces overfitting
    1. As it normalize the dataset as per mean and variance of current batch observation, which may not be equal to global mean and vaiance
    2. Due to above, it add some noise in dataset, which act as regularization similar to `dropout`

---

## Statistics:
1. Descriptive Statistics

Descriptive Statistics is the study of understanding patterns that might emerge from data. It is a way to summarize our data and interpret it in a meaningful way. It includes important attributes of the dataset like mean, mode, median and also the deviation or measuring the spread. These attributes help guide us to know the quality of convergence during model evaluation.
2. Inferential Statistics

Sometimes it is not feasible to consume the entire model. This is where sampling comes in. Sampling is of great importance in inferential statistics and is the basis of breaking down data into samples for training, validation and test for your AI models. Sampling estimation and testing for hypothesis are two main aspects of inferential statistics.

---

## Credit Risk:
The idea of consumer credit is that the bank will gain from giving credit only if the client will not default (that means, will not repay the debt). Indeed, once accepted as creditworthy and received the credit, the client will plan, together with the bank, an amortization schedule according to which he will have to repay not only the debt, but also the interests.

So, if from one side giving credit is one of the incomes of a bank (because of the interests), from the other it involves a noticeable amount of risk. That’s why a great amount of time and money is invested in analyzing clients’ history, habits and likelihood of repaying the debt.

To do so, banks have always been relying on statistical models (especially scoring models), however today, with the aid of Machine Learning algorithms, their predictions about future repayments are far more reliable.

---

## Eigenvalues and Eigenvector
Now, each transformation might affect the direction and extension of a vector (for a clearer explanation about the shape of vectors in multidimensional spaces you can read my former article here). However, given a transformation T, there exists a very interesting class of vectors which are affected by that transformation only in terms of extension, since the direction remains unchanged. The generic vector v with this property is such that:

Where lambda is the extension factor. Those vectors are called Eigenvectors and the value lambda associated with them is called Eigenvalue.
Eigenvalues and Eigenvectors

As anticipated, eigenvectors are those vector whose direction remains unchanged once transformed via a fixed T, while eigenvalues are those values of the extension factor associated with them.

To be more precise, eigenvectors are vectors which are not trivial, hence different from 0. That’s because the equality above has always at least one solution, which is the trivial one where v=0.

How can we find our eigenvectors and eigenvalues, under the condition that those former are different from the trivial vector? For this purpose, let’s reframe our linear system with the representation theorem:

As anticipated, this system has at least one solution, which is the trivial one. Hence, we want to find those values of lambda for which the determinant of the matrix (A-lamda*I) is equal to zero (otherwise it would have meant that, because of the Cramer Theorem, the system has 1 unique solution).

- Some of them (more specifically, as many as the number of features), though, have a very interesting property: indeed, once applied the transformation T, they change length but not direction. Those vectors are called eigenvectors, and the scalar which represents the multiple of the eigenvector is called eigenvalue

---

## MLE:
- relies only on `likelihood`
- `theta = argmax_theta log(p(y/theta))`
- while doing MLE on `gaussian dist`, we assume `beta = 1 / sigma^2`, it will ease some of calculation
    1. `p(y/theta) = N(y; w x, beta^(-1))`
    2. `log(p(y/theta)) = sqrt(beta / (2 pi)^D) exp(- beta/2 (yn - w x)^2)`, where `w x is mean`
    3. solving above with MLE, we get 
        - `mean = 1 / N sum_n yn`
        - `beta^(-1) = sum_n (yn - w xn)^2 `
        - `sigma^2 = sum_n (yn - w xn)^2 / N`


## MAP:
- include `prior probabily as well`
- regularized loss function
- to solve for posterior probability, use prior as `conjugate prob` with the `likelihood`
- `p(theta/y) = P(y/theta) P(theta) / p(y)`
- `theta = argmax_theta log(p(y/theta)) + log(p(theta))`


### Conjugate-pair:
- Poisson : Gamma
- Beta : Bernoulli
- Beta : Binomial
- Dirichlet : Multnoulli
- Dirichlet : Multinomial
- Gaussian : Gaussian
- Gaussian : Gamma
- Gaussian : Inverse Gamma

---

## Parametric vs. Non-Parametric Models:
1. A parametric algorithm has a fixed number of parameters.  A parametric algorithm is computationally faster, but makes stronger assumptions about the data; the algorithm may work well if the assumptions turn out to be correct, but it may perform badly if the assumptions are wrong.  A common example of a parametric algorithm is linear regression.

2. In contrast, a non-parametric algorithm uses a flexible number of parameters, and the number of parameters often grows as it learns from more data.  A non-parametric algorithm is computationally slower, but makes fewer assumptions about the data.  A common example of a non-parametric algorithm is K-nearest neighbour.

To summarize, the trade-offs between parametric and non-parametric algorithms are in computational cost and accuracy


 in a parametric model, we have a finite number of parameters, and in nonparametric models, the number of parameters is (potentially) infinite. Or in other words, in nonparametric models, the complexity of the model grows with the number of training data; in parametric models, we have a fixed number of parameters (or a fixed structure if you will).

Linear models such as linear regression, logistic regression, and linear Support Vector Machines are typical examples of a parametric “learners;” here, we have a fixed size of parameters (the weight coefficient.) In contrast, K-nearest neighbor, decision trees, or RBF kernel SVMs are considered as non-parametric learning algorithms since the number of parameters grows with the size of the training set. – K-nearest neighbor and decision trees, that makes sense, but why is an RBF kernel SVM non-parametric whereas a linear SVM is parametric? In the RBF kernel SVM, we construct the kernel matrix by computing the pair-wise distances between the training points, which makes it non-parametric.

In the field of statistics, the term parametric is also associated with a specified probability distribution that you “assume” your data follows, and this distribution comes with the finite number of parameters (for example, the mean and standard deviation of a normal distribution); you don’t make/have these assumptions in non-parametric models. So, in intuitive terms, we can think of a non-parametric model as a “distribution” or (quasi) assumption-free model.

---

## Expectation Minimization:
- It is approximation of `alternative optimization` method
- one of favorite algorithm of statistician
- `log(p(x/theta)) = E_q(z) p(x,z/theta) + KL(q(z) || p(z/x, theta))`, where `z` is latent variable
- it works for any `q(z)`
- `lower bound definition`: `log(p(x/theta)) >= E_q(z) p(x,z/theta)`
- when target is unknown, then `MLE` doesn't work straight forward, For exp, look at `GMM` problem formulation.
- `Application`
    1.semi-supervised generative classification
    2. Probabilistic clustering
    3. mixture density estimation

---

## Gaussian Mixture Model
- while solving generative classification, we solve for `p(X,y/theta)`
- for `k-class gaussian class conditional` is defined as 
    - `p(X,y/theta) = p(X/y,theta) p(y/theta)`
    - `p(y/theta)` is `muliutnoulli dist` as `pi_k`
    - `p(X/y,theta) = N(mu_k, sigma_k^2)`
    - p(X,y/theta) = `(pi_k * N(mu_k, sigma_k^2))^ynk`, which mean each observation is relate to only one of class of gaussian distribution, which is defined  as ynk.
    - If we know `ynk` in advance, then `log` of whole term goes inside and otherwise we cann't do that
- `log(p(X,y/theta)) = log(sum_n sum_k (pi_k * N(mu_k, sigma_k^2))^ynk )`
- If `ynk` is known,
    - `log(p(X,y/theta)) = log(sum_n sum_k ynk [log(pi_k) + log(N(mu_k, sigma_k^2))]`
- otherwise,
    - `log(p(X,y/theta)) = sum_n log(sum_k [pi_k * N(mu_k, sigma_k^2)] )`

> Note ynk is nothing what znk, that is hidden latent variable

### Solution of GMM:
- `with ALT-OPT`:
    1. first find `znk` that is `argmax_k p(Z/X,theta)`
    2. with known label, we have nice MLE, where we optimize for `mean_k` and `var_k`
- `with EM`
    1. change `znk` to `expectation` as `E[znk]`
    2. `E[znk] = 0 * p(zn = 0) + 1 * p(zn = 1/xn)`
        - `E[znk] = p(zn = 1/xn) = p(zn = 1) p(xn / zn = 1)`
        - `E[znk] = pi_k N(xn; mu_k, sigma_k^2)`
    3. using `2`, we have nice `MLE`, where we optimize for `mean_k` and `var_k`

---

## Domain Adaptation:
- when train and test data are of different disribution, we can adapt that test distribution using following method: (similar to weighted important sampling)
- `w(x) = sum_i alpha_i phi_i(x)`
- `p_adapt(x) = w(x) p_test(x)`
- `kl[p_train(x) || p_adapt] = integ p_train(x) log( p_train(x) / p_adapt(x))`
- `maximize_alpha {sum_{n=1:N} log(sum_i alpha_i phi_i(x))}, subject to sum_n sum_i alpha_i phi_i(x) = 1, where alpha_i > 0`

---


## Naive Bayes Algorithm:
Assumption:

The fundamental Naive Bayes assumption is that each feature makes an:

    independent
    equal

contribution to the outcome.


    Step 1: Calculate the prior probability for given class labels
    Step 2: Find Likelihood probability with each attribute for each class
    Step 3: Put these value in Bayes Formula and calculate posterior probability.
    Step 4: See which class has a higher probability, given the input belongs to the higher probability class.

For simplifying prior and posterior probability calculation you can use the two tables frequency and likelihood tables. Both of these tables will help you to calculate the prior and posterior probability. The Frequency table contains the occurrence of labels for all features. There are two likelihood tables.



#### Zero Probability Problem

Suppose there is no tuple for a risky loan in the dataset, in this scenario, the posterior probability will be zero, and the model is unable to make a prediction. This problem is known as Zero Probability because the occurrence of the particular class is zero.

The solution for such an issue is the Laplacian correction or Laplace Transformation. Laplacian correction is one of the smoothing techniques. Here, you can assume that the dataset is large enough that adding one row of each class will not make a difference in the estimated probability. This will overcome the issue of probability values to zero.

For Example: Suppose that for the class loan risky, there are 1000 training tuples in the database. In this database, income column has 0 tuples for low income, 990 tuples for medium income, and 10 tuples for high income. The probabilities of these events, without the Laplacian correction, are 0, 0.990 (from 990/1000), and 0.010 (from 10/1000)

Now, apply Laplacian correction on the given dataset. Let's add 1 more tuple for each income-value pair. The probabilities of these events:

### Advantages

    It is not only a simple approach but also a fast and accurate method for prediction.
    Naive Bayes has very low computation cost.
    It can efficiently work on a large dataset.
    It performs well in case of discrete response variable compared to the continuous variable.
    It can be used with multiple class prediction problems.
    It also performs well in the case of text analytics problems.
    When the assumption of independence holds, a Naive Bayes classifier performs better compared to other models like logistic regression.

### Disadvantages

    The assumption of independent features. In practice, it is almost impossible that model will get a set of predictors which are entirely independent.
    If there is no training tuple of a particular class, this causes zero posterior probability. In this case, the model is unable to make predictions. This problem is known as Zero Probability/Frequency Problem.

### Application:
Real time Prediction
Multi class Prediction
Recommendation System

### Types:
There are three types of Naive Bayes model under the scikit-learn library:
1. Gaussian: It is used in classification and it assumes that features follow a normal distribution.
2. Multinomial: It is used for discrete counts. For example, let’s say,  we have a text classification problem. Here we can consider Bernoulli trials which is one step further and instead of “word occurring in the document”, we have “count how often word occurs in the document”, you can think of it as “number of times outcome number x_i is observed over the n trials”.
3. Bernoulli: The binomial model is useful if your feature vectors are binary (i.e. zeros and ones). One application would be text classification with ‘bag of words’ model where the 1s & 0s are “word occurs in the document” and “word does not occur in the document” respectively.

---
## Probabilistic Models:
Probabilistic models see features and target variables as random variables. The process of modelling represents and manipulates the level of uncertainty with respect to these variables. There are two types of probabilistic models: Predictive and Generative. Predictive probability models use the idea of a conditional probability distribution P (Y |X) from which Y can be predicted from X.  Generative models estimate the joint distribution P (Y, X).  Once we know the joint distribution for the generative models, we can derive any conditional or marginal distribution involving the same variables. Thus, the generative model is capable of creating new data points and their labels, knowing the joint probability distribution. The joint distribution looks for a relationship between two variables. Once this relationship is inferred, it is possible to infer new data points.

- they are really good at capturing uncertainty
- prob-model can be very useful when we have certain knowledge about the dataset and other proberties, which can help in defining prior probability
- a closed form solution is also possible if function(objective funtion) is tractable like in bayesian model
- MLE is very helpful for estimation for likelihood based models.
- add stochasticity. For exp: while doing clustering such as kmean, it assume constant variance(circular) for all clusters. But a probabilstic version of it, can imrove the model.

- Can get estimate of the theuncertaintyin the parameter estimates via theposterior distribution
- Useful when we only have limited data for learning each parameter
- Can get estimate of the theuncertainty in the model’s predictionsE.g., Instead of a single predictiony∗, we get a distribution over possiblepredictions (useful for applications such as diagnosis, decision making, etc.)
    - p(y∗|x∗,θ) = ∫ p(y∗|x∗,θ) p(θ|X,y) dθ
- Can handle missing and noisy data in a principled way
- Easy/more natural to do semi-supervised learning, active learning, etc.
- Can generate(synthesize) data by simulating from the data distribution
- Hyperparameters can be learned from data (need not be tuned)


---

## Skewness vs Kurtusis
### Skewness

It is the degree of distortion from the symmetrical bell curve or the normal distribution. It measures the lack of symmetry in data distribution.
It differentiates extreme values in one versus the other tail. A symmetrical distribution will have a skewness of 0.

There are two types of Skewness: Positive and Negative

Positive Skewness means when the tail on the right side of the distribution is longer or fatter. The mean and median will be greater than the mode.

Negative Skewness is when the tail of the left side of the distribution is longer or fatter than the tail on the right side. The mean and median will be less than the mode.
So, when is the skewness too much?

The rule of thumb seems to be:

    If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.
    If the skewness is between -1 and -0.5(negatively skewed) or between 0.5 and 1(positively skewed), the data are moderately skewed.
    If the skewness is less than -1(negatively skewed) or greater than 1(positively skewed), the data are highly skewed.

### Kurtosis

Kurtosis is all about the tails of the distribution — not the peakedness or flatness. It is used to describe the extreme values in one versus the other tail. It is actually the measure of outliers present in the distribution.

High kurtosis in a data set is an indicator that data has heavy tails or outliers. If there is a high kurtosis, then, we need to investigate why do we have so many outliers. It indicates a lot of things, maybe wrong data entry or other things. Investigate!
Low kurtosis in a data set is an indicator that data has light tails or lack of outliers. If we get low kurtosis(too good to be true), then also we need to investigate and trim the dataset of unwanted results.


---
## References:
1. Hadamard : element wise multiplication
2. Image Processing Algo: https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-3-greyscale-conversion/
3. [Data science prep by amazon](https://www.quora.com/What-is-the-best-site-for-preparing-data-science-interview)
4. [Various Question for all section](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-data-science-machine-learning-interview-guide/)
5. importance of positve definite matrix
  - In optimization problem, positve definiteness `guarantedd for existance for optima`
  - It is a class of symetric matrix, which has huge application in algebra. For example to find the inverse, it is O(n^3) and even impossible to compute for very high dimensional space because of memory constarint. `But using symetrical property, we can decoompose the matrix (using Cholesky decomposition) as A = L.L', where L is lower trainagular matrix`
6. [amazon interview question](https://medium.com/acing-ai/amazon-ai-interview-questions-acing-the-ai-interview-3ed4e671920f)