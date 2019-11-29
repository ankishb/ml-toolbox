
2. Typical Use Cases

    Ridge: It is majorly used to prevent overfitting. Since it includes all the features, it is not very useful in case of exorbitantly high #features, say in millions, as it will pose computational challenges.
    Lasso: Since it provides sparse solutions, it is generally the model of choice (or some variant of this concept) for modelling cases where the #features are in millions or more. In such a case, getting a sparse solution is of great computational advantage as the features with zero coefficients can simply be ignored.

Its not hard to see why the stepwise selection techniques become practically very cumbersome to implement in high dimensionality cases. Thus, lasso provides a significant advantage.

 
3. Presence of Highly Correlated Features

    Ridge: It generally works well even in presence of highly correlated features as it will include all of them in the model but the coefficients will be distributed among them depending on the correlation.
    Lasso: It arbitrarily selects any one feature among the highly correlated ones and reduced the coefficients of the rest to zero. Also, the chosen variable changes randomly with change in model parameters. This generally doesn’t work that well as compared to ridge regression.

This disadvantage of lasso can be observed in the example we discussed above. Since we used a polynomial regression, the variables were highly correlated. ( Not sure why? Check the output of data.corr() ). Thus, we saw that even small values of alpha were giving significant sparsity (i.e. high #coefficients as zero).



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

```c++
// gauss jordan method

// C++ Implementation for Gauss-Jordan 
// Elimination Method 
#include <bits/stdc++.h> 
using namespace std; 
  
#define M 10 
  
// Function to print the matrix 
void PrintMatrix(float a[][M], int n) 
{ 
    for (int i = 0; i < n; i++) { 
        for (int j = 0; j <= n; j++)  
          cout << a[i][j] << " "; 
        cout << endl; 
    } 
} 
  
// function to reduce matrix to reduced 
// row echelon form. 
int PerformOperation(float a[][M], int n) 
{ 
    int i, j, k = 0, c, flag = 0, m = 0; 
    float pro = 0; 
      
    // Performing elementary operations 
    for (i = 0; i < n; i++) 
    { 
        if (a[i][i] == 0)  
        { 
            c = 1; 
            while (a[i + c][i] == 0 && (i + c) < n)  
                c++;             
            if ((i + c) == n) { 
                flag = 1; 
                break; 
            } 
            for (j = i, k = 0; k <= n; k++)  
                swap(a[j][k], a[j+c][k]); 
        } 
  
        for (j = 0; j < n; j++) { 
              
            // Excluding all i == j 
            if (i != j) { 
                  
                // Converting Matrix to reduced row 
                // echelon form(diagonal matrix) 
                float pro = a[j][i] / a[i][i]; 
  
                for (k = 0; k <= n; k++)                  
                    a[j][k] = a[j][k] - (a[i][k]) * pro;                 
            } 
        } 
    } 
    return flag; 
} 
  
// Function to print the desired result  
// if unique solutions exists, otherwise  
// prints no solution or infinite solutions  
// depending upon the input given. 
void PrintResult(float a[][M], int n, int flag) 
{ 
    cout << "Result is : "; 
  
    if (flag == 2)      
      cout << "Infinite Solutions Exists" << endl;     
    else if (flag == 3)      
      cout << "No Solution Exists" << endl; 
      
      
    // Printing the solution by dividing constants by 
    // their respective diagonal elements 
    else { 
        for (int i = 0; i < n; i++)          
            cout << a[i][n] / a[i][i] << " ";         
    } 
} 
  
// To check whether infinite solutions  
// exists or no solution exists 
int CheckConsistency(float a[][M], int n, int flag) 
{ 
    int i, j; 
    float sum; 
      
    // flag == 2 for infinite solution 
    // flag == 3 for No solution 
    flag = 3; 
    for (i = 0; i < n; i++)  
    { 
        sum = 0; 
        for (j = 0; j < n; j++)         
            sum = sum + a[i][j]; 
        if (sum == a[i][j])  
            flag = 2;         
    } 
    return flag; 
} 
  
// Driver code 
int main() 
{ 
    float a[M][M] = {{ 0, 2, 1, 4 },  
                     { 1, 1, 2, 6 },  
                     { 2, 1, 1, 7 }}; 
                       
    // Order of Matrix(n) 
    int n = 3, flag = 0; 
      
    // Performing Matrix transformation 
    flag = PerformOperation(a, n); 
      
    if (flag == 1)      
        flag = CheckConsistency(a, n, flag);     
  
    // Printing Final Matrix 
    cout << "Final Augumented Matrix is : " << endl; 
    PrintMatrix(a, n); 
    cout << endl; 
      
    // Printing Solutions(if exist) 
    PrintResult(a, n, flag); 
  
    return 0; 
} 
```


## Difference between gauss-jordan and gauss-elimination method:
Both methods are used to find solutions for linear systems by pivoting and elimination like as Ax⃗ =b⃗ 

Gauss method end the matrix as a superior-triangular matrix and you find the solutions of a linear system by applying a regressive substitution.

Gauss-Jordan do the same as an additional: turn the desired current matrix A a identity matrix. Thus, Gauss-Jordan means Gauss method plus doing the operation sufficient to make the matrix triangular inferior as well, which ends in a identity matrix.

With Gauss-Jordan method you have the exactly solution that you want without the need of regressive/progressive substitution. The desired solution is directly encoded on the vectorb⃗ .

## Fastest method to solve for inverse of matrix (or find the solution for set of linear equation in general)
1. Fast method would be the Gauss-Jordan method. 
    - it reduce the matric to `reduced row echlon form`, which gives us inverse matrix directly
    - This is what gauss jordan method do: `[A | I] ==> [I | A^(-1)]`

2. Next would be LU Decomposition method.
    - A X = B -> L U X = B -> L Y = B
    - To find L and `U`, we can simply first run gauss elimination to get `U`, then to find `L` we use `forward elimination method`
    - L =  [ 1 0]
           [ * 1]
    - we can easily find value of * in L
    - Once we find `L` and `U`, we first solve `L Y = B`, and find values for `Y`
    - Then we use `U X = Y` and solve for `X`

3. And last would be using Gauss elimination. 
    - it reduce the matrix to `row echlon form`, which is use to solve the linear equation, to find coefficients of linear equation
    - A X = B -> `[A] => [A']` 
4. (cramer rule)Using the co-factor method would be the least efficient method - just see how much time it takes to find determinant of a matrix.


---







Precision means the percentage of your results which are relevant. On the other hand, recall refers to the percentage of total relevant results correctly classified by your algorithm.








Questions:

Ask yourself the following questions to get clear understanding about both these models.

    ‌ What are the problems these models can solve?
    ‌Which model learns joint probability?
    ‌Which model learns conditional probability?
    ‌What happens when we give correlated features in discriminative models?
    ‌What happens when we give correlated features in generative models?
    ‌Which models works very well even on less training data?
    Is it possible to generate data from with the help of these models?
    ‌Which model will take less time to get trained?
    ‌Which model will take less time to predict output?
    ‌Which model fails to work well if we give a lot of features?
    ‌Which model prone to overfitting very easily?
    ‌Which model prone to underfitting easily?
    ‌What happens when training data is biased over one class in Generative Model?
    ‌What happens when training data is biased over one class in Discriminative Models?
    ‌Which model is more sensitive to outliers?
    ‌Can you able to fill out the missing values in a dataset with the help of these models?







## Data Wrangling:
In very brief, data wrangling is all about transforming raw data into other form, which can lead to better visualization, better decision, less time consumption for data scientist or machine learning engeineer


What is Data Wrangling?

Data wrangling is the process of cleaning, structuring and enriching raw data into a desired format for better decision making in less time. Data wrangling is increasingly ubiquitous at today’s top firms. Data has become more diverse and unstructured, demanding increased time spent culling, cleaning, and organizing data ahead of broader analysis. At the same time, with data informing just about every business decision, business users have less time to wait on technical resources for prepared data.

It allows analysts to tackle more complex data more quickly, produce more accurate results, and make better decisions.



There are typically 5 iterative steps that make up the data wrangling process.
1. `Discovering`: Before you can dive deeply, you must better understand what is in your data, which will inform how you want to analyze it. How you wrangle customer data, for example, may be informed by where they are located, what they bought, or what promotions they received.
2. `Structuring`: This means organizing the data, which is necessary because raw data comes in many different shapes and sizes. A single column may turn into several rows for easier analysis. One column may become two. Movement of data is made for easier computation and analysis.
3. `Cleaning`: What happens when errors and outliers skew your data?  You clean the data. What happens when state data is entered as CA or California or Calif.? You clean the data. Null values are changed and standard formatting implemented, ultimately increasing data quality.
4. `Enriching`: Here you take stock in your data and strategize about how other additional data might augment it. Questions asked during this data wrangling step might be: what new types of data can I derive from what I already have or what other information would better inform my decision making about this current data?
5. `Validating`: Validation rules are repetitive programming sequences that verify data consistency, quality, and security. Examples of validation include ensuring uniform distribution of attributes that should be distributed normally (e.g. birth dates) or confirming accuracy of fields through a check across data.


## How to deal with any DS problem:
1. Collect data
2. EDA
3. Handling Missing value
4. Handle Outliers
5. Check Correlation(feature transformation such as PCA)
6. Imbalanced Target distribution
7. Feature Cleaning(BTI to Bathinda, all lower case, etc)
8. Feature engineering
    - Transformations (normalisation, standardisation, scaling, pivoting, ...)
    - Data Replacement (cutting, splitting, merging, ...)
    - Attribute Generation (ID generation, ...)
    - Imputation (replacement of missing observations by using statistical algorithms)
    - Binning (count-based, handling of missing values as its own group, …)


Feature Engineering selects the right attributes to analyze. You use domain knowledge of the data to select or create attributes that make machine learning algorithms work. Feature Engineering process includes:
1. Feature engineering
2. Feature selection
3. Validation of how the features work with your model
4. Improvement of features, if needed
5. Return to brainstorming / creation of more features until the work is done


## Measuring Similarity and Distance
Distance between numeric data points

When the dimension of a data point is numeric, the general form is called the Minkowski distance, as shown here:

p = ∞, Chebychev Distance

( (x 1 - y 1) p + (x 2 - y 2) p ) 1/p


When p = 2, this is equivalent to Euclidean distance.  When p = 1, this is equivalent to Manhattan distance.


This measure is independent of the underlying data distribution.  But what if the value along the x-dimension is much bigger than the value along the y-dimension?  We need to bring all of them down to scale first.  A common way is to perform a z-transform where each data point first subtracts the mean value, then divides the standard deviation.



(x 1, y 1) becomes ( (x 1 – μ x)/σ x , (y 1 – μ y)/σ y )


This measure, although taking into consideration the distribution of each dimension, assumes the dimensions are independent of each other. But what if the x-dimension has some correlation with the y-dimension? To consider correlations between different dimensions, we use this:



Mahalanobis distance = (v 1→-v 2→)T.CovMatrix.(v 1→-v 2→)where v 1→= (x1, y1)


 

If we care about the direction of the data rather than the magnitude, then using the cosine distance is a common approach. It computes the dot product of the two data points divided by the product of their magnitude. Cosine distance, together with the term/document matrix, is commonly used to measure the similarity between documents.

### Distance between categorical data points
Since we have no ordering between categorical values, we can only measure whether the categorical values are the same or not. Basically we are measuring the degree that attribute values overlap. `Hamming distance` can be used to measure how many attributes must be changed in order to match one another. We can calculate the ratio to determine the similarity (or difference) between two data points using the simple matching coefficient:

noOfMatchAttributes / noOfAttributes
- However, in some cases, equality of certain values don't mean anything. For example, let's say the data point represents a user with attributes representing each movie. The data point contains a high dimensional binary value indicating that the user has or has not seen the movie (1 represent yes and 0 represent no). Given that most users only see a small portion of all movies, if both users haven't seen a particular movie (a value of 0 for both), it doesn't indicate a similarity between the users. On the other hand, if both users saw the same movie (a value of 1 for each), it is implied that the users have many similarities. In this case, equality of 1 should carry a much higher weight than equality of 0. This leads to the `Jaccard similarity`, as seen here:

noOfOnesInBoth / (noOfOnesInA + noOfOnesInB - noOfOnesInAandB)


Whether or not values are matching, though, if the category is structured as a Tree hierarchy, then the distance between the two categories can be quantified by the path length of their common parent. For example, "/product/spot/ballgame/basketball" is closer to "/product/spot/ballgame/soccer/shoes" than "/product/luxury/handbags" because the common parent has a longer path.

Distance between mixed categorical and numeric data points

When the data point contains a mixture of numeric and categorical attributes, we can calculate the distance of each group and then treat each measure of distance as a separate dimension (numeric value).


distance final = α.distance numeric + (1- α).distance categorical



## Distance between sequence (String, TimeSeries)
In case each attribute represents an element of a sequence, we need a different way to measure the distance. For example, let's say each data point is a string (which contains a sequence of characters) — then edit distance is a common measuring tool. Basically, `edit distance` reveals how many "modifications" (which can be insert, modify, delete) are needed to change stringA into stringB. This is usually calculated by using thedynamic programming technique.


Time Series is another example of sequence data. Similar to the concept of edit distance, Dynamic Time Warp distorts the time dimension by adding more data points in both time series, minimizing the square error between corresponding pairs. We discover where to add these data points by using a similar dynamic programming technique.  Here is a very good paper that describe the details.

### Distance between nodes in a network

In a homogenous undirected graph (nodes are of the same type), distance between nodes can be measured by the shortest path.


In a bi-partite graph, there are two types of nodes in which each node only connects to the other type.  (e.g., People joining communities.) Node similarity can be measured by analyzing how similar their connected communities are as long as the nodes are the same type.
`SimRank` is an iterative algorithm that computes the similarity of nodes. It does this by adding up the similarities between all node pairs of different types. Other types of node similarities are computed in the same way.





We can also use a probabilistic approach, such as RandomWalk, to determine the similarity. Each 'people node' will pass a token (label with the people's name) along a randomly picked, connected 'community node' (weighted by the strength of connectivity). Each community node will propagate the received token back to a randomly picked people node. Now the people who received the propagated token may drop the token (with a chance beta) or propagate it to a randomly chosen community again. This process continues until all the tokens are gone (since they have a chance of being dropped). After that, we obtain the trace Matrix and compute the similarity based on the dot product of the tokens it receives.




### Distance between population distribution

Instead of measuring distance between individual data points, we can also compare a collection of data points (e.g., populations), and measure the distance between them. In fact, one important part of statistics is to measure the distance between two groups of samples to see if the "difference" is significant enough to conclude they are from different populations.
Let's say the population contains members that belong to different categories and we want to measure if population A and population B have the same or different proportions of members across such categories. We can use `Chi-Square` or `KL-Divergence` to measure their distance of separation.


In case every member of the population has two different numeric attributes (e.g. weight and height), and we want to infer one attribute from the other, given that they correlate, the correlation coefficient quantifies their degree of correlation. And it does not matter whether these two attributes are moving along the same direction (heavier people are taller), in a different direction (heavier people are shorter), or independently of one another.  The correlation coefficient ranges from -1 (negatively correlated) to 0 (no correlation) to 1 (positively correlated).


If the two attributes are categorical (rather than numeric), then mutual information is a common way to measure their dependencies. This determines whether knowing the value of one attribute will help infer the value of the other.
Suppose two judges rank a collection of items and we are interested in how much their ranking orders agree.  We can use `Spearman's rank coefficient` to measure their degree of consensus in the ranking order.

 
### Cosine similarity: 
to determine how similar two documents or corpus of text are.

### Jaccard distance: Lastly, we will change our focus of attention. Instead of calculating distances between vectors, we will work with sets.

A set is an unordered collection of objects. So for example, {1, 2, 3, 4} is equal to {2, 4, 3, 1}. We can calculate its cardinality (represented as |set|) which is no other thing than the number of elements contained in the set.









## Time series is strong stationary given that unconditional joint probability distribution does not change when shifted in time, this means that the distribution is the same through time. In weak stationary, a time series is when the mean and the variance are constant through time.

Stationary process is the one which generates time-series values such that distribution mean and variance is kept constant. Strictly speaking, this is known as weak form of stationarity or covariance/mean stationarity.

Weak form of stationarity is when the time-series has constant mean and variance throughout the time.

Let's put it simple, practitioners say that the stationary time-series is the one with no trend - fluctuates around the constant mean and has constant variance.

Covariance between different lags is constant, it doesn't depend on absolute location in time-series. For example, the covariance between t and t-1 (first order lag) should always be the same
A strong form of stationarity is when the distribution of a time-series is exactly the same trough time. In other words, the distribution of original time-series is exactly same as lagged time-series (by any number of lags) or even sub-segments of the time-series. For example, strong form also suggests that the distribution should be the same even for a sub-segments 1950-1960, 1960-1970 or even overlapping periods such as 1950-1960 and 1950-1980. This form of stationarity is called strong because it doesn't assume any distribution. It only says the probability distribution should be the same. In the case of weak stationarity, we defined distribution by its mean and variance. We could do this simplification because implicitly we assumed normal distribution, and normal distribution is fully defined by its mean and variance or standard deviation. This is nothing but saying that probability measure of the sequence (within time-series) is the same as that for lagged/shifted sequence of values within same time-series.




## Technique which works for varying size input

Three possibilities come to mind.

The easiest is the zero-padding. Basically, you take a rather big input size and just add zeroes if your concrete input is too small. Of course, this is pretty limited and certainly not useful if your input ranges from a few words to full texts.

Recurrent NNs (RNN) are a very natural NN to choose if you have texts of varying size as input. You input words as word vectors (or embeddings) just one after another and the internal state of the RNN is supposed to encode the meaning of the full string of words. This is one of the earlier papers.

Another possibility is using recursive NNs. This is basically a form of preprocessing in which a text is recursively reduced to a smaller number of word vectors until only one is left - your input, which is supposed to encode the whole text. This makes a lot of sense from a linguistic point of view if your input consists of sentences (which can vary a lot in size), because sentences are structured recursively. For example, the word vector for "the man", should be similar to the word vector for "the man who mistook his wife for a hat", because noun phrases act like nouns, etc. Often, you can use linguistic information to guide your recursion on the sentence. If you want to go way beyond the Wikipedia article, this is probably a good start.





## 
 How to perform a series of calculations without a calculator and your logic behind the steps.
  Three friends in Seattle told you it's rainy. Each has a probability of 1/3 of lying. What's the probability of Seattle is rainy.   
    The answer is 8/9.
    The only thing we know is that the reality is identical as well as their answers, so if one is telling the truth all of them tell the truth.
    P(truth|identical_answers) = P(Truth and identical_answers) / P(identical_answers) =
    = (8/27) / (8/27 + 1/27) = 8/9.


  1.Can you explain the Naive Bayes fundamentals? How did you set the threshold?
2.Can you explain what MapReduce is and how it works?   
Explain Adam

I) Generate a fair coin from a biased one. II) Generate 7 integers with equal probability from a function which returns 1/0 with probability p and (1-p). These were not worded this way, but essentially this was question.   …  href="/Interview/Of-all-the-questions-he-could-ask-he-picked-the-following-ones-although-to-be-fair-to-the-interviewer-he-can-ask-any-thi-QTN_2637163.htm" class="questionResponse">Answer Question


 What are the ROC curve and the meaning of sensitivity, specificity, confusion matrix   



## External merge sort

One example of external sorting is the external merge sort algorithm, which is a K-way merge algorithm. It sorts chunks that each fit in RAM, then merges the sorted chunks together.[1][2]

The algorithm first sorts M items at a time and puts the sorted lists back into external memory. It then recursively does a M B {\displaystyle {\tfrac {M}{B}}} {\displaystyle {\tfrac {M}{B}}}-way merge on those sorted lists. To do this merge, B elements from each sorted list are loaded into internal memory, and the minimum is repeatedly outputted.

For example, for sorting 900 megabytes of data using only 100 megabytes of RAM:

    Read 100 MB of the data in main memory and sort by some conventional method, like quicksort.
    Write the sorted data to disk.
    Repeat steps 1 and 2 until all of the data is in sorted 100 MB chunks (there are 900MB / 100MB = 9 chunks), which now need to be merged into one single output file.
    Read the first 10 MB (= 100MB / (9 chunks + 1)) of each sorted chunk into input buffers in main memory and allocate the remaining 10 MB for an output buffer. (In practice, it might provide better performance to make the output buffer larger and the input buffers slightly smaller.)
    Perform a 9-way merge and store the result in the output buffer. Whenever the output buffer fills, write it to the final sorted file and empty it. Whenever any of the 9 input buffers empties, fill it with the next 10 MB of its associated 100 MB sorted chunk until no more data from the chunk is available. This is the key step that makes external merge sort work externally -- because the merge algorithm only makes one pass sequentially through each of the chunks, each chunk does not have to be loaded completely; rather, sequential parts of the chunk can be loaded as needed.

Historically, instead of a sort, sometimes a replacement-selection algorithm[3] was used to perform the initial distribution, to produce on average half as many output chunks of double the length.

-  There will be ceil(log_B-1(ceil(N/B))) passes. Each pass will have 2N I/Os. So O(nlogn). 

### Additional passes

The previous example is a two-pass sort: first sort, then merge. The sort ends with a single k-way merge, rather than a series of two-way merge passes as in a typical in-memory merge sort. This is because each merge pass reads and writes every value from and to disk.

The limitation to single-pass merging is that as the number of chunks increases, memory will be divided into more buffers, so each buffer is smaller. This causes many smaller reads rather than fewer larger ones. Thus, for sorting, say, 50 GB in 100 MB of RAM, using a single merge pass isn't efficient: the disk seeks to fill the input buffers with data from each of the 500 chunks (we read 100MB / 501 ~ 200KB from each chunk at a time) take up most of the sort time. Using two merge passes solves the problem. Then the sorting process might look like this:

    Run the initial chunk-sorting pass as before.
    Run a first merge pass combining 25 chunks at a time, resulting in 20 larger sorted chunks.
    Run a second merge pass to merge the 20 larger sorted chunks.

Like in-memory sorts, efficient external sorts require O(n log n) time: linear increases in data size require logarithmic increases in the number of passes, and each pass takes a linear number of reads and writes. Using the large memory sizes provided by modern computers the logarithmic factor grows very slowly. Under reasonable assumptions at least 500 GB of data can be sorted using 1 GB of main memory before a third pass becomes advantageous, and many times that much data can be sorted before a fourth pass becomes useful.[4] Low-seek-time media like solid-state drives (SSDs) also increase the amount that can be sorted before additional passes improve performance.

Main memory size is important. Doubling memory dedicated to sorting halves the number of chunks and the number of reads per chunk, reducing the number of seeks required by about three-quarters. The ratio of RAM to disk storage on servers often makes it convenient to do huge sorts on a cluster of machines[5] rather than on one machine with multiple passes.


##
There are n number of people in a room. If any two people doesn’t know each other, they shake hand. At the end, ever body announces their number of hand shake.
What is the possibility that ever body’s answer is unique/repetitive. Give proof for your answer.


https://www.springboard.com/blog/data-science-interview-questions/

https://www.edureka.co/blog/interview-questions/data-science-interview-questions/


## 18. What techniques can be used to evaluate a Machine Learning model?

Machine Learning algorithms can be evaluated using various metrics depending on the nature of the problem and the type of model used. Following are some of the techniques to evaluate for regression and classification models respectively:

Regression:

    Mean Absolute Error
    Mean Squared Error
    R square
    Adjusted R square
    Root Mean Squared Logarithmic Error

Classification:

    Classification Accuracy
    Logarithmic Loss
    Precision
    Recall
    F1 Score
    Confusion Matrix
    Receiver Operating Characteristics (ROC) curve
    Area under Curve (AUC)
    Gini coefficient


17. While working at Facebook, you're asked to implement some new features. What type of experiment would you run to implement these features?

A/B testing can be used to check the response on new features by the general audience. A/B testing can be valuable because different audiences behave, well, differently. Something that works for one company may not necessarily work for another. A/B testing is a marketing experiment wherein you "split" your audience to test a number of variations of a campaign/new feature and determine which performs better. For example, in marketing or a web design, you might be comparing two different landing pages with or two different newsletters. Version A shows the layout of a page. Now, you decide to move the content body to the right versus the left. In order for A/B testing to work, you must call out your criteria for success before you begin. What do you think will happen if you change Version A to Version B? Maybe you're hoping to increase newsletter sign ups or decrease the bounce rate. This way you can determine the success rate of both the versions.


## Types of biases:
1. Sampling Bias:
- As ML model highly depend on the data, so if the collected data doesn't reflect the population, it is sampling bias
-  The decision makers have to remember that if humans are involved at any part of the process, there is a greater chance of bias in the model.

 
2. Prejudice Bias
- if someone is intentionally sharing wrong information such as gender, nationality, etc.
- doing analysis of such data can create bias
- very hard to deal with
 
 
3. Confirmation Bias
- Confirmation bias, the tendency to process information by looking for, or interpreting, information that is consistent with one’s existing beliefs.
- If the people of intended use have a pre-existing hypothesis that they would like to confirm with machine learning (there are probably simple ways to do it depending on the context) the people involved in the modelling process might be inclined to intentionally manipulate the process towards finding that answer.
 
4. Group attribution Bias
- This type of bias results from when you train a model with data that contains an asymmetric view of a certain group. 
- For example, in a certain sample dataset if the majority of a certain gender would be more successful than the other or if the majority of a certain race makes more than another, your model will be inclined to believe these falsehoods. There is label bias in these cases. 
- The sample used to understand and analyse the current situation cannot just be used as training data without the appropriate pre-processing to account for any potential unjust bias.

