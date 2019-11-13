
2. Typical Use Cases

    Ridge: It is majorly used to prevent overfitting. Since it includes all the features, it is not very useful in case of exorbitantly high #features, say in millions, as it will pose computational challenges.
    Lasso: Since it provides sparse solutions, it is generally the model of choice (or some variant of this concept) for modelling cases where the #features are in millions or more. In such a case, getting a sparse solution is of great computational advantage as the features with zero coefficients can simply be ignored.

Its not hard to see why the stepwise selection techniques become practically very cumbersome to implement in high dimensionality cases. Thus, lasso provides a significant advantage.

 
3. Presence of Highly Correlated Features

    Ridge: It generally works well even in presence of highly correlated features as it will include all of them in the model but the coefficients will be distributed among them depending on the correlation.
    Lasso: It arbitrarily selects any one feature among the highly correlated ones and reduced the coefficients of the rest to zero. Also, the chosen variable changes randomly with change in model parameters. This generally doesn’t work that well as compared to ridge regression.

This disadvantage of lasso can be observed in the example we discussed above. Since we used a polynomial regression, the variables were highly correlated. ( Not sure why? Check the output of data.corr() ). Thus, we saw that even small values of alpha were giving significant sparsity (i.e. high #coefficients as zero).










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


Data Wrangling in Practice: What to Expect

There are typically 5 iterative steps that make up the data wrangling process.
1. Discovering: Before you can dive deeply, you must better understand what is in your data, which will inform how you want to analyze it. How you wrangle customer data, for example, may be informed by where they are located, what they bought, or what promotions they received.
2. Structuring: This means organizing the data, which is necessary because raw data comes in many different shapes and sizes. A single column may turn into several rows for easier analysis. One column may become two. Movement of data is made for easier computation and analysis.
3. Cleaning: What happens when errors and outliers skew your data?  You clean the data. What happens when state data is entered as CA or California or Calif.? You clean the data. Null values are changed and standard formatting implemented, ultimately increasing data quality.
4. Enriching: Here you take stock in your data and strategize about how other additional data might augment it. Questions asked during this data wrangling step might be: what new types of data can I derive from what I already have or what other information would better inform my decision making about this current data?
5. Validating: Validation rules are repetitive programming sequences that verify data consistency, quality, and security. Examples of validation include ensuring uniform distribution of attributes that should be distributed normally (e.g. birth dates) or confirming accuracy of fields through a check across data.


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
4. Improvement of features if needed
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

 
### Cosine similarity: Let’s imagine that you need to determine how similar two documents or corpus of text are. Which distance metrics will you use?

The answer is cosine similarity.

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
