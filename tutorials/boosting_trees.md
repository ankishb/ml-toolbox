
## XGBOOST:
Boosted Tree Iteratively fit a new model in the current representation or predicted output as
```
y^0 = 0
y^1 = y^0 + f1(x)
....
y^t = y^(t-1) + ft(x)
```

> ft(x) = (y^t - y^(t-1)) = error

Objective function = loss(y, y^t) + reg_loss

obj(t) = loss(y,y^(t)) + Ω(fi) 
    = loss(y, y^(t−1) + ft(x))^2 + Ω(ft)

If we consider using mean squared error (MSE) as our loss function, the objective becomes
obj(t) = (y −(y^(t−1) + ft(x)))^2 + Ω(f) 
= [2(y^(t−1)−y) ft(x) + ft(x)^2] + Ω(ft) 

In the general case, we take the Taylor expansion of the loss function up to the second order:
obj(t) = [loss(y,y^(t−1)) + grad ft(x) + hessian f^2t(x)] + Ω(ft)

grad = ∂y^(t−1)loss(y,y^(t−1))
hessian = ∂2y^(t−1)loss(y,y^(t−1))


For reg_loss, it use complexity of tree(number of leafs and weights at leaf nodes)
reg_loss = γT+ w^2

Overall Objective function: [grad*weight + (hessian + γ)*weight^2] + γ*T

Optimal weight: -grad/(hessian + γ)
So obejctive: grad^2/(hessian + γ) + γ * T

Where first terms tell how good it the split is or ho good the tree structure is

To split at node:
gain  = [gain of left child + gain of right child - gain, if we don't split] - complexity at that split

where gain: grad^2/(hessian + γ)

- xgboost doesn't handle categorical variable
- use dummy variable/one hot encoding for cat variable

### Unique feature of xgboost
1. regularization
2. handling sparse data
3. cache awareness, to save computation from recomputing gradient
4. parallel learning
5. scalable (used by CERN on petabytes of data)
6. weighted quantile sketch (weight column to each sample row)
7. out of core usuage (optimize the disk space for huge dataset)

---


XGBoost: It need all data in the int/float form
LightGBM: It can handle categorical columns, if dtype is chosen to be category, or if we explicitly provide columns name, shown in following example
Note: LightGBM can not work on string dtypes

```python
lg = lgb.LGBMClassifier(silent=False, categorical_feature=cat_cols)
```
or better
```python
cat_col = train.select_dtypes('object').columns.tolist()

d_train = lgb.Dataset(
    X_train, 
    label = y_train, 
    feature_name = list(X_train.columns), 
    categorical_feature = cat_cols
)
```

To handle missing values, use `use_missing=True`
Use min_data_per_group, cat_smooth to deal with over-fitting (when #data is small or #category is large).

`min_data_per_group` (default = 100)

    minimal number of data per categorical group


`cat_l2` (default = 10.0)

    L2 regularization in categorcial split

`cat_smooth` (default = 10.0)

    this can reduce the effect of noises in categorical features, especially for categories with few data






Catboost:

`grow_policy`:

    `SymmetricTree` —A tree is built level by level until the specified depth is reached. On each iteration, all leaves from the last tree level are split with the same condition. The resulting tree structure is always symmetric.
    `Depthwise` — A tree is built level by level until the specified depth is reached. On each iteration, all non-terminal leaves from the last tree level are split. Each leaf is split by condition with the best loss improvement.
    `Lossguide` — A tree is built leaf by leaf until the specified maximum number of leaves is reached. On each iteration, non-terminal leaf with the best loss improvement is split.


There are some category specific parameters, such as cat_encoding.


Can be used only with the Lossguide and Depthwise growing policies.



## Parameter Tuning

For heavily unbalanced datasets such as 1:10000:

    max_bin: keep it only for memory pressure, not to tune (otherwise overfitting)
    learning rate: keep it only for training speed, not to tune (otherwise overfitting)
    n_estimators: must be infinite (like 9999999) and use early stopping to auto-tune (otherwise overfitting)
    num_leaves: [7, 4095]
    max_depth: [2, 63] and infinite (I personally saw metric performance increases with such 63 depth with small number of leaves on sparse unbalanced datasets)
    scale_pos_weight: [1, 10000] (if over 10000, something might be wrong because I never saw it that good after 5000)
    min_child_weight: [0.01, (sample size / 1000)] if you are using logloss (think about the hessian possible value range before putting "sample size / 1000", it is dataset-dependent and loss-dependent)
    subsample: [0.4, 1]
    bagging_freq: only 1, keep as is (otherwise overfitting)
    colsample_bytree: [0.4, 1]
    is_unbalance: false (make your own weighting with scale_pos_weight)
    USE A CUSTOM METRIC (to reflect reality without weighting, otherwise you have weights inside your metric with premade metrics like xgboost)

Never tune these parameters unless you have an explicit requirement to tune them:

    Learning rate (lower means longer to train but more accurate, higher means smaller to train but less accurate)
    Number of boosting iterations (automatically tuned with early stopping and learning rate)
    Maximum number of bins (RAM dependent)


## LightGBM docs summary:

### For Faster Speed:
1. Use bagging by setting `bagging_fraction` and `bagging_freq`
2. Use feature sub-sampling by setting `feature_fraction`
3. Use small `max_bin`
4. Use `save_binary` to speed up data loading in future learning

### For Better Accuracy:
1. Use large `max_bin` (may be slower)
2. Use small `learning_rate` with large `num_iterations`
3. Use large `num_leaves` (may cause over-fitting)
4. Use bigger training data
5. Try dart

### Deal with Over-fitting:
1. Use small `max_bin`
2. Use small `num_leaves`
3. Use `min_data_in_leaf` and `min_sum_hessian_in_leaf`
4. Use bagging by set `bagging_fraction` and `bagging_freq`
5. Use feature sub-sampling by set `feature_fraction`
6. Use bigger training data
7. Try `lambda_l1`, `lambda_l2` and `min_gain_to_split` for regularization
8. Try `max_depth` to avoid growing deep tree

Reference: https://sites.google.com/view/lauraepp/parameters












Questions from Data Science Interviews at Top Tech Companies (get solved code examples for hands-on experience)

Data Scientist Interview Questions for Top Tech Companies

These questions listed here are after a thorough research of the companies’ sites and high quality discussion forums. This is not a guarantee that these very questions will be asked in data science interviews, but this is just to give the readers an idea of what can be expected when they apply for the position of Data Scientists in these tech companies.

Learn Data Science in Python to Land a Top Gig as a Data Scientist at Top Tech Companies!
Facebook Data Science Interview Questions

1.         A building has 100 floors. Given 2 identical eggs, how can you use them to find the threshold floor? The egg will break from any particular floor above floor N, including floor N itself.

2.         In a given day, how many birthday posts occur on Facebook?

3.         You are at a Casino. You have two dices to play with. You win $10 every time you roll a 5. If you play till you win and then stop, what is the expected pay-out?

4.         How many big Macs does McDonald sell every year in US?

5.         You are about to get on a plane to Seattle, you want to know whether you have to bring an umbrella or not. You call three of your random friends and as each one of them if it’s raining. The probability that your friend is telling the truth is 2/3 and the probability that they are playing a prank on you by lying is 1/3. If all 3 of them tell that it is raining, then what is the probability that it is actually raining in Seattle.

6.         You can roll a dice three times. You will be given $X where X is the highest roll you get. You can choose to stop rolling at any time (example, if you roll a 6 on the first roll, you can stop). What is your expected pay-out?

7.         How can bogus Facebook accounts be detected?

8.       You have been given the data on Facebook user’s friending or defriending each other. How will you determine whether a given pair of Facebook users are friends or not?

9.         How many dentists are there in US?

10.         You have 2 dices. What is the probability of getting at least one 4? Also find out the probability of getting at least one 4 if you have n dices.

11.       Pick up a coin C1 given C1+C2 with probability of trials p (h1. =.7, p (h2. =.6 and doing 10 trials. And what is the probability that the given coin you picked is C1 given you have 7 heads and 3 tails? 

12.     You are given two tables- friend_request and request_accepted. Friend_request contains requester_id, time and sent_to_id and request_accepted table contains time, acceptor_id and requestor_id. How will you determine the overall acceptance rate of requests?

13.       How would add new Facebook members to the database of members, and code their relationships to others in the database? (click here to get solved use-cases + code)

14.       What would you add to Facebook and how would you pitch it and measure its success?

15.  How will you test that there is increased probability of a user to stay active after 6 months given that a user has more friends now?

16. You have two tables-the first table has data about the users and their friends, the second table has data about the users and the pages they have liked. Write an SQL query to make recommendations using pages that your friends liked. The query result should not recommend the pages that have already been liked by a user.

17. What is the probability of pulling a different shape or a different colour card from a deck of 52 cards?

18. Which technique will you use to compare the performance of two back-end engines that generate automatic friend recommendations on Facebook?

19. Implement a sorting algorithm for a numerical dataset in Python.

20. How many people are using Facebook in California at 1.30 PM on Monday?

21. You are given 50 cards with five different colors- 10 Green cards, 10 Red Cards, 10 Orange Cards, 10 Blue cards, and 10 Yellow cards. The cards of each colors are numbered from one to ten. Two cards are picked at random. Find out the probability that the cards picked are not of same number and same color.
 
Get hands-on experience for your interview
with free solved use-cases + code - Click here

Insight Data Science Interview Questions

1.         Which companies participating in Insight would you be interested in working for? 

2.         Create a program in a language of your choice to read a text file with various tweets. The output should be 2 text files-one that contains the list of all unique words among all tweets along with the count for repeated words and the second file should contain the medium number of unique words for all tweets.

3.         What motivates you to transition from academia to data science?
Twitter Data Scientist Interview Questions                       

1.    How can you measure engagement with given Twitter data?

2.    Give a large dataset, find the median.

3.    What is the good measure of influence of a Twitter user?
AirBnB Data Science Interview Questions

1.  Do you have some knowledge of R - analyse a given dataset in R?

2.  What will you do if removing missing values from a dataset cause bias?

3.  How can you reduce bias in a given data set?

4. How will you impute missing information in a dataset?
Google Data Science Interview Questions (click here to get solved use-cases + code)

1.  Explain about string parsing in R language

2. A disc is spinning on a spindle and you don’t know the direction in which way the disc is spinning. You are provided with a set of pins.How will you use the pins to describe in which way the disc is spinning?

3.  Describe the data analysis process.

4. How will you cut a circular cake into 8 equal pieces?
LinkedIn Data Science Interview Questions

1.  Find out K most frequent numbers from a given stream of numbers on the fly.

2.  Given 2 vectors, how will you generate a sorted vector?

3.  Implementing pow function

4.  What kind of product you want to build at LinkedIn?

5.  How will you design a recommendation engine for jobs?

6.  Write a program to segment a long string into a group of valid words using Dictionary. The result should return false if the string cannot be segmented. Also explain about the complexity of the devised solution.

7. Define an algorithm to discover when a person is starting to search for new job.

8. What are the factors used to produce “People You May Know” data product on LinkedIn?

9.  How will you find the second largest element in a Binary Search tree ? (Asked for a Data Scientist Intern job role)

 

Master Machine Learning with interesting machine learning project ideas
 
Mu Sigma Data Science Interview Questions

1.   Explain the difference between Supervised and Unsupervised Learning through examples.

2.   How would you add value to the company through your projects?

3.   Case Study based questions – Cars are implanted with speed tracker so that the insurance companies can track about our driving state. Based on this new scheme what kind of business questions can be answered?

4.  Define standard deviation, mean, mode and median.

5. What is a joke that people say about you and how would you rate the joke on a scale of 1 to 10?

6. You own a clothing enterprise and want to improve your place in the market. How will you do it from the ground level ?

7. How will you customize the menu for Cafe Coffee Day ?

Amazon Data Science Interview Questions

1. Estimate the probability of a disease in a particular city given that the probability of the disease on a national level is low.

2. How will inspect missing data and when are they important for your analysis?

3. How will you decide whether a customer will buy a product today or not given the income of the customer, location where the customer lives, profession and gender? Define a machine learning algorithm for this.

4. From a long sorted list and a short 4 element sorted list, which algorithm will you use to search the long sorted list for 4 elements.

5. How can you compare a neural network that has one layer, one input and output to a logistic regression model?

6. How do you treat colinearity?

7. How will you deal with unbalanced data where the ratio of negative and positive is huge?

8. What is the difference between -

i. Stack and Queue

ii. Linkedin and Array
Uber Data Science Interview Questions (click here to get solved use-cases + code)

1. Will Uber cause city congestion?

2. What are the metrics you will use to track if Uber’s paid advertising strategies to acquire customers work? How will you figure out the acceptable cost of customer acquisition?

3. Explain principal components analysis with equations.

4. Explain about the various time series forecasting technqiues.

5. Which machine learning algorithm will you use to solve a Uber driver accepting  request?

6)How will you compare the results of various machine learning algorithms?

7. How to solve multi-collinearity?

8. How will you design the heatmap for Uber drivers to provide recommendation on where to wait for passengers? How would you approach this?

9. If we added one rider to the current SF market, how would that affect the existing riders and drivers?  

10. What are the different performance metrics for evaluating Uber services?

11. How will you decide which version (Version 1 or Version 2. of the Surge Pricing Algorithms is working better for Uber ?

12. How will you explain JOIN function in SQL to a 10 year old ?

## Netflix Data Science Interview Questions

1. How can you build and test a metric to compare ranked list of TV shows or Movies for two Netflix users?

2. How can you decide if one algorithm is better than the other?

## Microsoft Data Science Interview Questions
1. Write a function to check whether a particular word is a palindrome or not.

2. How can you compute an inverse matrix faster by playing with some computation tricks?

3. You have a bag with 6 marbles. One marble is white.  You reach the bag 100 times. After taking out a marble, it is placed back in the bag. What is the probability of drawing a white marble at least once?

## Apple Data Science Interview Questions
1. How do you take millions of users with 100's of transactions each, amongst 10000's of products and group the users together in a meaningful segments?

## Adobe Data Scientist Interview Questions
1. Check whether a given integer is a palindrome or not without converting it to a string.

2. What is the degree of freedom for lasso?

3. You have two sorted array of integers, write a program to find a number from each array such that the sum of the two numbers is closest to an integer i.

## design a scalable system, which can handle millions of request:
As explained by Talle it have different layers. If not then you have to structure it to different layers.

1. Load Balancer

It is needed if you have so many requests which can not be handled by one web server. Typically 10-15k requests per second can be handled by one web server for a dynamic website, but it depends totally on complexity of website/web application. Load balancer contains multiple web servers and just forwards incoming requests to one of them to distribute.

2. Web Server

Tune the configuration of web server for your use case. Set number of threads, connections, network buffer size, open file descriptor etc. Different servers have different configuration files to tune the performance.

3. MySql/database

Each web server must serve same content, hence should talk to same database. If many web server talking to one db server, it will become bottle neck. Even if there is one web server, sometimes db server may become bottle neck. Many database server have scalable architecture. mysql server supports master-slave and master-master configuration.

    1. In master-slave one server is master where data is written and it is replicated to multiple slave servers. In this case write is done on master and read from slaves. This is useful when very few write happens on database but many reads. typically less than 10-15% of write but depends on the use case.
    2. In master-master is similar to above but all are masters, any data written to any server gets replicated to other servers. Read and write can be done on any server. This is useful for applications which have high writes.

Above is for mysql but similar kind of scalability is supported by other database servers too.

4. Caching Server

Reading from disk(db) is expensive. Here caching server comes to help you. They keep cache data to the memory. If data is there in cache(hit), then disk read is saved, if it is not(miss) then read from disk and save it in cache for next time. If you get 70-80% hit ratio then it will help in scaling.

Keep all of above servers in same LAN close to each other in high speed LAN so that when they communicate, netword doesn't become bottle neck.

5. Separate cdn server

To server static content(js, css, images) setup a cdn server which is optimized to serve static content. This will reduce load from web server.



## American Express Data Scientist Interview Questions
1. Suppose that American Express has 1 million card members along with their transaction details. They also have 10,000 restaurants and 1000 food coupons. Suggest a method which can be used to pass the food coupons to users given that some users have already received the food coupons so far.

2. You are given a training dataset of users that contain their demographic details, the pages on Facebook they have liked so far and results of psychology test  based on their personality i.e. their openness to like FB pages or not. How will you predict the age, gender and other demographics of unseen data?

## Quora Data Scientist Interview Questions
1. How will you test a machine learning model for accuracy?

2. Print the elements of a matrix in zig-zag manner.

3. How will you overcome overfitting in predictive models?

4. Develop an algorithm to sort two lists of sorted integers into a single list.

## Goldman Sachs Data Scientist Interview Questions
1. Count the total number of trees in United States.

2. Estimate the number of square feet pizza’s eaten in US each year.

3. A box has 12 red cards and 12 black cards. Another box has 24 red cards and 24 black cards. You want to draw two cards at random from one of the two boxes, which box has a higher probability of getting cards of same colour and why?

4. How will you prove that the square root of 2 is irrational?

5. What is the probability of getting a HTT combination before getting a TTH combination?

6. There are 8 identical balls and only one of the ball is slightly heavier than the others. You are given a balance scale to find the heavier ball. What is the least number of times you have to use the balance scale to find the heavier ball?

## Walmart Data Science Interview Questions
1. Write the code to reverse a Linked list.

2. What assumptions does linear regression machine learning algorithm make?

3. A stranger uses a search engine to find something and you do not know anything about the person. How will you design an algorithm to determine what the stranger is looking for just after he/she types few characters in the search box?

4. How will you fix multi-colinearity in a regression model?

5. What data structures are available in the Pandas package in Python programming language?

6. State some use cases where Hadoop MapReduce works well and where it does not.

7. What is the difference between an iterator, generator and list comprehension in Python?

8. What is the difference between a bagged model and a boosted model?

9. What do you understand by parametric and non-parametric methods? Explain with examples.

10. Have you used sampling? What are the various types of sampling have you worked with?

11. Explain about cross entropy ?

12. What are the assuptions you make for linear regression ?

13. Differentiate between gradient boosting and random forest.

14. What is the signigicance of log odds ?

## IBM Data Science Interview Questions
1. How will you handle missing data ?

## Yammer Data Science Interview Questions

    How can you solve a problem that has no solution?
    On rolling a dice if you get $1 per dot on the upturned face,what are your expected earnings from rolling a dice?
    In continuation with question #2, if you have 2 chances to roll the dice and you are given the opportunity to decide when to stop rolling the dice (in the first roll or in the second roll). What will be your rolling strategy to get maximum earnings?
     What will be your expected earnings with the two roll strategy?
    You are creating a report for user content uploads every month and observe a sudden increase in the number of upload for the month of November. The increase in uploads is particularly in image uploads. What do you think will be the cause for this and how will you test this sudden spike?

## Citi Bank Data Science Interview Questions
1. A dice is rolled twice, what is the probability that on the second chance it will be a 6?

2. What are Type 1 and Type 2 errors ?

3. Burn two ropes, one needs 60 minutes of time to burn and the other needs 30 minutes of time. How will you achieve this in 45 minutes of time ?

 
Data Science interview coding questions + solution code

Here are some solved data science code snippets that you can use in your interviews or projects. Click on these links below to download the code for these problems. Complete list of ready-to-use solved use-cases is available here. 
How to Flatten a Matrix?
How to Calculate Determinant of a Matrix or ndArray?
How to calculate Diagonal of a Matrix?
How to Calculate Trace of a Matrix?
How to invert a matrix or nArray in Python?
How to convert a dictionary to a matrix or nArray in Python?
How to reshape a Numpy array in Python?
How to select elements from Numpy array in Python?
How to create a sparse Matrix in Python?
How to Create a Vector or Matrix in Python?
How to run a basic RNN model using Pytorch?
How to save and reload a deep learning model in Pytorch?
How to use auto encoder for unsupervised learning models?​
How to create RANDOM Numbers in Python?
How to define WHILE Loop in Python?
How to define FOR Loop in Python?
How to find MIN, MAX in a Dictionary?
How to deal with Dictionary Basics in Python?
How to deal with Date & Time Basics in Python?
How to Create and Delete a file in Python?
How to convert STRING to DateTime in Python?
How to use CONTINUE and BREAK statement within a loop in Python?
How to do numerical operations in Python using Numpy?
 
## Data Science Interview Questions Asked at Other Top Tech Companies
2. Explain the working of a Random Forest Machine Learning Algorithm (Asked at Cyient)

3. Describe K-Means Clustering.(Asked at Symphony Teleca)

4. What is the difference between logistic and linear regression? (Asked at Symphony Teleca)

5. What kind of distribution does logistic regression follow? (Asked at Symphony Teleca)

6. How do you parallelize machine learning algorithms? (Asked at Vodafone. (get interview problems + solution code)
----
7. When required data is not available for analysis, how do you go about collecting it? (Asked at Vodafone)

8. What do you understand by heteroscadisticity (Asked at Vodafone)

9. What do you understand by confidence interval? (Asked at Vodafone)

10. Difference between adjusted r and r square. (Asked at Vodafone)

11. How Facebook recommends items to newsfeed? (Asked at Finomena)

12.  What do you understand by ROC curve and how is it used? (Asked at MachinePulse)

13. How will you identify the top K queries from a file? (Asked at BloomReach)

14. Given a set of webpages and changes on the website, how will you test the new website feature to determine if the change works positively? (Asked at BloomReach)

15. There are N pieces of rope in a bucket. You put your hand into the bucket, take one end piece of the rope .Again you put your hand into the bucket and take another end piece of a rope. You tie both the end pieces together. What is the expected value of the number of loops within the bucket? (Asked at Natera)

16. How will you test if a chosen credit scoring model works or not? What data will you look at? (Asked at Square)

17. There are 10 bottles where each contains coins of 1 gram each. There is one bottle of that contains 1.1 gram coins. How will you identify that bottle after only one measurement? (Data Science Puzzle asked at Latent View Analytics)

18. How will you measure a cylindrical glass filled with water whether it is exactly half filled or not? You cannot measure the water, you cannot measure the height of the glass nor can you dip anything into the glass. (Data Science Puzzle asked at Latent View Analytics)

19. What would you do if you were a traffic sign? (Data Science Interview Question asked at Latent View Analytics)

20.  If you could get the dataset on any topic of interest, irespective of the collection methods or resources then how would the dataset look like and what will you do with it. (Data Scientist Interview Question asked at CKM Advisors)

21. Given n samples from a uniform distribution [0,d], how will you estimate the value of d? (Data Scientist Interview Question asked at Spotify)

22. How will you tune a Random Forest? (Data Science Interview Question asked at Instacart). (get interview problems + solution code)

23. Tell us about a project where you have extracted useful information from a large dataset. Which machine learning algorithm did you use for this and why? (Data Scientist Interview Question asked at Greenplum)

24. What is the difference between Z test and T test ? (Data Scientist Interview Questions asked at Antuit)

25. What are the different models you have used for analysis and what were your inferences? (Data Scientist Interview Questions asked at Cognizant)

26. Given the title of a product, identify the category and sub-category of the product. (Data Scientist interview question asked at Delhivery)

27. What is the difference between machine learning and deep learning? ( Data Scientist Interview Question asked at InfoObjects)

28. What are the different parameters in ARIMA models ? (Data Science Interview Question asked at Morgan Stanley)

29. What are the optimisations you would consider when computing the similarity matrix for a large dataset? (Data Science Interview questions asked at MakeMyTrip)

30. Use Python programming language to implement a toolbox with specific image processing tasks.(Data Science Interview Question asked at Intuitive Surgical)

31. Why do you use Random Forest instead of a simple classifier for one of the classification problems ? (Data Science Interview Question asked at Audi)

32. What is an n-gram? (Data Science Interview Question asked at Yelp)

33. What are the problems related to Overfitting and Underfitting  and how will you deal with these ? (Data Science Interview Question asked at Tiger Analytics)

34. Given a MxN dimension matrix with each cell containing an alphabet, find if a string is contained in it or not.(Data Science Interview Question asked at Tiger Analytics)

35. How do you "Group By" in R programming language without making use of any package ? (Data Scientist Interview Question asked at OLX)

36. List 15 features that you will make use of to build a classifier for OLX website.(Data Scientist Interview Question asked at OLX)

37. How will you build a caching system using an advanced data structure like hashmap ? (Data Scientist Interview Question asked at OLX)

38. How to reverse strings that have changing positions ? (Data Scientist Interview Question asked at Tiger Analytics)

39. How do you select a cricket team ? (Data Scientist Interview Question asked at Quantiphi)

40. What is the difference between trees and random forest ? (Data Scientist Interview Question asked at Salesforce)