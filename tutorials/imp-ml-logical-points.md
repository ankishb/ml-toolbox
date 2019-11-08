
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