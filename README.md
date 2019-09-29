# ml-toolbox
This repo contains a few data-science relative features to deal with structure and unstructure data. It contains feature-preprocessing, feature-engineering, machine learning models, bayesian-parameter-tuning and much more. This is my collection and implementations using existed libraries such as scikit-learn, keras,xgboost,h2o,etc. I have included some strategy for feature engineering on my understanding, following tutorial, Data-science practitioner and of-course reading research paper. I consistently update it, to make more generic as i practice(in hackhathons).

> In my free time, i work on this project, and try to improve it for my own good. Any feedback will be appreciated.

## Contents:
1. Feature preprocessing
    + cleaning
    + handling null value
    + normalization
    + grouping unknow variable
    + memory optimization
    + text preprocessing 
2. Feature Engineering
    + label-encoder/one-hot/binary/hashing
    + binning/quantile-binning
    + target-encoding
    + bayesian-encoding
    + feature-interaction
    + date-time feature
    + time-lag featue(in time series)
    + rounding/decimal value
    + relation feature(aggregation based)
    + text feature using tf-idf, count-vect
    + clustering based feature(linear/non-linear)
    + polynomial feature
    + statistical ferature
3. EDA
    + boxplot/kdeplot/countplot/pairplot
    + heatmap
4. Machine Learning models
    + Tree base model
        - xgboost/lighgbm/catboost
        - sklearn: decision-tree/random-forest/extra-tree/GBM
    + Linear/Non-Linear Model:
        - Logistic-Regression/lasso/Ridge/Passive-Agreesive/SVM
    + Regularized Greedy forest(in progress)
    + field aware factorization machine
    + online learning(vowpal rabbit/follow the regularized leader)(in progress)
    + h2o models (gbm/rf/nn/auto-ml)
5. Deep learning models
    + neural networks(keras/tensorflow)
    + Attention mechanism for LSTM
    + Data augmentation(for image)
    + cyclic-learning-rate 
    + keras custom loss-function/metric/callbacks
    + pretrained model
    + word2vec usuage using gensim
    + entity embedding
    + data on fly(efficient training)
    + segmentation
    + graph based neural network
5. Advances
    + gridsearch / bayesian optimization
    + Stacking/blending/rank-average
    + Bert-Model(pretrained text model)
    + parallel-processing for feature-engineering(in progress)