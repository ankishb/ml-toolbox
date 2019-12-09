# ml-toolbox
This repo contains various data science strategy and machine learning models to deal with structure as well as unstructured data. It contains module on feature-preprocessing, feature-engineering, machine-learning-models, bayesian-parameter-tuning, etc. Some of these features are collected from the existed libraries such as scikit-learn, keras, h2o, xgboost, lightgbm, catboost, etc. I have also added some technique, which I implemented by following the Research Paper and Data-Scientist advice(on kaggle). There are a lot of feature engineering strategy as well, which I developed during ML-contest and helped me a lot in those contest.

> I use this toolbox for my personal usuage. I consistently update it, to make more generic.


## Contents:
1. [Feature preprocessing](https://github.com/ankishb/ml-toolbox/tree/master/src/feature_eng)
    + cleaning
    + handling null value
    + normalization
    + grouping unknow variable
    + memory optimization
    + text preprocessing 
2. [Feature Engineering](https://github.com/ankishb/ml-toolbox/tree/master/src/feature_eng)
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
3. [EDA](https://github.com/ankishb/ml-toolbox/tree/master/src/eda_tool)
    + boxplot/kdeplot/countplot/pairplot
    + heatmap
4. [Machine Learning models](https://github.com/ankishb/ml-toolbox/tree/master/src/ml_models)
    + Tree base model
        - xgboost/lighgbm/catboost
        - sklearn: decision-tree/random-forest/extra-tree/GBM
    + Linear/Non-Linear Model:
        - Logistic-Regression/lasso/Ridge/Passive-Agreesive/SVM
    + Regularized Greedy forest(in progress)
    + field aware factorization machine
    + online learning(vowpal rabbit/follow the regularized leader)(in progress)
    + h2o models (gbm/rf/nn/auto-ml)
5. [Deep learning models](https://github.com/ankishb/ml-toolbox/tree/master/src/deep_ml)
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

