# ml-toolbox

## Add EDA tool for classification using hue.
- [ ] boxplot
- [ ] kdeplot
- [ ] distplot
- [ ] pairplot
- [ ] multivariate plot
- [ ] heatmap
- [ ] 
    https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners
    plt.style.use('fivethirtyeight')
---

## Common feature
- [x] Outliers Handling  
        - filling with nan
        - removing them
        - fill with lower/upper bound 
        - compute quantile range
- [x] memory optimization
    <span style="color:red">some **Add category feature dtype** text</span>
- [x] standardization-scaling
        - min-max
        - standardization
- [ ]  
  
---

## Feature diversity specially for catgorical variable such as 
    <span style="color:red">some **Remember: Handle nan carefully while label encoding, for float, it will treat all nan as same, whereas for int, it will consider all nan as different value.** text</span>
- [ ] eigen-decomoposition
- [ ] one-hot
- [ ] target-encoding (cv based your and h2o)
    https://github.com/mohsinkhn/ltfs-av/blob/dev/TargetEncoder.py
    https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study?utm_medium=email&utm_source=mailchimp&utm_campaign=datanotes-20181004
    https://maxhalford.github.io/blog/target-encoding-done-the-right-way/
    http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/target-encoding.html
    https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features

    https://www.kaggle.com/scirpus/hybrid-jeepy-and-lgb (most impotant)
- [ ] bayesian encoding (prior information) 
        https://github.com/MaxHalford/xam/blob/master/docs/feature-extraction.md#smooth-target-encoding
- [ ] quantile binning
- [ ] Label encoder
- [ ] Hash encoder
- [ ] Binary encoding
    https://stats.idre.ucla.edu/spss/faq/coding-systems-for-categorical-variables-in-regression-analysis-2/#DEVIATION%20EFFECT%20CODING
- [ ] 

---

## Feature-engineering:
- [ ] intraction based w/wo RF and boosting method
- [ ] intraction based on time-series statistics
- [ ] use tree terminal leaf prediction as the feature for next tree.
    <span style="color:red">some **check out tree.apply() method to collect tree terminal leaf node output.** text</span>
- [ ] date-time based features
- [ ] lag feature(time series) (grouping using previous day/hour sales or sth)
- [x] text preprocessing
    - Count-vectorizer
    - extensive text processing with stemmer and lemmantizer
    <span style="color:red">some **Will add extensive preprocessing at last.** text</span>
    - Hashing Tricks for highly sparse data(text data)
- [ ] rounding/split decimal value 
    for example sales prices is 899.99, 500.01, this .99, .01 can be feature
- [x] linear/non-linear dimensionality reduction (PCA, Kernel-PCA, tsne, svd(std scaling after doing this), NMF)
- [x] polynomial features
- [x] svd/pca/nmf transformation
- [ ] ica/fastica/factor analysis/ppca transformation
- [ ] isomap/lle/spectral features
- [ ] Quantile/Robust scaler feature transormation
- [ ] outliers handling using matrix factorization
- [ ] numerical to catgorical tranformation
- [ ] 

---
 
<!-- 
[click on this link](#my-multi-word-header)
### My Multi Word Header -->
<!-- [just](#like-this-one) -->

## Deep learning module:
- [x] Augmentatioan function
    Add example of using augmentation library for segmentation and object detections
    <span style="color:red">some **Add After completing general task** text</span>
    <span style="color:blue">some **Another example can be seen in my all-in-one-place/keras git repo.** text</span>
- [x] custom callbacks example
    https://www.kaggle.com/rspadim/gini-keras-callback-earlystopping-validation
- [x] pretrained model (classes as_ in object detection)
- [x] simple layer in functional form
- [x] learning rate callbacks
- [x] keras metrics/losses/activation-func/optimizers list
    https://github.com/mohsinkhn/ltfs-av/blob/dev/TargetEncoder.py
- [x] pretrained Word2Vec embedding for nlp task (quora question answer)
- [x] LSTM usuage
- [x] word2vec from gensim for category variables. (Entity embedding)
    <span style="color:red">some **Add example of cuisine prediction using phase2vec model built using word2vec from gensim (kaggle dataset).** text</span>
    http://kavita-ganesan.com/how-to-incorporate-phrases-into-word2vec-a-text-mining-approach/#.WuiiKtMvyds
- [ ] data on fly 
- [ ] Segmentation (U-net)
    https://github.com/mohsinkhn/ltfs-av
- [ ] 

---

## Little more on Feature engineering
-[ ] Text features such as no of words, no of characters etc


---

7. ML-Model:
- [x] XGBoost/LightGBM/CatBoost
- [ ] ExtraTree/ Adaptive GBM/ Random-Forest
- [ ] Linear model/ Lasso/Ridge/Logistic/SVM
    <span style="color:red">some **For large dataset, sunning svm is not wise, there is an online approximation of SVM solver, please follow following link.** text</span>
    https://scikit-learn.org/stable/modules/metrics.html#metrics (kernel function)
    https://leon.bottou.org/projects/lasvm
- [ ] KNN(coursera)/tsne-multicore/clustering
- [ ] H2o models with all important parameters specifically with categorical_encoding 
    Generalized Linear Model with all target distributions
- [ ] Non negative linear regression (scipy)/ lasso(positive=True)
- [ ] Regularized Random forest
    https://github.com/RGF-team/rgf/tree/master/python-package
    https://github.com/TimSalimans/HiggsML
    https://www.kaggle.com/scirpus/regularized-greedy-forest
    <span style="color:red">some **For FastRFG, we need c++ build system, follow these instruction for that[https://github.com/RGF-team/rgf/tree/master/FastRGF]** text</span>
- [ ] field aware factorization machine
    https://www.kaggle.com/scirpus/kernels
- [ ] Matrix factorization/low-rank
- [ ] social network (networkx)
- [ ] FTRL Proximal Model
    https://www.kaggle.com/ogrellier/multi-process-ftrl
    https://www.kaggle.com/supernova117/ftrl-with-validation-and-auc
    
---

## Advanced features
- [ ] feature importance toolbox, eli5/shap-value
- [x] gridsearch / bayesian optimization
- [ ] psuedo labeling
- [ ] object detection
- [ ] subsemble
- [ ] stacking
- [ ] online learning (vowpal rabbit/ follow the regularized leader)
- [ ] NLP Transform ULMAFit/ELMO/Bert
    https://www.kaggle.com/christofhenkel/bert-embeddings-lstm/data
    https://www.kaggle.com/christofhenkel/ulmfit-fast-ai-starter
- [x] Set random seed for reproducible features
- [ ] Adverserial validation example
- [ ] Relation Data based Feature enginnering (featuretool)
    https://medium.com/@rrfd/simple-automatic-feature-engineering-using-featuretools-in-python-for-classification-b1308040e183

---

## Special features
- [x] parameter tuning
    <span style="color:red">some **Another tuning method: hyperparameter-hunter, will visit at last.** text</span>
- [ ] parameter tuning for the baseline fit, using gridsearch and less no of samples(50,000).
- [ ] Parallel processing (using dask and numba)
- [ ] Feature Selection (Recusive feature elimination)
- [ ] Sparse Matrix handling and along with Sparse SVD
- [ ] OOF-analysis(correlation plot and analysis)
- [ ] Error Analysis
    - statistics of wrong observation as well as correct observation
- [ ] CV Vs leaderboard analysis
- [ ] Rank Average
- [ ] sklearn estimator class
    https://www.slideshare.net/PyData/julie-michelman-pandas-pipelines-and-custom-transformers
- [ ] Image based feature such as quality, canny, etc
    https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality
- [ ] Automatic Feature Engineering
    https://www.kaggle.com/willkoehrsen/kernels

---

# Experiments:
- [ ] https://www.kaggle.com/ogrellier/scale-pos-weight-vs-duplication


### Useful resources:
1. https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/select-features
2. https://forums.aws.amazon.com/message.jspa?messageID=774050
3. https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.nnls.html
4. http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/categorical_encoding.html
5. http://h2o-release.s3.amazonaws.com/h2o/master/3484/docs-website/h2o-py/docs/intro.html
6. https://christophm.github.io/interpretable-ml-book/extend-lm.html (book)
7. https://maxhalford.github.io/blog/streaming-groupbys-in-pandas-for-big-datasets/
8. https://web.stanford.edu/~hastie/ElemStatLearn/
9. https://www.dummies.com/programming/big-data/data-science/data-science-how-to-create-interactions-between-variables-with-python/
10. https://github.com/diefimov/MTH594_MachineLearning/blob/master/ipython/Lecture1.ipynb
11. https://stats.idre.ucla.edu/spss/faq/coding-systems-for-categorical-variables-in-regression-analysis-2/#DEVIATION%20EFFECT%20CODING (different way of encoding)
12. https://scikit-learn.org/stable/modules/classes.html (sklearn complete classes/functions list)
13. https://www.youtube.com/watch?v=TJU8NfDdqNQ (ml tutorials)
14. https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial (many references and ebooks for ml)
15. http://www.chioka.in/kaggle-competition-solutions/ (kaggle solutions)
16. http://ndres.me/kaggle-past-solutions/ (kaggle prev solutions)
17. https://www.kaggle.com/shivamb/data-science-glossary-on-kaggle (best resources)


