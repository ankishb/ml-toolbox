# ml-toolbox
It contains detailed infomation of feature of this library.

## Add EDA tool for classification using hue.
- [ ] boxplot
- [ ] kdeplot
- [ ] distplot
- [ ] pairplot
- [ ] multivariate plot
- [ ] heatmap
    ([ref-20-plots](https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners))

<span style="color: red">use style **plt.style.use('fivethirtyeight')**</span>

---

## Common feature
- [x] Outliers Handling
    - filling with nan
    - removing them
    - fill with lower/upper bound 
    - compute quantile range
- [x] memory optimization 
    1. **Add parallel processing**
    2. **Add category feature dtype**
- [x] standardization-scaling
    - min-max
    - standardization
 
---

## Feature diversity specially for catgorical variable such as 
<span style="color:red">some **Remember: Handle nan carefully while label encoding, for float, it will treat all nan as same, whereas for int, it will consider all nan as different value.** text</span>
- [x] eigen-decomoposition
- [x] one-hot
- [x] target-encoding
    ([lstf-av](https://github.com/mohsinkhn/ltfs-av/blob/dev/TargetEncoder.py))
    ([kaggle](https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study?utm_medium=email&utm_source=mailchimp&utm_campaign=datanotes-20181004))
    ([MaxHalford](https://maxhalford.github.io/blog/target-encoding-done-the-right-way/))
    ([kaggle](https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features))
    ([kaggle-most-imp](https://www.kaggle.com/scirpus/hybrid-jeepy-and-lgb (most impotant))
- [x] bayesian encoding (prior information)
    ([maxhalford-git](https://github.com/MaxHalford/xam/blob/master/docs/feature-extraction.md#smooth-target-encoding))
- [ ] quantile binning
- [x] Label encoder
- [x] Hash encoder
- [x] Binary encoding

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
    - Hashing Tricks for highly sparse data(text data)
<span style="color:red">some **Will add extensive preprocessing at last.** text</span>
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
- [ ] discriminant_analysis
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
    ([gini-kaggle](https://www.kaggle.com/rspadim/gini-keras-callback-earlystopping-validation))
- [x] pretrained model (classes as_ in object detection)
- [x] simple layer in functional form
- [x] learning rate callbacks
- [x] keras metrics/losses/activation-func/optimizers list
- [x] pretrained Word2Vec embedding for nlp task (quora question answer)
- [x] LSTM usuage
- [x] word2vec from gensim for category variables. (Entity embedding)
    <span style="color:red">some **Add example of cuisine prediction using phase2vec model built using word2vec from gensim (kaggle dataset).** text</span>
    ([ltfs-av](https://github.com/mohsinkhn/ltfs-av))
    ([blog](http://kavita-ganesan.com/how-to-incorporate-phrases-into-word2vec-a-text-mining-approach/#.WuiiKtMvyds))
- [x] data on fly 
- [x] Segmentation (U-net)

---

## ML-Model:
- [x] XGBoost/LightGBM/CatBoost
- [x] ExtraTree/ Adaptive GBM/ Random-Forest
- [x] Linear model/ Lasso/Ridge/Logistic/SVM
    <span style="color:red">some **For large dataset, sunning svm is not wise, there is an online approximation of SVM solver, please follow following link.** text</span>
    ([sklearn-metric](https://scikit-learn.org/stable/modules/metrics.html#metrics (kernel function)))
    ([blog](https://leon.bottou.org/projects/lasvm))
- [ ] KNN(coursera)/tsne-multicore/clustering
- [x] H2o models 
- [ ] Non negative linear regression (scipy)/ lasso(positive=True)
- [ ] Regularized Greedy forest
    ([rgf official](https://github.com/RGF-team/rgf/tree/master/python-package))
    ([HiggsML exp](https://github.com/TimSalimans/HiggsML))
    ([kaggle](https://www.kaggle.com/scirpus/regularized-greedy-forest))
    <span style="color:red">some **For FastRFG, we need c++ build system, follow these instruction for that[link](https://github.com/RGF-team/rgf/tree/master/FastRGF)** text</span>
- [ ] field aware factorization machine
    ([kaggle](https://www.kaggle.com/scirpus/kernels))
- [ ] Matrix factorization/low-rank
- [ ] social network (networkx)
- [ ] FTRL Proximal Model
    ([link1](https://www.kaggle.com/ogrellier/multi-process-ftrl))
    ([link2](https://www.kaggle.com/supernova117/ftrl-with-validation-and-auc))
- [x] Naive Bayes Algorithm

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
    ([bert-kaggle](https://www.kaggle.com/christofhenkel/bert-embeddings-lstm/data))
    ([ulmafit-kaggle](https://www.kaggle.com/christofhenkel/ulmfit-fast-ai-starter))
- [x] Set random seed for reproducible features
- [ ] Adverserial validation example
- [ ] Relation Data based Feature enginnering (featuretool)
    ([feature-tool](https://medium.com/@rrfd/simple-automatic-feature-engineering-using-featuretools-in-python-for-classification-b1308040e183))
- [ ] Text features such as no of words, no of characters etc
- [ ] Category encoding for time series data
- [ ] FeatureAgglomeration
- [ ] Random Forest embedding
- [ ] Feature transformation as [auto-sklearn](https://ml.informatik.uni-freiburg.de/papers/15-NIPS-auto-sklearn-supplementary.pdf)

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
    ([sklearn-estimator](https://www.slideshare.net/PyData/julie-michelman-pandas-pipelines-and-custom-transformers))
- [ ] Image based feature such as quality, canny, etc
    ([imgae-feature-kaggle](https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality))
- [ ] Automatic Feature Engineering
    ([automatic-feature-eng](https://www.kaggle.com/willkoehrsen/kernels))
- [ ] Bortua Feature Selection
    ([bortua-oliver](https://www.kaggle.com/ogrellier/noise-analysis-of-porto-seguro-s-features))
    ([bortua-tilli](https://www.kaggle.com/tilii7/boruta-feature-elimination))
- [ ] Univariate feature selection

---

# Experiments:
1. https://www.kaggle.com/ogrellier/scale-pos-weight-vs-duplication
2. https://ml.informatik.uni-freiburg.de/papers/15-NIPS-auto-sklearn-supplementary.pdf


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
13. https://www.youtube.com/watch?v=TJU8NfDdqNQ (ml tutorials)
14. https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial (many references and ebooks for ml)
15. http://www.chioka.in/kaggle-competition-solutions/ (kaggle solutions)
16. http://ndres.me/kaggle-past-solutions/ (kaggle prev solutions)
17. https://www.kaggle.com/shivamb/data-science-glossary-on-kaggle (best resources)
18. https://stats.idre.ucla.edu/spss/faq/coding-systems-for-categorical-variables-in-regression-analysis-2/#DEVIATION%20EFFECT%20CODING (cat-enc)
19. https://github.com/flennerhag/mlens/tree/master/mlens (higher level API for ensemble, superlearner, subsemble and advance fetures...)
20. https://www.automl.org/book/ (All important topics)
21. https://www.cs.ubc.ca/~nando/540-2013/lectures.html (nandi lectures)
22. 