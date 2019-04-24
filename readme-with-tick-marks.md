# ml-toolbox

1. Add EDA tool for classification using hue.
	[ ] boxplot
	[ ] kdeplot
	[ ] distplot
	[ ] pairplot
	[ ] multivariate plot
	[ ] heatmap
		https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners

2. Common feature
    [x] Outliers Handling
        - filling with nan
        - removing them
        - fill with lower/upper bound 
        - compute quantile range
    [x] memory optimization
    	<span style="color:red">some **Add category feature dtype** text</span>
    [x] standardization-scaling
    	- min-max
    	- standardization
  
  
    
3. Feature diversity specially for catgorical variable such as 
	[ ] eigen-decomoposition
	[ ] one-hot
	[ ] target-encoding (cv based your and h2o)
        https://maxhalford.github.io/blog/target-encoding-done-the-right-way/
        http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/target-encoding.html
	[ ] bayesian encoding (prior information) 
		https://github.com/MaxHalford/xam/blob/master/docs/feature-extraction.md#smooth-target-encoding
	[ ] quantile binning
	[ ] Label encoder
	[ ] Hash encoder

4. Feature-engineering:
	[ ] intraction based w/wo RF and boosting method
	[ ] date-time based features
	[ ] lag feature(time series) (grouping based on previous day/hour sales or sth)
	[ ] relational (grouping based)
	[ ] Matrix factorization/low-rank
	[ ] social network (networkx)
	[ ] Count-vectorizer
	[ ] Hashing Tricks for highly sparse data(text data)
	[ ] text preprocessing
	[ ] rounding 
	[ ] split decimal value 
		for example sales prices is 899.99, 500.01, this .99, .01 can be feature
	[ ] non-linear dimensionality reduction (PCA, Kernel-PCA, tsne, svd(std scaling after doing this), NMF)
    [ ] outliers handling using matrix factorization




5. Deep learning module:
	[x] Augmentatioan function
		Add example of using augmentation library for segmentation and object detections
		<span style="color:red">some **Add After completing general task** text</span>
	[ ] custom callbacks example
	[ ] data on fly 
	[x] pretrained model (classes as_ in object detection)
	[ ] simple layer in functional form
	[x] learning rate callbacks


 


6. ML-Model:
	[ ] XGBoost/LightGBM/CatBoost
	[ ] ExtraTree/ Adaptive GBM/ Random-Forest
	[ ] Linear model/ Lasso/Ridge/Logistic/SVM
	[ ] KNN(coursera)/tsne-multicore/clustering
	[ ] H2o models with all important parameters specifically with categorical_encoding
		Generalized Linear Model with all target distributions
	[ ] Non negative linear regression (scipy)/ lasso(positive=True)
	[ ] Regularized Random forest
		https://github.com/RGF-team/rgf/tree/master/python-package
		https://github.com/TimSalimans/HiggsML
		https://www.kaggle.com/scirpus/regularized-greedy-forest
		<span style="color:red">some **For FastRFG, we need c++ build system, follow these instruction for that[https://github.com/RGF-team/rgf/tree/master/FastRGF]** text</span>
	[ ] field aware factorization machine


7. Advanced features
    [ ] feature importance toolbox, eli5/shap-value
    [ ] gridsearch / bayesian optimization
    [ ] psuedo labeling
    [ ] object detection
    [ ] subsemble
    [ ] stacking
    [ ] online learning (vowpal rabbit/ follow the regularized leader)

  

8. Special features:
	[ ] parameter tuning
		- hyperopt
		- space (tpe)
		- bayesian optimization
		- gridsearch
		- hyperparameter-hunter
	[ ] parameter tuning for the baseline fit, using gridsearch and less no of samples(50,000).
    [ ] tutorial/example of how to use each of them.
    	- gradient boosting tree
    	- random forest
    	- reguralized greedy forest
    [ ] Parallel processing
    [ ] Dask tutorial for data preprocessing
    [ ] Feature Selection (Recusive feature elimination)
    [ ] Sparse Matrix handling and along with Sparse SVD
    [ ] OOF-analysis(correlation plot and analysis)
    [ ] Error Analysis
    	- statistics of wrong observation as well as correct observation
    	- 
    [ ] CV Vs leaderboard analysis
    [ ] Rank Average




Useful resources:
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




