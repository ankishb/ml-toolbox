
def get_all_functioan_details():
	"""
	1. lgb cv function
	

	"""

@small_lgb_function
def get_score(df, usecols, params, dropcols=[]):  
     dtrain = lgb.Dataset(df[usecols].drop(dropcols, axis=1), df['TARGET'])
     eval = lgb.cv(params,
             dtrain,
             nfold=5,
             stratified=True,
             num_boost_round=20000,
             early_stopping_rounds=200,
             verbose_eval=100,
             seed = 5,
             show_stdv=True)
     return max(eval['auc-mean'])
	 
