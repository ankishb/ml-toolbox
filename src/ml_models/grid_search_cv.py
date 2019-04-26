
from sklearn.model_selection import GridSearchCV

def grid_search_cv(model, params, X_train, y_train, cv_fold=3, score_func=None):
	"""return grid_search model instance
	Args:
		model: estimator
		params: parameters grid (dict of variables) as {'depth': [1,2,3], 'itr':[100,200]}
		cv_fold: cv (by default: 3)
		score_func: score function (This will be maximized by the grid_cv)
	"""

	grid_search = GridSearchCV( estimator, param_grid, 
								scoring=score_func, n_jobs=-1, 
								verbose=1, cv=cv_fold)
	grid_search.fit(X_train, y_train)

	return grid_search