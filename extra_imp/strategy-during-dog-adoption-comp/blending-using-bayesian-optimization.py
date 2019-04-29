# def blending_hyper_opt(param):
    
#     w1,w2,w3,w4,w5,w6,w7 = param['w1'],param['w2'],param['w3'],param['w4'],param['w5'],param['w6'],param['w7']
#     train_stack = train_stack1.values
#     test_stack = test_stack1.values
    
#     blended = np.dot(train_stack,np.array([w1, w2, w3, w4, w5, w6, w7]))
#     (np.array([1,2,3,4,5,6,7])*train_stack)
#     optR = OptimizedRounder()
#     optR.fit(blended, X_train['AdoptionSpeed'].values)
#     coefficients = optR.coefficients()
#     valid_pred = optR.predict(blended, coefficients)
#     qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
    
#     return -qwk








# from hyperopt import tpe, fmin, Trials
# # Create the algorithm
# tpe_algo = tpe.suggest
# tpe_trials = Trials()

# blend_space = {
#     'w1': hp.uniform('w1', 0.0, 1),
#     'w2': hp.uniform('w2', 0.0, 1),
#     'w3': hp.uniform('w3', 0.0, 1),
#     'w4': hp.uniform('w4', 0.0, 1),
#     'w5': hp.uniform('w5', 0.0, 1),
#     'w6': hp.uniform('w6', 0.0, 1),
#     'w7': hp.uniform('w7', 0.0, 1),
# }

# # Run 2000 evals with the tpe algorithm
# tpe_best = fmin(fn=blending_hyper_opt, space=blend_space, 
#                 algo=tpe_algo, trials=tpe_trials, 
#                 max_evals=2000)

# print(tpe_best)
# # # Dataframe of results from optimization
# # tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results], 
# #                             'iteration': tpe_trials.idxs_vals[0]['x'],
# #                             'x': tpe_trials.idxs_vals[1]['x']})
                            
# # tpe_results.head()








import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import roc_auc_score

from bayes_opt import BayesianOptimization

df_train = pd.read_csv('../input/hcdr-5-prediction-for-train-set/5_predictions.csv', index_col = 'SK_ID_CURR')
df_train.head()


for c in df_train.columns.drop('TARGET'):
    print(c, roc_auc_score(df_train['TARGET'], df_train[c]))

def ROC_evaluate(**params):
    warnings.simplefilter('ignore')
    
    s = sum(params.values())
    for p in params:
        params[p] = params[p] / s
    
    test_pred_proba = pd.Series(np.zeros(df_train.shape[0]), index = df_train.index)
    
    feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR', 'index']]
    
    for f in feats:
        test_pred_proba += df_train[f] * params[f]
    
    return roc_auc_score(df_train['TARGET'], test_pred_proba)

params = {}
for c in df_train.columns.drop('TARGET'):
    params[c] = (0, 1)
    
bo = BayesianOptimization(ROC_evaluate, params)
bo.maximize(init_points = 50, n_iter = 10)

