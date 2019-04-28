
# 10th place beautiful; solution with codes
https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88995#latest-513787


Take-away

1. Qwk Optimization on the ranking of prediction, instead of the direct prediction
2. Make diverse set of model with diverse set of features(there should not be much
	of correlation.)
3. Try word2vec or doc2vec
4. Try ratio-features
5. Feature selection tools(recursive feature selection,partial plot, )


1. Try log, power, ratio, sum, diff etc based feature
2. Use PCA to transform the space to another space
3. Use Hash code on catgorical data
4. logloss clipping from [0,1] ==> [0.05,0.95], this way, it always learn sth.
5. df.to_csv(‘submission.csv.gz’, index=False, compression=‘gzip’)






==> Aggregate relational features such as:
	
	Sum, Mean, Std, Max, Min, Count, Num_unique, Mode

==> For catboost
categorical_features = np.where(feature_matrix.dtypes == 'object')[0]
for i in categorical_features:
    feature_matrix.iloc[:,i] = feature_matrix.iloc[:,i].astype('str')



Identify Highly Correlated Features

# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

Drop Marked Features

# Drop features 
df.drop(df.columns[to_drop], axis=1)





https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.08-Aggregation-and-Grouping.ipynb


Aggregation 	        Description
count() 	        Total number of items
first(), last() 	First and last item
mean(), median() 	Mean and median
min(), max() 	    Minimum and maximum
std(), var() 	    Standard deviation and variance
mad() 	            Mean absolute deviation
prod() 	            Product of all items
sum() 	            Sum of all items




df.groupby('key').aggregate(['min', np.median, max])

df.groupby('key').aggregate({'data1': 'min',
                             'data2': 'max'})







###################################################################
###################################################################
###################### Feature Engineering ########################

# Separating the member_id column of test dataframe to help create a csv after predictions
test_member_id = pd.DataFrame(dfTest['member_id'])


# Creating target variable pandas series from train dataframe, this will be used by cross validation to calculate
# the accuracy of the model
train_target = pd.DataFrame(dfTrain['loan_status'])


# It's good to create a copy of train and test dataframes. this way we can play around different features as we tune the
# performance of the classifier with important features
selected_cols = ['member_id', 'emp_length', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'int_rate', 'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'total_rec_late_fee', 'mths_since_last_major_derog', 'last_week_pay', 'tot_cur_bal', 'total_rev_hi_lim', 'tot_coll_amt', 'recoveries', 'collection_recovery_fee', 'term', 'acc_now_delinq', 'collections_12_mths_ex_med']
finalTrain = dfTrain[selected_cols]
finalTest = dfTest[selected_cols]

# How big the loan a person has taken with respect to his earnings, annual income to loan amount ratio
finalTrain['loan_to_income'] = finalTrain['annual_inc']/finalTrain['funded_amnt_inv']
finalTest['loan_to_income'] = finalTest['annual_inc']/finalTest['funded_amnt_inv']


# All these attributes indicate that the repayment was not all hunky-dory. All the amounts caclulated are ratios 
# like, recovery to the loan amount. This column gives a magnitude of how much the repayment has gone off course 
# in terms of ratios.
finalTrain['bad_state'] = finalTrain['acc_now_delinq'] + (finalTrain['total_rec_late_fee']/finalTrain['funded_amnt_inv']) + (finalTrain['recoveries']/finalTrain['funded_amnt_inv']) + (finalTrain['collection_recovery_fee']/finalTrain['funded_amnt_inv']) + (finalTrain['collections_12_mths_ex_med']/finalTrain['funded_amnt_inv'])
finalTest['bad_state'] = finalTest['acc_now_delinq'] + (finalTest['total_rec_late_fee']/finalTest['funded_amnt_inv']) + (finalTest['recoveries']/finalTest['funded_amnt_inv']) + (finalTest['collection_recovery_fee']/finalTest['funded_amnt_inv']) + (finalTrain['collections_12_mths_ex_med']/finalTest['funded_amnt_inv'])

# For the sake of this model, I have used just a boolean flag if things had gone bad, with this case I didn't see
# a benifit of including above computations
finalTrain.loc[finalTrain['bad_state'] > 0, 'bad_state'] = 1
finalTest.loc[finalTest['bad_state'] > 0, 'bad_state'] = 1


# Total number of available/unused 'credit lines'
finalTrain['avl_lines'] = finalTrain['total_acc'] - finalTrain['open_acc']
finalTest['avl_lines'] = finalTest['total_acc'] - finalTest['open_acc']


# Interest paid so far
finalTrain['int_paid'] = finalTrain['total_rec_int'] + finalTrain['total_rec_late_fee']
finalTest['int_paid'] = finalTest['total_rec_int'] + finalTest['total_rec_late_fee']


# Calculating EMIs paid (in terms of percent)
finalTrain['emi_paid_progress_perc'] = ((finalTrain['last_week_pay']/(finalTrain['term']/12*52+1))*100)
finalTest['emi_paid_progress_perc'] = ((finalTest['last_week_pay']/(finalTest['term']/12*52+1))*100)


# Calculating total repayments received so far, in terms of EMI or recoveries after charge off
finalTrain['total_repayment_progress'] = ((finalTrain['last_week_pay']/(finalTrain['term']/12*52+1))*100) + ((finalTrain['recoveries']/finalTrain['funded_amnt_inv']) * 100)
finalTest['total_repayment_progress'] = ((finalTest['last_week_pay']/(finalTest['term']/12*52+1))*100) + ((finalTest['recoveries']/finalTest['funded_amnt_inv']) * 100)


'''
Split data set into train-test-cv
Train model & predict
'''
# Split train and cross validation sets
X_train, X_test, y_train, y_test = train_test_split(np.array(finalTrain), np.array(train_target), test_size=0.30)
eval_set=[(X_test, y_test)]

In [14]:

print('Initializing xgboost.sklearn.XGBClassifier and starting training...')

st = datetime.now()

clf = xgboost.sklearn.XGBClassifier(
    objective="binary:logistic", 
    learning_rate=0.05, 
    seed=9616, 
    max_depth=20, 
    gamma=10, 
    n_estimators=500)

clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=eval_set, verbose=True)

print(datetime.now()-st)

y_pred = clf.predict(X_test)
submission_file_name = 'Submission_'

accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
print("Accuracy: %.10f%%" % (accuracy * 100.0))
submission_file_name = submission_file_name + ("_Accuracy_%.6f" % (accuracy * 100)) + '_'

accuracy_per_roc_auc = roc_auc_score(np.array(y_test).flatten(), y_pred)
print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))
submission_file_name = submission_file_name + ("_ROC-AUC_%.6f" % (accuracy_per_roc_auc * 100))

final_pred = pd.DataFrame(clf.predict_proba(np.array(finalTest)))
dfSub = pd.concat([test_member_id, final_pred.ix[:, 1:2]], axis=1)
dfSub.rename(columns={1:'loan_status'}, inplace=True)
dfSub.to_csv((('%s.csv') % (submission_file_name)), index=False)

import matplotlib.pyplot as plt
print(clf.feature_importances_)
idx = 0
for x in list(finalTrain):
    print('%d %s' % (idx, x))
    idx = idx + 1
plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
plt.show()

###################################################################
###################################################################
###################################################################












###################################################################
###################################################################
########################## VISUALIZATION ##########################
# make general plots to examine each feature
def plot_var(col_name, full_name, continuous):
    """
    Visualize a variable with/without faceting on the loan status.
    - col_name is the variable name in the dataframe
    - full_name is the full variable name
    - continuous is True for continuous variables
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, figsize=(15,3))
    # plot1: counts distribution of the variable
    
    if continuous:  
        sns.distplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(df[col_name], order=sorted(df[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(full_name)
    ax1.set_ylabel('Count')
    ax1.set_title(full_name)

          
    # plot2: bar plot of the variable grouped by loan_status
    if continuous:
        sns.boxplot(x=col_name, y='loan_status', data=df, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(full_name + ' by Loan Status')
    else:
        Charged_Off_rates = df.groupby(col_name)['loan_status'].value_counts(normalize=True)[:,'Charged Off']
        sns.barplot(x=Charged_Off_rates.index, y=Charged_Off_rates.values, color='#5975A4', saturation=1, ax=ax2)
        ax2.set_ylabel('Fraction of Loans Charged Off')
        ax2.set_title('Charged Off Rate by ' + full_name)
        ax2.set_xlabel(full_name)
    
    # plot3: kde plot of the variable gropued by loan_status
    if continuous:  
        facet = sns.FacetGrid(df, hue = 'loan_status', size=3, aspect=4)
        facet.map(sns.kdeplot, col_name, shade=True)
        #facet.set(xlim=(df[col_name].min(), df[col_name].max()))
        facet.add_legend()  
    else:
        fig = plt.figure(figsize=(12,3))
        sns.countplot(x=col_name, hue='loan_status', data=df, order=sorted(df[col_name].unique()) )
     
    plt.tight_layout()

plot_var('loan_amnt', 'Loan Amount', continuous=True)
########################## VISUALIZATION ##########################
###################################################################
###################################################################








####################### Hypothesis Testing ########################
###################################################################
###################################################################

Here I can group the loans to "fully paid" & "charged-off", and then use hypothesis tests to compare the two distributions of each feature.

If the test statistic is small or the p-value is high (>0.05, 95% confidence level), we cannot reject the null hypothesis that the distributions of the two samples are the same and if if p<0.05, different distributions.

K-S Tests, Z Tests and chi-squared Tests:

    Numerical features: We can use K-S tests
    Features with only 0 or 1 values, we can use proportion Z tests to check whether the difference in mean values is statistically significant.
    For categorical features, we can use chi-squared Tests

In [425]:

list_float = df.select_dtypes(exclude=['object']).columns

In [426]:

def run_KS_test(feature):
    dist1 = df.loc[df.Charged_Off == 0,feature]
    dist2 = df.loc[df.Charged_Off == 1,feature]
    print(feature+':')
    print(ks_2samp(dist1,dist2),'\n')

In [427]:

from statsmodels.stats.proportion import proportions_ztest
def run_proportion_Z_test(feature):
    dist1 = df.loc[df.Charged_Off == 0, feature]
    dist2 = df.loc[df.Charged_Off == 1, feature]
    n1 = len(dist1)
    p1 = dist1.sum()
    n2 = len(dist2)
    p2 = dist2.sum()
    z_score, p_value = proportions_ztest([p1, p2], [n1, n2])
    print(feature+':')
    print('z-score = {}; p-value = {}'.format(z_score, p_value),'\n')

In [428]:

from scipy.stats import chi2_contingency
def run_chi2_test(df, feature):

    dist1 = df.loc[df.loan_status == 'Fully Paid',feature].value_counts().sort_index().tolist()
    dist2 = df.loc[df.loan_status == 'Charged Off',feature].value_counts().sort_index().tolist()
    chi2, p, dof, expctd = chi2_contingency([dist1,dist2])
    print(feature+':')
    print("chi-square test statistic:", chi2)
    print("p-value", p, '\n')

In [429]:

for i in list_float:
    run_KS_test(i)

###################################################################
###################################################################
###################################################################








####################### Correlation Test ##########################
###################################################################
###################################################################
list_float = df.select_dtypes(exclude=['object']).columns

fig, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches
cm_df = sns.heatmap(df[list_float].corr(),annot=True, fmt = ".2f", cmap = "coolwarm", ax=ax)




The linearly correlated features are:

    "installment" vs "loan_amnt" (0.95)
    "mo_sin_old_rev_tl_op"* vs "earliest_cr_line" (0.91)
    "pub_rec_bankruptcies"* vs "pub_rec" (0.75)
    "total_acc" vs "open_acc" (0.69)
    (*) with null values

Dependence of Charged-off on the predictors: "int_rate" is the most correlated one. (Also see the Table1)
In [434]:

cor = df[list_float].corr()
cor.loc[:,:] = np.tril(cor, k=-1) # below main lower triangle of an array
cor = cor.stack()
cor[(cor > 0.1) | (cor < -0.1)]

Out[434]:

term                  loan_amnt               0.386449
int_rate              loan_amnt               0.158214
                      term                    0.426839
installment           loan_amnt               0.953588
                      term                    0.145842
                      int_rate                0.160821
annual_inc            loan_amnt               0.504394
                      term                    0.122812
                      int_rate               -0.102222
                      installment             0.483259
                      emp_length              0.136435
dti                   int_rate                0.170415
                      annual_inc             -0.215161
earliest_cr_line      loan_amnt               0.148525
                      int_rate               -0.112131
                      installment             0.131444
                      emp_length              0.216412
                      annual_inc              0.202806
open_acc              loan_amnt               0.193697
                      installment             0.183500
                      annual_inc              0.226926
                      dti                     0.279120
                      earliest_cr_line        0.121138
revol_bal             loan_amnt               0.363518
                      term                    0.110520
                      installment             0.349779
                      emp_length              0.107722
                      annual_inc              0.306059
                      dti                     0.219748
                      earliest_cr_line        0.189986
                      open_acc                0.305423
                      pub_rec                -0.140176
revol_util            loan_amnt               0.108488
                      int_rate                0.256850
                      installment             0.126761
                      dti                     0.173029
                      open_acc               -0.141107
                      revol_bal               0.462365
total_acc             loan_amnt               0.216434
                      installment             0.194081
                      emp_length              0.108882
                      annual_inc              0.306976
                      dti                     0.211689
                      earliest_cr_line        0.272654
                      open_acc                0.689616
                      revol_bal               0.228850
                      revol_util             -0.108681
mo_sin_old_il_acct    loan_amnt               0.125841
                      installment             0.105532
                      emp_length              0.132067
                      annual_inc              0.201579
                      earliest_cr_line        0.360532
                      open_acc                0.128233
                      revol_bal               0.111792
                      total_acc               0.337302
mo_sin_old_rev_tl_op  loan_amnt               0.164298
                      int_rate               -0.131125
                      installment             0.145231
                      emp_length              0.211206
                      annual_inc              0.202146
                      earliest_cr_line        0.911349
                      open_acc                0.132229
                      revol_bal               0.212607
                      total_acc               0.284477
                      mo_sin_old_il_acct      0.219667
mort_acc              loan_amnt               0.230501
                      term                    0.100799
                      installment             0.197997
                      emp_length              0.208462
                      annual_inc              0.350509
                      earliest_cr_line        0.302861
                      open_acc                0.115478
                      revol_bal               0.187637
                      total_acc               0.373703
                      mo_sin_old_il_acct      0.206741
                      mo_sin_old_rev_tl_op    0.307306
pub_rec_bankruptcies  loan_amnt              -0.104761
                      pub_rec                 0.750146
                      revol_bal              -0.143372
fico_score            loan_amnt               0.100319
                      int_rate               -0.425425
                      annual_inc              0.108242
                      earliest_cr_line        0.114372
                      pub_rec                -0.220529
                      revol_util             -0.454739
                      mo_sin_old_rev_tl_op    0.118714
                      mort_acc                0.103025
                      pub_rec_bankruptcies   -0.206954
Charged_Off           term                    0.177708
                      int_rate                0.247815
                      dti                     0.123031
                      fico_score             -0.139429
dtype: float64

In [435]:

df[["installment","loan_amnt","mo_sin_old_rev_tl_op","earliest_cr_line","total_acc","open_acc", "pub_rec_bankruptcies", "pub_rec"]].isnull().any()

Out[435]:

installment             False
loan_amnt               False
mo_sin_old_rev_tl_op     True
earliest_cr_line        False
total_acc               False
open_acc                False
pub_rec_bankruptcies     True
pub_rec                 False
dtype: bool

In [436]:

list_linear = ['installment', 'mo_sin_old_rev_tl_op','total_acc','pub_rec_bankruptcies']

In [437]:

linear_corr = pd.DataFrame()

In [438]:

# Pearson coefficients
for col in df[list_float].columns:
    linear_corr.loc[col, 'pearson_corr'] = df[col].corr(df['Charged_Off'])
linear_corr['abs_pearson_corr'] = abs(linear_corr['pearson_corr'])

Sort the results by the absolute value of the Pearson Correlation
In [439]:

linear_corr.sort_values('abs_pearson_corr', ascending=False, inplace=True)
linear_corr.drop('abs_pearson_corr', axis=1, inplace=True)
linear_corr.drop('Charged_Off', axis=0, inplace=True)

In [440]:

linear_corr.reset_index(inplace=True)
#linear_corr.rename(columns={'index':'variable'}, inplace=True)

Table 1:
In [441]:

linear_corr

Out[441]:
	index 	pearson_corr
0 	int_rate 	0.247815
1 	term 	0.177708
2 	fico_score 	-0.139429
3 	dti 	0.123031
4 	mort_acc 	-0.079739
5 	annual_inc 	-0.074216
6 	revol_util 	0.072185
7 	loan_amnt 	0.064139
8 	mo_sin_old_rev_tl_op 	-0.048529
9 	installment 	0.046291
10 	earliest_cr_line 	-0.042325
11 	open_acc 	0.034652
12 	mo_sin_old_il_acct 	-0.026019
13 	pub_rec 	0.025395
14 	pub_rec_bankruptcies 	0.017314
15 	emp_length 	-0.012463
16 	total_acc 	-0.011187
17 	revol_bal 	0.002233

The variables most linearly correlated with our target variable are interest rate, loan term, Fico Score and debt-to-income ratio. The least correlated features are the revolving balance, employment length, and public record.
In [442]:

# Drop the linear correlated features
drop_cols(list_linear)



###################################################################
###################################################################
###################################################################




https://github.com/yanxiali/Predicting-Default-Clients-of-Lending-Club-Loans/blob/master/LC_Loan_full.ipynb
########################## linear_corr ############################
###################################################################
###################################################################
Linear dependence of Charged-Off

linear_corr = pd.DataFrame()
# Pearson coefficients
for col in X_train.columns:
    linear_corr.loc[col, 'pearson_corr'] = X_train[col].corr(y_train)
linear_corr['abs_pearson_corr'] = abs(linear_corr['pearson_corr'])

Sort the results by the absolute value of the Pearson Correlation

linear_corr.sort_values('abs_pearson_corr', ascending=False, inplace=True)
linear_corr.drop('abs_pearson_corr', axis=1, inplace=True)


linear_corr.reset_index(inplace=True)
#linear_corr.rename(columns={'index':'variable'}, inplace=True)

linear_corr.head(10)





###################################################################
###################################################################
###################################################################







################### remove_collinear_features #####################
###################################################################
###################################################################


# # # Feature Engineering and Selection

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Dont want to remove correlations between loss
    y = x['loss']
    x = x.drop(columns = ['loss'])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    
    # Add the score back in to the data
    x['loss'] = y
               
    return x

# Remove the collinear features above a specified correlation coefficient
data = remove_collinear_features(data, 0.6);

###################################################################
###################################################################
###################################################################







################ Recursive Elimination Feature ####################
###################################################################
###################################################################

import lightgbm as lgb
from sklearn.feature_selection import RFE

# Feature importance

#lightGBM model fit
gbm = lgb.LGBMRegressor()
gbm.fit(train, target)
gbm.booster_.feature_importance()

# importance of each attribute
fea_imp_ = pd.DataFrame({'cols':train.columns, 'fea_imp':gbm.feature_importances_})
fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)



#Recursive Feature Elimination(RFE)

# create the RFE model and select 10 attributes
rfe = RFE(gbm, 10)
rfe = rfe.fit(train, target)

# summarize the selection of the attributes
print(rfe.support_)

# summarize the ranking of the attributes
fea_rank_ = pd.DataFrame({'cols':train.columns, 'fea_rank':rfe.ranking_})
fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by=['fea_rank'], ascending = True)

###################################################################
###################################################################
###################################################################















################ Recursive Elimination Feature ####################
###################################################################
###################################################################
__author__ = 'Tilii: https://kaggle.com/tilii7'

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))


train = pd.read_csv('../input/train.csv', dtype={'id': np.str, 'loss': np.float32})
y = np.array(train['loss'])
test = pd.read_csv('../input/test.csv', dtype={'id': np.str})

# Analyze all features ; modify categorical features
trainc = train.drop(['id', 'loss'], axis=1)
testc = test.drop(['id'], axis=1)
ntrain = trainc.shape[0]
ntest = testc.shape[0]
train_test = pd.concat((trainc, testc)).reset_index(drop=True)
all_features = [x for x in trainc.columns]
cat_features = [x for x in trainc.select_dtypes(include=['object']).columns]
num_features = [x for x in trainc.select_dtypes(exclude=['object']).columns]
print('\n Categorical features: %d' % len(cat_features))
print('\n Numerical features: %d\n' % len(num_features))
for c in range(len(cat_features)):
    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes
trainc = train_test.iloc[:ntrain,:]
testc = train_test.iloc[ntrain:,:]
X = np.array(trainc)
Xt = np.array(testc)

# Define regressor and RFECV parameters
# To test the features properly, it is probably a good idea to change step=2, n_estimators to 200
# and max_depth=20 (or remove max_depth). It will take a long time, on the order of 5-10 hours
rfr = RandomForestRegressor(n_estimators=100, max_features='sqrt', max_depth=12, n_jobs=-1)
rfecv = RFECV(estimator=rfr,
              step=10,
              cv=KFold(y.shape[0],
                       n_folds=5,
                       shuffle=False,
                       random_state=101),
              scoring='neg_mean_absolute_error',
              verbose=2)

# Estimate feature importance and time the whole process
start_time = timer(None)
rfecv.fit(X, y)
timer(start_time)

# Summarize the output
print(' Optimal number of features: %d' % rfecv.n_features_)
sel_features = [f for f, s in zip(all_features, rfecv.support_) if s]
print(' The selected features are {}'.format(sel_features))

# Plot number of features vs CV scores
plt.figure()
plt.xlabel('Number of features tested x 10')
plt.ylabel('Cross-validation score (negative MAE)')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig('Allstate-RFECV.png')
plt.show()

# Save sorted feature rankings
ranking = pd.DataFrame({'Features': all_features})
ranking['Rank'] = np.asarray(rfecv.ranking_)
ranking.sort_values('Rank', inplace=True)
ranking.to_csv('./Allstate-RFECV-ranking.csv', index=False)
print(' Ranked features saved:  Allstate-RFECV-ranking.csv')

# Make a prediction ; this is only a proof of principle as
# the prediction will be poor until smaller step is are used
score = round(-np.max(rfecv.grid_scores_), 3)
test['loss'] = rfecv.predict(Xt)
test = test[['id', 'loss']]
now = datetime.now()
sub_file = 'submission_5xRFECV-RandomForest_' + str(score) + '_' + str(
    now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing submission file: %s" % sub_file)
test.to_csv(sub_file, index=False)

###################################################################
###################################################################
###################################################################

