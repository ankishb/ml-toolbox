
##############################
##     Permutation Plot     ##
##############################
# To make interpretation about the feature, it shuffle the values from one feature and 
# check if it affect the accuracy and on basis of that, it computes the feature-importances.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_X, train_y)
print("score: ", model.score(val_X, val_y))

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())


##############################
##       Partial Plot       ##
##############################
# Didn't get it completely, what idea behind is to interpret the feature affect on the model.
# let's say, we have 2 features in our data, we did prediction from model, now we interpret the
# feature, in a following way.

# Modify the features, in a random fashion, and see the affect it is creating on the predcition, 
# by keeping the other feature value fixed in one row. And repreat it for multiple rows.

# Another example, we have a row of having features `distance-covered`:10 KM and `fuel-consumption`:0.3 L
# and traget is `fare-rate`. As such we have many row. Now we pick one row and fix one variable(fuel) and 
# change values of other features and see the affect on distance on fare, and repeat for other sample(row).

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_X, train_y)
print("score: ", model.score(val_X, val_y))

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feature_names = val_X.columns.tolist()
feat_to_analyze = 'feat'

pdp_goals = pdp.pdp_isolate(model=model, dataset=val_X, 
							model_features=feature_names, feature=feat_to_analyze)
pdp.pdp_plot(pdp_goals, feat_to_analyze)
plt.show()


# https://www.kaggle.com/learn-forum/65782
# Note: It doesn't tell about the importance of feature, so it means that, even the partial-
# 		importance is zero(steady), it doesn't mean that it is unimportant