
https://www.kaggle.com/ogrellier/things-you-need-to-be-aware-of-before-stacking

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.io import output_file, show, output_notebook
from bokeh.layouts import column, gridplot
from bokeh.plotting import figure
from bokeh.palettes import brewer
output_notebook()
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold



oof_dir = '../input/lgbm-with-words-and-chars-n-gram/'
oof = pd.read_csv(oof_dir +"lvl0_lgbm_clean_oof.csv")

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
class_preds = [c_ + "_oof" for c_ in class_names]
folds = KFold(n_splits=4, shuffle=True, random_state=1)

# To show what's happening I often like to display F1 scores against probability thresholds. This shows 
# how different each fold behaves for the same threshold.

# When folds do not behave properly for the same threshold the overall AUC will usually degrade and optimal 
# weights found on OOF data may not yield good results when applied to test predictions.

figures = []
for i_class, class_name in enumerate(class_names):
    # create a new plot for current class
    # Compute full score :
    full = roc_auc_score(oof[class_names[i_class]], oof[class_preds[i_class]])
    # Compute average score
    avg = 0.0
    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        avg += roc_auc_score(oof[class_names[i_class]].iloc[val_idx], oof[class_preds[i_class]].iloc[val_idx]) / folds.n_splits
    
    s = figure(plot_width=750, plot_height=300, 
               title="F1 score vs threshold for %s full oof %.6f / avg fold %.6f" % (class_name, full, avg))
    
    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        # Get False positives, true positives and the list of thresholds used to compute them
        fpr, tpr, thresholds = roc_curve(oof[class_names[i_class]].iloc[val_idx], 
                                         oof[class_preds[i_class]].iloc[val_idx])
        # Compute recall, precision and f1_score
        recall = tpr
        precision = tpr / (fpr + tpr + 1e-5)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-5)
        # Finally plot the f1_scores against thresholds
        s.line(thresholds, f1_scores, name="Fold %d" % n_fold, color=brewer["Set1"][4][n_fold])
    figures.append(s)

# put the results in a column and show
show(column(figures))




figures = []
for i_class, class_name in enumerate(class_names):
    # create a new plot for current class
    # Compute full score :
    full = roc_auc_score(oof[class_names[i_class]], oof[class_preds[i_class]])
    # Compute average score
    avg = 0.0
    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        avg += roc_auc_score(oof[class_names[i_class]].iloc[val_idx], oof[class_preds[i_class]].iloc[val_idx]) / folds.n_splits
    
    s = figure(plot_width=400, plot_height=400, 
               title="%s ROC curves OOF %.6f / Mean %.6f" % (class_name, full, avg))
    
    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        # Get False positives, true positives and the list of thresholds used to compute them
        fpr, tpr, thresholds = roc_curve(oof[class_names[i_class]].iloc[val_idx], 
                                         oof[class_preds[i_class]].iloc[val_idx])
        s.line(fpr, tpr, name="Fold %d" % n_fold, color=brewer["Set1"][4][n_fold])
        s.line([0, 1], [0, 1], color='navy', line_width=1, line_dash="dashed")

    figures.append(s)

# put the results in a column and show
show(gridplot(np.array_split(figures, 3)))



# Read submission data 
sub = pd.read_csv(oof_dir +"lvl0_lgbm_clean_sub.csv")
figures = []
for i_class, class_name in enumerate(class_names):
    s = figure(plot_width=600, plot_height=300, 
               title="Probability logits for %s" % class_name)

    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        probas = oof[class_preds[i_class]].values[val_idx]
        p_log = np.log((probas + 1e-5) / (1 - probas + 1e-5))
        hist, edges = np.histogram(p_log, density=True, bins=50)
        s.line(edges[:50], hist, legend="Fold %d" % n_fold, color=brewer["Set1"][4][n_fold])
    
    oof_probas = oof[class_preds[i_class]].values
    oof_logit = np.log((oof_probas + 1e-5) / (1 - oof_probas + 1e-5))
    hist, edges = np.histogram(oof_logit, density=True, bins=50)
    s.line(edges[:50], hist, legend="Full OOF", color=brewer["Paired"][6][1], line_width=3)
    
    sub_probas = sub[class_name].values
    sub_logit = np.log((sub_probas + 1e-5) / (1 - sub_probas + 1e-5))
    hist, edges = np.histogram(sub_logit, density=True, bins=50)
    s.line(edges[:50], hist, legend="Test", color=brewer["Paired"][6][5], line_width=3)
    figures.append(s)

# put the results in a column and show
show(column(figures))





figures = []
for i_class, class_name in enumerate(class_names):
    s = figure(plot_width=600, plot_height=300, 
               title="Probability logits for %s using rank()" % class_name)

    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        probas = (1 + oof[class_preds[i_class]].iloc[val_idx].rank().values) / (len(val_idx) + 1)
        p_log = np.log((probas + 1e-5) / (1 - probas + 1e-5))
        hist, edges = np.histogram(p_log, density=True, bins=50)
        s.line(edges[:50], hist, legend="Fold %d" % n_fold, color=brewer["Set1"][4][n_fold])
    
    oof_probas = (1 + oof[class_preds[i_class]].rank().values) / (oof.shape[0] + 1)
    oof_logit = np.log((oof_probas + 1e-5) / (1 - oof_probas + 1e-5))
    hist, edges = np.histogram(oof_logit, density=True, bins=50)
    s.line(edges[:50], hist, legend="Full OOF", color=brewer["Paired"][6][1], line_width=3)
    
    sub_probas = (1 + sub[class_name].rank().values) / (sub.shape[0] + 1)
    sub_logit = np.log((sub_probas + 1e-5) / (1 - sub_probas + 1e-5))
    hist, edges = np.histogram(sub_logit, density=True, bins=50)
    s.line(edges[:50], hist, legend="Test", color=brewer["Paired"][6][5], line_width=3)
    figures.append(s)

# put the results in a column and show
show(column(figures))

