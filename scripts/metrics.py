from sklearn import metrics
import numpy as np

def cohen_kappa_score_multilabel(y1, y2, compute_mean=True):
    score_list = np.array([])
    
    n_labels = y1.shape[1]
    for i in range(n_labels):
        label_score = metrics.cohen_kappa_score(y1[:,i], y2[:,i])
        score_list = np.append(score_list, label_score)

    if compute_mean:
        return np.mean(score_list)
    
    return score_list

def compute_challenge_metrics(gt, preds, th=0.5):
    kappa = cohen_kappa_score_multilabel(gt, preds>th)
    f1 = metrics.f1_score(gt, preds>th, average='weighted')
    auc = metrics.roc_auc_score(gt, preds, average='weighted')
    final_score = (kappa+f1+auc)/3.0

    return kappa, f1, auc, final_score
