from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrices(y_real, y_pred, label_names):
    f, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    for i, label in enumerate(label_names):
        disp = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_real[:, i],
                                                                       y_pred[:, i]>0.5),
                                                                       display_labels=[0, 1])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(label)
        if i<4:
            disp.ax_.set_xlabel('')
        if i%4!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()

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
