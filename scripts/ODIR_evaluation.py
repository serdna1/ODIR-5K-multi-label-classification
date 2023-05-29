# -*- coding: utf-8 -*-
""" This file is only an example approach for evaluating the classification 
    performance in Ocular Disease Intelligent Recognition (ODIR-2019). 
    
    To run this file, sklearn and numpy packages are required 
    in a Python 3.0+ environment.
    
    Author: Shanggong Medical Technology, China.
    Date: July, 2019.
"""
from sklearn import metrics
import numpy as np
import sys
import xlrd
import csv

# read the ground truth from xlsx file and output case id and eight labels 
def importGT(filepath):
    data = xlrd.open_workbook(filepath)
    table = data.sheets()[0]
    data = [ [int(table.row_values(i,0,1)[0])] + table.row_values(i,-8) for i in range(1,table.nrows)]
    return np.array(data)


# read the submitted predictions in csv format and output case id and eight labels 
def importPR(gt_data,filepath):
    with open(filepath,'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        pr_data = [ [int(row[0])] + list(map(float, row[1:])) for row in reader]
    pr_data = np.array(pr_data)
    
    # Sort columns if they are not in predefined order
    order = ['ID','N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    order_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    order_dict = {item: ind for ind, item in enumerate(order)}
    sort_index = [order_dict[item] for ind, item in enumerate(header) if item in order_dict]
    wrong_col_order = 0
    if(sort_index!=order_index):
        wrong_col_order = 1
        pr_data[:,order_index] = pr_data[:,sort_index] 
    
    # Sort rows if they are not in predefined order
    wrong_row_order = 0
    order_dict = {item: ind for ind, item in enumerate(gt_data[:,0])}
    order_index = [ v for v in order_dict.values() ]
    sort_index = [order_dict[item] for ind, item in enumerate(pr_data[:,0]) if item in order_dict]
    if(sort_index!=order_index):
        wrong_row_order = 1
        pr_data[order_index,:] = pr_data[sort_index,:]
        
    # If have missing results
    missing_results = 0
    if(gt_data.shape != pr_data.shape):
        missing_results = 1
    return pr_data,wrong_col_order,wrong_row_order,missing_results


#calculate kappa, F-1 socre and AUC value
def ODIR_Metrics(gt_data, pr_data):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr>th)
    f1 = metrics.f1_score(gt, pr>th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0
    return kappa, f1, auc, final_score


if __name__ == '__main__':
    argc = len(sys.argv)  
    if argc != 3:
        print(sys.argv[0], "\n Usage: \n First String: ground_truth_filepath, \n Second String: predicted_result_filepath.")
        sys.exit(-1)  
    
    GT_filepath = sys.argv[1]
    PR_filepath = sys.argv[2]

    gt_data = importGT(GT_filepath)
    pr_data, wrong_col_order, wrong_row_order, missing_results = importPR(gt_data,PR_filepath)

    if wrong_col_order:
        print(sys.argv[0], "\n Error: Submission with disordered columns.")
        sys.exit(-1)
        
    if wrong_row_order:
        print(sys.argv[0], "\n Error: Submission with disordered rows.")
        sys.exit(-1)
        
    if missing_results:
        print(sys.argv[0], "\n Error: Incomplete submission with missing data.")
        sys.exit(-1)
        
    kappa, f1, auc, final_score = ODIR_Metrics(gt_data[:,1:], pr_data[:,1:])
    print("kappa score:", kappa, " f-1 score:", f1, " AUC vlaue:", auc, " Final Score:", final_score)