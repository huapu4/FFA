import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


def judge_file(path, img_path):
    for i in os.listdir(path):
        if os.path.isdir(os.path.join(path, i)):
            judge_file(os.path.join(path, i), img_path)
        else:
            img_path.append(os.path.join(path, i))
    return img_path


def get_allfile(path):
    img_path = []
    return judge_file(path, img_path=img_path)


def compare_model(alist, cp_value):
    if np.prod(alist) > cp_value:
        return True, np.prod(alist)
    else:
        return False, cp_value


def calculate_t_f(y_true, y_pred):
    sensitivity_list = []
    cm = metrics.confusion_matrix(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    for i in range(len(cm)):
        sensitivity_list.append(report[str(i)]['recall'])
    sensitivity_list.append(metrics.accuracy_score(y_true, y_pred))
    return sensitivity_list


def output_class(y_true, score, class_index):
    single_true, single_score = [], []
    for i in range(len(y_true)):
        if y_true[i] == class_index:
            single_true.append(1)
            single_score.append(float(score[i][y_true[i]].data.cpu()))
        else:
            single_true.append(0)
            single_score.append(1 - float(score[i][y_true[i]].data.cpu()))
    fpr, tpr, threshold = roc_curve(single_true, single_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)
    return [fpr, tpr, roc_auc]


def draw_curve(index_list, name_list, roc_name):
    matplotlib.use('Agg')
    # plt.rc('font', family='Times New Roman')
    lw = 5
    plt.figure(figsize=(8, 8))
    for i, item in enumerate(index_list):
        fpr, tpr, roc_auc = iter(item)
        line_name = name_list[i]
        plt.plot(fpr, tpr, lw=lw, label=line_name + ' AUC = {:.3f}'.format(roc_auc))

        plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1-Specificity', fontdict={'size': 20})
        plt.ylabel('Sensitivity', fontdict={'size': 20})
        plt.title('ROC')
        plt.grid(linestyle='-.')
        plt.legend(loc="lower right", prop={'size': 16})
    plt.savefig(roc_name, dpi=600)


def WriteToExcel(file_path, lists):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'output'
    for c in range(len(lists)):
        for r in range(len(lists[c])):
            ws.cell(r + 1, c + 1).value = lists[c][r]
    wb.save(file_path + '.xlsx')
    print('Finish saving')
