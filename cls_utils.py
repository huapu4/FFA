import os
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import openpyxl


def judge_file(path, img_path):
    for i in os.listdir(path):
        if os.path.isdir(os.path.join(path, i)):
            judge_file(os.path.join(path, i), img_path)
        else:
            img_path.append(os.path.join(path, i))
    return img_path

# get all file path
def get_allfile(path):
    img_path = []
    return judge_file(path, img_path=img_path)

# compare index between two models
def compare_model(alist, cp_value):
    if np.prod(alist) > cp_value:
        return True, np.prod(alist)
    else:
        return False, cp_value

# output all classes' sensitivity
def calculate_t_f(y_true, y_pred):
    sensitivity_list = []
    cm = metrics.confusion_matrix(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    for i in range(len(cm)):
        sensitivity_list.append(report[str(i)]['recall'])
    sensitivity_list.append(metrics.accuracy_score(y_true, y_pred))
    return sensitivity_list

# calculate the confidence interval
def cal_confidence_interval(value, n, inter_value):
    if inter_value == 0.9:
        const = 1.64
    elif inter_value == 0.95:
        const = 1.96
    elif inter_value == 0.98:
        const = 2.33
    elif inter_value == 0.99:
        const = 2.58

    confidence_interval_upper = value + const * np.sqrt((value * (1 - value)) / n)
    confidence_interval_lower = value - const * np.sqrt((value * (1 - value)) / n)
    if confidence_interval_lower < 0:
        confidence_interval_lower = 0
    if confidence_interval_upper > 1:
        confidence_interval_upper = 1.000
    if confidence_interval_upper == 1:
        ci = '{}-{}'.format(round(confidence_interval_lower, 3), '1.000')
    else:
        ci = '{}-{}'.format(round(confidence_interval_lower, 3), round(confidence_interval_upper, 3))
    return ci


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
    plt.rc('font', family='Times New Roman')
    lw = 1
    plt.figure(figsize=(5.5, 5.5))
    for i, item in enumerate(index_list):
        fpr, tpr, roc_auc = iter(item)
        ci = cal_confidence_interval(roc_auc, len(fpr), 0.95)
        line_name = name_list[i]
        plt.plot(fpr, tpr, lw=lw, label=line_name + ' AUC = {:.3f} (95% CI:{})'.format(roc_auc, ci))

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity', fontdict={'size': 14})
        plt.ylabel('Sensitivity', fontdict={'size': 14})
        plt.title('ROC')
        plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 13})
    plt.savefig(roc_name, dpi=600)


def WriteToExcel(file_path, lists):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'output'
    for c in range(len(lists)):
        for r in range(len(lists[c])):
            ws.cell(r + 1, c + 1).value = lists[c][r]
    wb.save(file_path)
    print('Finish saving')
