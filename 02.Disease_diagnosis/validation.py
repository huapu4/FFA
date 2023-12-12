import torch
import torch.nn as nn
from torch.utils import data
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cls_utils import *
from sklearn import metrics

from config import DefaultConfig
from dataset.dataset import Disease_data
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(trained_model, data_root, batch_size, excel_name, roc_name):
    test_set = Disease_data(data_root, train=False, test=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                  drop_last=False, pin_memory=False)
    trained_model.eval()
    with torch.no_grad():
        y_true, y_pred, scores, acc = [], [], [], []
        img_name = test_set.imgs
        for test_data, test_label in test_loader:
            inputs = test_data.cuda()
            labels = test_label.cuda()
            out = trained_model(inputs)
            pred = torch.max(out, 1)[1].cuda()
            score = nn.Softmax(dim=1)(out)
            scores.extend(score)
            y_true.extend(labels.data.cpu().numpy())
            y_pred.extend(pred.data.cpu().numpy())

        print(metrics.classification_report(y_true, y_pred))
        print(metrics.confusion_matrix(y_true, y_pred))
        t_normal, t_brvo, t_nonnp, t_np, accuracy = calculate_t_f(y_true, y_pred)
        print('normal_sensitivity:%.3f' % t_normal)
        print('brvo_sensitivity:%.3f' % t_brvo)
        print('non_np_sensitivity:%.3f' % t_nonnp)
        print('with_np_sensitivity:%.3f' % t_np)
        print('accuracy:%.3f' % accuracy)

        if roc_name:
            # to draw the multi-label roc curve
            roc_0 = output_class(y_true, scores, 0)
            roc_1 = output_class(y_true, scores, 1)
            roc_2 = output_class(y_true, scores, 2)
            roc_3 = output_class(y_true, scores, 3)
            roc_list = [roc_0, roc_1, roc_2, roc_3]
            name_list = ['Normal', 'BRVO', 'DR without NPAs', 'DR with NPAs']
            draw_curve(roc_list, name_list, roc_name=roc_name)

        if excel_name:
            # to record the image_name with its label and prediction
            img_name = img_name[:len(y_true)]
            names = []
            for item in img_name:
                name = item.split('/')[-1]
                names.append(name)

            excel_list = [names, y_true, y_pred]
            WriteToExcel(excel_name, excel_list)


if __name__ == '__main__':
    # set optional parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Dirpath of testdata", required=True)
    parser.add_argument('--model', help="Path of model(pth)", required=True)
    parser.add_argument('--excel', help="'Name of excel' or None",default='data_valid')
    parser.add_argument('--roc', help="'Name of roc_curve' or None",default='data_valid_roc')
    args = parser.parse_args()

    opt = DefaultConfig()
    torch.cuda.set_device(opt.pri_device)
    model = torch.load(args.model)
    test(model, args.dataset, opt.test_batch_size, args.excel, args.roc)
