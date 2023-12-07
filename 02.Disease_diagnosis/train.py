import sys
import os
import torch as t
from sklearn import metrics
from torch import nn, optim
from torch.utils import data
from torchvision.models import resnet152 as ResNet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DefaultConfig
from dataset.dataset import Disease_data
from cls_utils import *
from tensorboardX import SummaryWriter

opt = DefaultConfig()
t.cuda.set_device(opt.pri_device)

writer_train = SummaryWriter(log_dir=opt.train_log)
writer_test = SummaryWriter(log_dir=opt.test_log)

model = ResNet(pretrained=False, progress=True)
fc_features = model.fc.in_features
model.fc = nn.Linear(in_features=fc_features, out_features=4, bias=True)

try:
    model = nn.DataParallel(model, opt.device_ids).cuda()
except:
    model = model.cuda()


def train():
    train_set = Disease_data(opt.train_data_root, train=True)
    train_loader = data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                                   drop_last=False, pin_memory=False)

    val_set = Disease_data(opt.test_data_root, train=False, test=True)
    val_loader = data.DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                 drop_last=False, pin_memory=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    cp_value = 0.5
    for epoch in range(1, opt.max_epoch + 1):
        print('epoch', epoch)
        '''Start training'''
        train_loss = 0
        y_true, y_pred, acc = [], [], []
        model.train()
        for train_data, train_label in train_loader:
            inputs = train_data.cuda()
            labels = train_label.cuda()
            out = model(inputs)
            loss = criterion(out, labels)
            train_loss += loss.item()
            pred = t.max(out, 1)[1].cuda()
            for i in labels.data.cpu().numpy():
                y_true.append(i)
            for i in pred.data.cpu().numpy():
                y_pred.append(int(i))
            targets = ['normal', 'brvo', 'none_np', 'with_np']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(metrics.classification_report(y_true, y_pred, target_names=targets))
        print(metrics.confusion_matrix(y_true, y_pred))

        t_normal, t_brvo, t_nonnp, t_np, accuracy = calculate_t_f(y_true, y_pred)
        writer_train.add_scalar('normal_sensitivity', t_normal, global_step=epoch)
        writer_train.add_scalar('brvo_sensitivity', t_brvo, global_step=epoch)
        writer_train.add_scalar('non_np_sensitivity', t_nonnp, global_step=epoch)
        writer_train.add_scalar('with_np_sensitivity', t_np, global_step=epoch)
        writer_train.add_scalar('accuracy', accuracy, global_step=epoch)

        '''Internal Test'''
        model.eval()
        with t.no_grad():
            val_loss = 0
            y_true, y_pred, acc = [], [], []
            for val_data, val_label in val_loader:
                inputs = val_data.cuda()
                labels = val_label.cuda()
                out = model(inputs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                pred = t.max(out, 1)[1].cuda()
                for i in labels.data.cpu().numpy():
                    y_true.append(i)
                for i in pred.data.cpu().numpy():
                    y_pred.append(i)
        print(metrics.classification_report(y_true, y_pred))
        print(metrics.confusion_matrix(y_true, y_pred))
        t_normal, t_brvo, t_nonnp, t_np, accuracy = calculate_t_f(y_true, y_pred)

        writer_test.add_scalar('normal_sensitivity', t_normal, global_step=epoch)
        writer_test.add_scalar('brvo_sensitivity', t_brvo, global_step=epoch)
        writer_test.add_scalar('non_np_sensitivity', t_nonnp, global_step=epoch)
        writer_test.add_scalar('with_np_sensitivity', t_np, global_step=epoch)
        writer_test.add_scalar('accuracy', accuracy, global_step=epoch)

        judge_save, cp_value = compare_model([t_normal, t_brvo, t_nonnp, t_np], cp_value)
        if judge_save:
            print('Saved the best model')
            save_path = "./checkpoints/ep{}-ep{}_lr{}.pth".format(epoch, opt.max_epoch, opt.lr)
            t.save(model, save_path)


if __name__ == '__main__':
    train()
