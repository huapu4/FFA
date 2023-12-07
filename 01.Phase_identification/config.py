from datetime import datetime as dt
import torch


class DefaultConfig(object):
    train_data_root = './dataset/trainn/'  # Train set path
    test_data_root = './dataset/testt/'  # Test set path
    exter_data_root = []  # external test set path (if has)
    batch_size = 60  # the learning amount of each batch
    test_batch_size = 200  # the testing amount of each batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU
    num_workers = 0  # how many workers for loading data
    resize_value = 512  # set the image icls_3ut size
    train_log = './result_logging/train_{}-{}_{}_{}'.format(dt.now().month, dt.now().day, dt.now().hour,
                                                            dt.now().minute)  # tensorboard record
    test_log = './result_logging/test_{}-{}_{}_{}'.format(dt.now().month, dt.now().day, dt.now().hour,
                                                          dt.now().minute)  # tensorboard record
    max_epoch = 1000  # the maximun epoch in this task
    lr = 0.001  # learning rate
    # if you have mutiple gpu devices, so you may set under two param.
    device_ids = [0]  # Select which devices， e.g. gpu0,gpu1,gpu4->[0,1,4]
    pri_device = 0  # select the primary GPU number
