import os
import cv2
import numpy as np
import shutil
from random import shuffle
from tqdm import tqdm
from cls_utils import get_allfile
import argparse


# Use txt file in ImageSets to split jpg and png data into dataset
# ├── dataset
# │   ├── test
# │   │   ├── arterial_phase
# │   │   ├── non_ffa
# │   │   └── venous_phase
# │   └── train
# │       ├── arterial_phase
# │       ├── non_ffa
# │       └── venous_phase

def data2txt(all_image_path, txt_file, labels):
    '''
    function: use txt to record all image and its label
    all_image_path : the path and label of all original data
    txt_file : the path of record txt
    labels: all label list
    '''
    # reset the txtfile
    if os.path.exists(txt_file):
        os.remove(txt_file)
    # shuffle and record data
    shuffle(all_image_path)
    with open(txt_file, 'w') as f:
        for id, label in enumerate(all_image_path):
            if labels[0] in label:
                f.writelines(label + ' ' + str(0) + '\n')
            elif labels[1] in label:
                f.writelines(label + ' ' + str(1) + '\n')
            elif labels[2] in label:
                f.writelines(label + ' ' + str(2) + '\n')
            else:
                f.writelines(label + ' ' + str(3) + '\n')
    f.close()


def img2jpg(load_list, save_list):
    for num in tqdm(range(len(load_list))):
        img_code = cv2.imdecode(np.fromfile(load_list[num], dtype=np.uint8), -1)
        file_name = load_list[num].replace('\\', '/').split('/')[-1]
        save_path = os.path.join(save_list[num], file_name)
        cv2.imencode('.jpg', img_code)[1].tofile(save_path)


def mk_datasetdir(target_dir, labels):
    os.mkdir(target_dir)
    for label in labels:
        os.mkdir(os.path.join(target_dir, label))


def deploy(txt_file, datasetdir, labels, split_prop):
    traindir, testdir = os.path.join(datasetdir, 'train'), os.path.join(datasetdir, 'test')
    trainsave_dir, testsave_dir = [], []
    if os.path.exists(traindir):
        shutil.rmtree(traindir)
    if os.path.exists(testdir):
        shutil.rmtree(testdir)
    mk_datasetdir(traindir, labels)
    mk_datasetdir(testdir, labels)
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        img_list = [line.split()[0] for line in lines]
        label_list = [int(line.split()[1]) for line in lines]
    f.close()

    train_list = img_list[:int(split_prop * len(img_list))]
    test_list = img_list[int(split_prop * len(img_list)):]
    train_label_list = label_list[:int(split_prop * len(label_list))]
    test_label_list = label_list[int(split_prop * len(label_list)):]
    for item in train_label_list:
        trainsave_dir.append(os.path.join(traindir, labels[item]))
    for item in test_label_list:
        testsave_dir.append(os.path.join(testdir, labels[item]))

    img2jpg(train_list, trainsave_dir)
    img2jpg(test_list, testsave_dir)


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # set optional parameters
    parser.add_argument('--task', choices=['phase', 'diagnosis'], help="Select phase data or diagnostic data",
                        required=True)
    parser.add_argument('--input', help="Input original data dir, e.g. './origin_data/01.phase_identification'",
                        required=True)
    parser.add_argument('--output', help="The dataset dir, e.g. './01.Phase_identification/dataset'", required=True)
    parser.add_argument('--prop', default=0.7, type=float,
                        help="The proportion of the trainset to the total data, prop∈(0,1)", required=True)
    parser.add_argument('--r', action='store_true', help="Record the data set in the txt")
    args = parser.parse_args()

    # different task has its own labels
    if args.task == 'phase':
        labels = ['non_ffa', 'arterial_phase', 'venous_phase']
    elif args.task == 'diagnosis':
        labels = ['normal', 'brvo', 'none_np', 'with_np']

    all_images_path = get_allfile(args.input)

    record_txt = os.path.join(args.output, args.task + '_data.txt')
    data2txt(all_images_path, record_txt, labels)

    # processe and place the original data in the train and test sets
    deploy(record_txt, args.output, labels=labels, split_prop=0.7)
