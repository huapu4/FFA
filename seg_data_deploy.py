import os
import cv2
import numpy as np
import shutil
from random import shuffle
from tqdm import tqdm
from cls_utils import get_allfile
import argparse

# Use txt file in ImageSets to split jpg and png data into dataset
# |-- dataset
# |  |-- train
# |  |  |-- images
# |  |  |-- labels
# |  |-- valid
# |  |  |-- images
# |  |  |-- labels
# |  |-- test
# |  |  |-- images
# |  |  |-- labels

def data2txt(all_image_path, txt_file):
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
        for path in all_image_path:
            f.writelines(path + '\n')
    f.close()


def img2jpg(img_list, label_list, dataset_dir):
    for num in tqdm(range(len(img_list))):
        shutil.copy(img_list[num], os.path.join(dataset_dir, 'images'))
        shutil.copy(label_list[num], os.path.join(dataset_dir, 'labels'))



def mk_datasetdir(target_dir):
    os.mkdir(target_dir)
    for item in ['images', 'labels']:
        os.mkdir(os.path.join(target_dir, item))


def deploy(txt_file, datasetdir, split_prop):
    traindir = os.path.join(datasetdir, 'train')
    validdir, testdir = os.path.join(datasetdir, 'valid'), os.path.join(datasetdir, 'test')
    if os.path.exists(traindir):
        shutil.rmtree(traindir)
    if os.path.exists(validdir):
        shutil.rmtree(validdir)
    if os.path.exists(testdir):
        shutil.rmtree(testdir)
    mk_datasetdir(traindir)
    mk_datasetdir(validdir)
    mk_datasetdir(testdir)
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        img_list = [line.split('\n')[0] for line in lines]
        label_list = [img.replace('jpg', 'png').replace('JPEGImages', 'SegmentationClassPNG') for img in img_list]
    f.close()

    train_list = img_list[:int(split_prop * len(img_list))]
    valid_list = img_list[int(split_prop * len(img_list)):int(((1-split_prop)/2+split_prop) * len(img_list))]
    test_list  = img_list[int(((1-split_prop)/2+split_prop) * len(img_list)):]
    train_label_list = label_list[:int(split_prop * len(label_list))]
    valid_label_list = label_list[int(split_prop * len(label_list)):int(((1-split_prop)/2+split_prop) * len(label_list))]
    test_label_list = label_list[int(((1-split_prop)/2+split_prop) * len(label_list)):]

    img2jpg(train_list, train_label_list, os.path.join(datasetdir, 'train'))
    img2jpg(valid_list, valid_label_list, os.path.join(datasetdir, 'valid'))
    img2jpg(test_list, test_label_list, os.path.join(datasetdir, 'test'))


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # set optional parameters
    parser.add_argument('--input', help="Input voc_data dir, e.g. './origin_data/03.area_segmentation/voc_data'",
                        required=True)
    parser.add_argument('--output', help="The dataset dir, e.g. './03.Area_segmentation/FFA_dataset'", required=True)
    parser.add_argument('--prop', default=0.7, type=float,
                        help="The proportion of the trainset to the total data, prop∈(0,1)", required=True)
    parser.add_argument('--r', action='store_true', help="Record the data set in the txt")
    args = parser.parse_args()

    all_images = get_allfile(os.path.join(args.input, 'JPEGImages'))
    record_txt = os.path.join(args.output, 'data.txt')
    data2txt(all_images, record_txt)

    # processe and place the original data in the train and test sets
    deploy(record_txt, args.output, split_prop=0.7)
