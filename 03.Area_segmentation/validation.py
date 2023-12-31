import argparse
import os

import tensorflow as tf

from builders.model_builder import builder
from utils.data_generator import ImageDataGenerator
from utils.helpers import get_dataset_info, check_related_path
from utils.losses import categorical_crossentropy_with_logits, miou_loss, focal_loss, self_balanced_focal_loss
from utils.metrics import MeanIoU


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Choose the semantic segmentation methods.', type=str, required=True)
parser.add_argument('--base_model', help='Choose the backbone model.', type=str, default=None)
parser.add_argument('--dataset', help='The path of the dataset.', type=str, required=True)
parser.add_argument('--num_classes', help='The number of classes to be segmented.', type=int, default=3, required=True)
parser.add_argument('--crop_height', help='The height to crop the image.', type=int, default=512)
parser.add_argument('--crop_width', help='The width to crop the image.', type=int, default=512)
parser.add_argument('--batch_size', help='The testing batch size.', type=int, default=5)
parser.add_argument('--weights', help='The path of weights to be loaded.', type=str, default=None)
parser.add_argument('--loss', help='The loss function for testing.', type=str, default=None,
                    choices=['ce', 'focal_loss', 'miou_loss', 'self_balanced_focal_loss'])
args = parser.parse_args()

# check related paths
paths = check_related_path(os.getcwd())

# get image and label file names for testing
_, _, _, _, test_image_names, test_label_names = get_dataset_info(args.dataset)

# build the model
net, base_model = builder(args.num_classes, (args.crop_height, args.crop_width), args.model, args.base_model)

# summary
net.summary()

# chose loss
losses = {'ce': categorical_crossentropy_with_logits,
          'focal_loss': focal_loss(),
          'miou_loss': miou_loss(num_classes=args.num_classes),
          'self_balanced_focal_loss': self_balanced_focal_loss()}
loss = losses[args.loss] if args.loss is not None else categorical_crossentropy_with_logits

# load weights
print('Loading the weights...')
if args.weights is None:
    net.load_weights(filepath=os.path.join(
        paths['weigths_path'], '{model}_based_on_{base_model}.h5'.format(model=args.model, base_model=base_model)))
else:
    if not os.path.exists(args.weights):
        raise ValueError('The weights file does not exist in \'{path}\''.format(path=args.weights))
    net.load_weights(args.weights)

# compile the model
net.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=loss,
            metrics=[MeanIoU(args.num_classes)])
# data generator
test_gen = ImageDataGenerator()

test_generator = test_gen.flow(images_list=test_image_names,
                               labels_list=test_label_names,
                               num_classes=args.num_classes,
                               batch_size=args.batch_size,
                               target_size=(args.crop_height, args.crop_width))

# begin testing
print("\n***** Begin testing *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Base Model -->", base_model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", args.num_classes)

print("")

# some other testing parameters
steps = len(test_image_names) // args.batch_size

# testing
scores = net.evaluate_generator(test_generator, steps=steps, workers=os.cpu_count(), use_multiprocessing=False)

print('*****************************************')
print('* The loss and Mean IoU are as follows: *')
print('loss={loss:0.3f}, MeanIoU={mean_iou:0.3f}'.format(loss=scores[0], mean_iou=scores[1]))
print('*****************************************')

