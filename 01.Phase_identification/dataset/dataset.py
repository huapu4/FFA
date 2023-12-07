import random
from PIL import Image
from config import DefaultConfig
from torch.utils import data
from torchvision import transforms as T
from cls_utils import get_allfile

opt = DefaultConfig()


class Phase_data(data.Dataset):
    """
    this class would Get the address of all the data include image and label to train/test set
    """

    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        all_imgs = get_allfile(root)

        random.shuffle(all_imgs)

        if self.test:
            self.imgs = all_imgs
        elif train:
            self.imgs = all_imgs[:]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(opt.resize_value),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(opt.resize_value),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        Return one image each time
        """
        img_path = self.imgs[index]
        if 'non_ffa' in img_path:
            label = 0
        elif 'arterial_phase' in img_path:
            label = 1
        else:
            label = 2
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
