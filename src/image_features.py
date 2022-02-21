import cv2
import numpy as np
import torch
from tqdm import tqdm
import pretrainedmodels
import pretrainedmodels.utils as utils
from PIL import Image
import os

"""
This is a modified version of https://github.com/chsasank/image_features/blob/master/image_features/image_features.py
and contains the necessary code for performing feature extraction from an image using pre-trained models
"""


class MyLoadImage(object):
    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, img):
        with Image.fromarray(img) as img:
            img = img.convert(self.space)
        return img


def get_model(model_name):
    # https://pypi.org/project/pretrainedmodels/
    model = getattr(pretrainedmodels, model_name)(pretrained='imagenet')
    model.eval()
    return model



class ImageLoader():
    def __init__(self, imgs, model, img_size=224, augment=False):
        if isinstance(imgs, np.ndarray):
            self.load_img = lambda x: Image.fromarray(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
        else:
            self.load_img = utils.LoadImage()
        additional_args = {}
        if augment:
            additional_args = {
                'random_crop': True, 'random_hflip': False,
                'random_vflip': False
            }
        self.tf_img = utils.TransformImage(model, scale=img_size / 256, **additional_args)
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        input_img = self.load_img(self.imgs[idx])
        input_tensor = self.tf_img(input_img)
        return input_tensor



def image_features(
        img_paths, model_name='resnet50', use_gpu=torch.cuda.is_available(),
        batch_size=32, num_workers=4, progress=False, augment=False):
    """
    Extract deep learning image features from images.

    Args:
        img_paths(list): List of paths of images to extract features from.
        model_name(str, optional): Deep learning model to use for feature
            extraction. Default is resnet50. List of avaiable featureModels are here:
            https://github.com/Cadene/pretrained-models.pytorch
        use_gpu(bool): If gpu is to be used for feature extraction. By default,
            uses cuda if nvidia driver is installed.
        batch_size(int): Batch size to be used for feature extraction.
        num_workers(int): Number of workers to use for image loading.
        progress(bool): If true, enables progressbar.
        augment(bool): If true, images are augmented before passing through
            the model. Useful if you're training a classifier based on these
            features.
    """
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if isinstance(img_paths, str):
        raise ValueError(f'img_paths should be a list of image paths.')

    model = get_model(model_name).to(device)
    dataset = ImageLoader(img_paths, model, augment=augment)
    # print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        if progress:
            pbar = tqdm(total=len(img_paths), desc='Computing image features')

        output_features = []
        for batch in dataloader:
            batch = batch.to(device)
            ftrs = model.features(batch).cpu()
            ftrs = ftrs.mean(-1).mean(-1)   # average pool
            output_features.append(ftrs)

            if progress:
                pbar.update(batch.shape[0])

        if progress:
            pbar.close()

    output_features = torch.cat(output_features).numpy()
    return output_features



def compare_vectors(v1, v2):
    v = v1-v2
    v = v**2
    return np.sum(v)**0.5

