# Getting Started
Easy to use pipeline for extracting image deep features using a convolutional neural network (CNN).

First ensure that the python environment being used is python 3.8, and the necessary packages in the
reqs.txt file are installed using pip.
```bash
> which python
> pip install -r reqs.txt
```
If there are problems with mkl-random or mkl-fft, remove all mkl-* from the reqs file and install mkl by itself with pip.

## Extracting Features to H5
[features_to_h5.py](src/features_to_h5.py) can be used to extract features from any of the following datasets and
save them to a h5 file:
   * DOTA
   * Visdrone
   * Neovision

The only input required is to set the paths to the correct dataset locations in the dictionaries defined at the bottom of
the file, and change which dataset to extract features from by changing the input argument when instantiating the annotator object.

```python

if __name__ == '__main__':
    args = {
        'dataset': 'DATASET_NAME',
        'dataset_path': '/path/to/dataset/root/',
    }
    annotator = FeaturesH5(args)
    annotator.extract_features()

```
This script is part of our data processing pipeline for efficient feature extraction. More information about 
how each component in the pipeline functions can be found [here](src/README.md).


# Image Feature Extraction
[image_features.py](src/image_features.py) can be used to extract features from an image or group of images with
the function <code>image_features()</code>. This must take in a list of images as a numpy array and requires the 
name of the network to also be passed. Any networks that come pre-trained with pytorch can be used, but our work primarily uses:
   * resnet50
   * resnet18
   * densenet121
   * polynet

A list of other pre-trained networks can be found at https://github.com/Cadene/pretrained-models.pytorch.

Below is an example to show how features are extracted from images.

```python
from image_features import image_features
import cv2
import numpy as np

img1 = cv2.imread('sample1.png')
img2 = cv2.imread('sample2.png')

features = image_features(np.array([img1, img2]), model_name='resnet50')

feat1 = features[0]
feat2 = features[1]
```



