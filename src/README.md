# Data Processing Pipeline

Our data processing pipeline is configured to extract features in the same format regardless of how they
exist in the dataset. Since different datasets store annotations in different formats, the file [get_annotations.py](get_annotations.py)
is resposible for providing features using a generator function in a regularized format to [features_to_h5.py](features_to_h5.py)
where features are extracted for each object and stored in the h5 database.


# Getting Annotations
[get_annotations.py](get_annotations.py) contains a class <code>Annotation_Factory</code> which is initialized with an arguments dictionary that 
contains all the necessary information about the dataset being processed. The goal of this class is to provide each
of the individual annotations one by one as an image, and a dictionary containing the annotation details through
a generator function called <code>annotations()</code>. The implementation of this generator function is determined
automatically depending on the dataset.

A generator function is one that uses the yield statement rather than the return statement which saves the progress of
the function and when called again, the function will pick back up where it left off. This is convent for us because
our function will read the annotations from file in any format then provide them one by one in a regularized format.

The syntax for using the <code>annotations()</code> generator is as follows:

```python
from get_annotations import Annotation_Factory

args = {
    'architecture': 'resnet50',
    'dataset': 'DATASET_NAME',
    'dataset_path': '/path/to/dataset/root/',
}

dataset = Annotation_Factory(args)
for img, anno in dataset.annotations():
    pass
```

Note that <code>anno</code> which is the dictionary containing the details about the current annotaiton, may contain
different variables depending on the dataset however it will always contain: 'bbox_xyxy', 'category', 'image_path'
(or 'video_path' with 'frame_num'), and 'key' (recommended h5 key). all variables in the annotation should be saved to
the H5 data entry as attributes as so that the exact object may be retrieved when needed.



# Using Annotations to Extract Features
[features_to_h5.py](features_to_h5.py) is a script that uses the annotation generator function to extract features from any dataset.
It contains a class, <code>FeaturesH5</code>, that is initialized with the same dictionary as the annotation factory class mentioned before.
This will fists determine which network to use for feature extraction from the <code>architecture</code> key in the arguments dict. Then,
it dynamically names and creates the h5 files that the deep features will be saved to and creates the annotation factory object.
When the function <code>FeaturesH5.extract_features()</code> is called, features will be extracted from each object in the dataset
and saved to the h5 file. 


There may be multiple objects of interest per image in the dataset, however the annotation generator function processes each object individually.
Therefore, when extracting features, we must crop the image in each iteration based on 'bbox_xyxy' to extract features only for that object.
Each image and annotation is saved to a buffer and processed in batches by the feature extractor as explained in the main [readme](../README.md)
before being saved to the h5 file as this tends to be more efficient and causes the program to run in less time.

This class also supports creating a small test set when including the parameters
<code>'experiment': True</code> and <code>'n_test': n</code> in the arguments. <code>'n_test'</code> defines the number of
entries that will belong to the <code>test</code> H5 group, and the rest will belong to the <code>data</code> group.
By default, when these parameters are not provided, all entries in the h5 file will belong to the <code>data</code> group.

Information on how to read / write to h5 files can be found in the [H5 readme](../h5_files/H5.md)


