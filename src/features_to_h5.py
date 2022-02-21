import sys
import numpy as np
import h5py
import traceback
import time
from get_annotations import Annotation_Factory
from image_features import image_features

# sys.path.insert(0, '../../')
# from DeepFeatureExtraction.src.get_annotations import Annotation_Factory
# from DeepFeatureExtraction.src.image_features import image_features


class FeaturesH5:
    def __init__(self, args):
        self.args = args

        if 'experiment' not in args.keys():
            self.args['experiment'] = False

        if self.args['experiment']:
            a = 'experiment'
        else:
            a = ''

        self.hdf_path = f'../h5_files/{args["architecture"]}_{self.args["dataset"]}_{a}.h5'
        self.dataset = Annotation_Factory(args)
        self.extract_features = self.get_features



    def get_features(self):
        pix_thresh = 500
        n_per_dump = 100
        errors = 0
        current_index = 0
        completed = 0

        # t: all indices grater that this will be in train set
        if self.args['experiment']:
            t = self.args['n_test']
        else:
            t = -1

        try:
            with h5py.File(self.hdf_path, 'a') as h:
                h.attrs['dataset'] = self.args['dataset']
                h.attrs['architecture'] = arch

                print('getting data')
                a = time.time()

                try:  # to store features
                    data = h.create_group('data')
                except ValueError:
                    data = h.get('data')

                try:  # to store features
                    test = h.create_group('test')
                except ValueError:
                    test = h.get('test')

                preexisting_keys = list(data)
                # preexisting_keys = [x for x in data.keys()]
                b = time.time()
                print(len(preexisting_keys))
                print('done loading data')
                print(f'took {b - a:.2f} seconds')

                current_objects = []
                current_info = []
                for img, anno in self.dataset.annotations():
                    current_index += 1  # number of objects processed
                    bbox_xyxy = anno['bbox_xyxy']
                    key = anno['key']
                    x1, y1, x2, y2 = bbox_xyxy

                    if key in preexisting_keys:
                        print('key already exists')
                        continue

                    area = (x2 - x1) * (y2 - y1)
                    if area < pix_thresh:
                        # too small area to be useful
                        # print('too small')
                        continue

                    detected_object = img[y1:y2, x1:x2]

                    # print(detected_object.shape)
                    if any([x == 0 for x in detected_object.shape]):
                        print('empty object passed')
                        continue

                    # if current_index % 100 == 0:
                    #     plt.imshow(detected_object)
                    #     plt.show()

                    current_objects.append(detected_object)
                    current_info.append(anno)

                    # dump into h5 every new image to save progress in case of error and for mem efficiency
                    if current_index % n_per_dump == 0:
                        print(len(current_objects))
                        print(f'current index: {current_index}')
                        if len(current_objects) == 0:
                            continue
                        try:
                            obj_imgs = np.array(current_objects)
                            features = image_features(obj_imgs, batch_size=4, model_name=self.args['architecture'], progress=True)
                        except Exception:
                            errors += 1
                            print('error caught')
                            traceback.print_exc()
                            completed += len(current_objects)
                            current_info = []
                            current_objects = []
                            continue

                        if current_index > t:
                            for info, feat in zip(current_info, features):
                                h5_entry = data.create_dataset(info.pop('key'), data=feat)
                                for k in info.keys():
                                    h5_entry.attrs[k] = info[k]
                            completed = completed + len(current_objects)
                            current_info = []
                            current_objects = []
                            print('dump complete')
                        else:
                            for info, feat in zip(current_info, features):
                                h5_entry = test.create_dataset(info.pop('key'), data=feat)
                                for k in info.keys():
                                    h5_entry.attrs[k] = info[k]
                            completed = completed + len(current_objects)
                            current_info = []
                            current_objects = []
                            print('dump complete')
                # end reading annotations

                # feature extract the last remaining annotations
                print(f'current index: {current_index}')
                if len(current_objects) != 0:
                    try:
                        obj_imgs = np.array(current_objects)
                        features = image_features(obj_imgs, batch_size=4, model_name=self.args['architecture'], progress=True)
                    except Exception:
                        errors += 1
                        print('error caught')
                        traceback.print_exc()
                        current_info = []
                        current_objects = []

                    for info, feat in zip(current_info, features):
                        h5_entry = data.create_dataset(info.pop('key'), data=feat)
                        for k in info.keys():
                            h5_entry.attrs[k] = info[k]
                        completed = completed + len(current_objects)
            # close h5 file

        except KeyboardInterrupt:
            pass    # break execution and display info when ctl-c
        print(f'n errors: {errors}')
        print(f'number of objects processed: {current_index}')
        print(f'number of objects added to h5: {completed}')
        if errors > 300:
            print('many errors occurred, consider re-running')




if __name__ == '__main__':
    arch = 'resnet50'

    visdrone_args = {
        'dataset': 'visdrone',
        'dataset_path': '/run/media/george/STR/datasets/visdrone_old/',
        'architecture': arch,
        'visdrone_skips': 20
    }

    dota_args_exp = {
        'dataset': 'DOTA',
        'dataset_path': '/run/media/george/STR/datasets/DOTA/',
        'architecture': arch,
        'experiment': True,
        'n_test': 20000
    }

    dota_args = {
        'dataset': 'DOTA',
        'dataset_path': '/run/media/george/STR/datasets/DOTA/',
        'architecture': arch,
    }

    neovision_args = {
        'dataset': 'neovision',
        'dataset_path': '/run/media/george/STR/datasets/neovision2-training-heli/',
        'architecture': arch,
    }

    annotator = FeaturesH5(dota_args_exp)
    annotator.extract_features()
