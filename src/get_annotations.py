import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from video_handler import VideoHandler as vh
import glob
from sklearn.model_selection import train_test_split


class Annotation_Factory:
    """
    This class contains a function annotations() that yields (annotation, image).
    image will be a numpy array image and annotation is a dict that contains all necessary information to retrieve
    the object:
    annotation = {
        'image_path': ip,
        'bbox_xyxy': bbox_xyxy,
        'category': category,
        'key': key
        }

    The purpose of this class is to make sure that data is given in this regularized format for creating annotation database
    """
    def __init__(self, args:dict):
        self.args = args
        if args['dataset'] == 'DOTA':
            self.annotations = self.dota_annotations
        elif args['dataset'] == 'visdrone':
            self.annotations = self.visdrone_mot_annotations
        elif args['dataset'] == 'neovision':
            self.annotations = self.neovision_annotations
        else:
            exit(101)


    def get_dota_file_names(self):
        path_ = self.args['dataset_path']
        img_dir = path_ + 'images/'
        anno_dir = path_ + 'annotations/'
        str_nums = [x[1:-4] for x in os.listdir(img_dir)]
        img_paths = [img_dir + x for x in os.listdir(img_dir)]
        anno_paths = [anno_dir + f'p{x}.txt' for x in str_nums]
        return img_paths, anno_paths


    def dota_annotations(self):
        img_paths, anno_paths = self.get_dota_file_names()
        for ip, ap in zip(img_paths, anno_paths):
            img = cv2.imread(ip)
            # print('files ---------------- ', ip, ap)
            with open(ap, 'r') as f:
                for i, anno in enumerate(f.readlines()):

                    if i < 3:
                        # ignore headers
                        continue

                    a = anno.split(' ')
                    x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = a
                    box = [x1, y1, x2, y2, x3, y3, x4, y4]
                    box = [int(float(c)) for c in box]
                    x1, y1, x2, y2, x3, y3, x4, y4 = box
                    xs = [x1, x2, x3, x4]
                    ys = [y1,y2,y3,y4]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    bbox_xyxy = np.array([x1, y1, x2, y2], dtype=np.int32)
                    key = f"{ip.split('/')[-1]}-{category}-{i}"
                    annotation = {
                        'path': ip,
                        'bbox_xyxy': bbox_xyxy,
                        'category': category,
                        'key': key
                        }
                    yield img, annotation


    def visdrone_mot_annotations(self):
        all_categories = {
            1: 'pedestrian',
            2: 'people',
            3: 'bicycle',
            4: 'car',
            5: 'van',
            6: 'truck',
            7: 'tricycle',
            8: 'awning-tricycle',
            9: 'bus',
            10: 'motor'
        }
        visdrone_dir = self.args['dataset_path']
        last_img = -1
        current_img = None
        skips = self.args['visdrone_skips']
        # skips = 1  # for too much data

        for split in ['train', 'val']:
            # print('-------------', split, '-------------')
            split_dir = f'{visdrone_dir}/VisDrone2019-MOT-{split}/'
            annotations_dir = split_dir + 'annotations/'
            annotation_txt = os.listdir(annotations_dir)

            for i, txt in enumerate(annotation_txt):
                # print(txt)  # alive signal
                sequence_dir = txt[:-4]
                anno_file = annotations_dir + txt
                with open(anno_file, 'r') as f:
                    for xxx, line in enumerate(f):
                        if xxx % skips != 0: continue
                        l = line.split(',')
                        l[-1] = l[-1][:-1]
                        l = [int(a) for a in l]

                        # http://aiskyeye.com/evaluate/results-format_2021/
                        frame_index, target_id, x, y, h, w, s, cat, trunc, occ = l

                        if cat not in all_categories.keys():
                            continue
                        cat = all_categories[cat]

                        key = f'{split}_{sequence_dir}_{frame_index}_{target_id}_{cat}'
                        img_path = split_dir + f'sequences/{sequence_dir}/{str(frame_index).zfill(7)}.jpg'

                        if frame_index is not last_img:
                            current_img = cv2.imread(img_path)

                        annotation = {
                            'image_path': img_path,
                            'bbox_xyxy': [x, y, x+h, y+w],
                            'category': cat,
                            'key': key
                        }
                        yield current_img, annotation


    def neovision_annotations(self):
        nvdir = self.args['dataset_path']
        csv_files = glob.glob(f'{nvdir}*.csv')
        # using get_frame apparently does not work so out workarouund it to get the annotations in a dict first,
        # then read the frames with frame itr
        for csv in csv_files:
            all_annos_dct = {}
            vdo_path = csv[:-4]+'.mpg'

            with open(csv, 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        # skip header
                        continue
                    l = line.split(',')
                    # print('========', len(l))
                    # print(l)
                    frame_no,x1,y1,x2,y2,x3,y3,x4,y4,category,o,a,conf,_,v = l
                    frame_no,x1,y1,x2,y2,x3,y3,x4,y4 = [int(x) for x in [frame_no, x1, y1, x2, y2, x3, y3, x4, y4]]
                    xs = [x1, x2, x3, x4]
                    ys = [y1,y2,y3,y4]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    bbox_xyxy = np.array([x1, y1, x2, y2], dtype=np.int32)

                    key = f'{csv.split("/")[-1][:-4]}_{i}-{frame_no}-{category}'

                    annotation = {
                        'path': 'video_feed',
                        'video_path': vdo_path,
                        'frame_num': frame_no,
                        'bbox_xyxy': bbox_xyxy,
                        'category': category,
                        'key': key
                    }

                    all_annos_dct.setdefault(frame_no, [])
                    all_annos_dct[frame_no].append(annotation)

            # end csv read
            video = vh(vdo_path)
            for frame_img, fno in video.frame_itr():
                if fno not in all_annos_dct.keys():
                    continue

                for a in all_annos_dct[fno]:
                    yield frame_img, a
