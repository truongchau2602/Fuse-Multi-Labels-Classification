import numpy as np
from pycocotools.coco import COCO
from .custom import CustomDataset
from .registry import DATASETS
import mmcv
import pandas as pd
import os
import sys
import pickle

from .utils import to_tensor, random_scale
import os.path as osp
from mmcv.parallel import DataContainer as DC
from .transforms import ImageTransform, Numpy2Tensor
from .extra_aug import ExtraAugmentation
import cv2
from tqdm import tqdm
import torch


@DATASETS.register_module
class OLIVES(CustomDataset):
    CLASSES = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6')
    def __init__(self, **kwargs):
        super(OLIVES, self).__init__(**kwargs)
        print("======= OLIVES TRAINING \n\n")
        self.df = pd.read_csv(self.ann_file)
        self.single_label = False
        self.save_folder = "mllt/appendix/OLIVESdevkit"
        self.index_dic = self.get_index_dic()

    def load_annotations(self, ann_file, LT_ann_file=None):
        """Trả về list các tên ảnh để training"""
        
        self._check_csv_file(ann_file)
        if self.col_name_of_img_path in self.df.columns:
            img_infos = self.df[self.col_name_of_img_path].tolist()            
        else:
            raise ValueError(f"Khong ton tai column '{self.col_name_of_img_path}' trong file csv.")

        return img_infos
    
    def get_ann_info(self, idx):
        img_path = self.img_infos[idx]
        row = self.df.loc[self.df[self.col_name_of_img_path]== str(img_path)]
        label_value = row.loc[:, self.CLASSES[0]:self.CLASSES[-1]].values.tolist()[0]
        ann = dict(labels=label_value)
        return ann
    
    
    def get_index_dic(self, list=False, get_labels=False):
        """ build a dict with class as key and img_ids as values
        :return: dict()
        """
        print("\n\nXChest2.py ----- get_index_dic")
        if self.single_label:
            return


        """Load index_dic in .pkl file if exist"""
        
        pkl_path = os.path.join(self.save_folder, "index_dic_or_co_labels.pkl")
        index_dic, co_labels = dict(), dict()
        if os.path.isfile(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                index_dic = data['index_dic']

                co_labels = data['co_labels']
            if list==True:
                index_dic = [value for value in index_dic.values()]
            if get_labels:
                return index_dic, co_labels
            else:
                return index_dic    
        else:
            print(f"File {pkl_path} does not exist.")
        

        num_classes = len(self.get_ann_info(0)['labels'])
        gt_labels = []
        idx2img_id = []
        img_id2idx = dict()
        co_labels = [[] for _ in range(num_classes)]
        condition_prob = np.zeros([num_classes, num_classes])

        if list:
            index_dic = [[] for i in range(num_classes)]
        else:
            index_dic = dict()
            for i in range(num_classes):
                index_dic[i] = []
        
        for i, img_name in tqdm(enumerate(self.img_infos)):

            my_df = self.df.loc[self.df[self.col_name_of_img_path] == img_name]
            img_id = my_df.iloc[0][self.col_name_of_img_path]
            # print(img_id)
            label = self.get_ann_info(i)['labels']
            #rint(f"{img_id} ---- {label}")
            gt_labels.append(label)
            idx2img_id.append(img_id)
            img_id2idx[img_id] = i

            for idx in np.where(np.asarray(label) == 1)[0]:
                index_dic[idx].append(i)
                co_labels[idx].append(label)
    
        
        # for cla in range(11,13):
        for cla in tqdm(range(num_classes)):
            cls_labels = co_labels[cla]
            # print(f"{cla} --- {cls_labels}")
            num = len(cls_labels)
            condition_prob[cla] = np.sum(np.asarray(cls_labels), axis=0) / num
            
        # sys.exit()
        
        self._save_info_from_csv_annotation(self.save_folder, gt_labels, 
                                            img_id2idx, idx2img_id, condition_prob, 
                                            index_dic, co_labels)
        
        if get_labels:
            return index_dic, co_labels
        else:
            return index_dic

    def prepare_train_img(self, idx):
        # E.g: img_name = /TREX DME/GILA/0201GOD/V1/OD/TREXJ_000000.tif
        img_name = self.img_infos[idx]
 
        # load image
        img = mmcv.imread(self.img_prefix + img_name[1:])

        # img_height,img_width = img.shape[:2] 

        ann = self.get_ann_info(idx)
        gt_labels = np.asarray(ann['labels']).astype(np.float32)
        
        # extra augmentation
        if self.extra_aug is not None:
            img, gt_labels = self.extra_aug(img, gt_labels)

        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()

        # ori_shape = (img_height, img_width, 3)
        img_meta = dict(
            # ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_labels=to_tensor(gt_labels))
        
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_path = self.img_infos[idx]
        img = mmcv.imread(self.img_prefix + img_path)

        img_height,img_width = img.shape[:2]

        def prepare_single(img, scale, flip):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_height, img_width, 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            return _img, _img_meta

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            if self.flip_ratio > 0:
                _img, _img_meta = prepare_single(img, scale, True)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))

        data = dict(img=imgs, img_meta=img_metas)
        return data


    def _filter_imgs(self, min_size=32):
        return
    
    def _one_hot_encoding(self, labels, classes):
        # Tạo một vector one-hot với độ dài bằng số lượng lớp
        one_hot_vector = np.zeros((len(classes), ), dtype=np.int64)
        # one_hot_vector = [0] * len(classes)
        
        # Tìm vị trí của nhãn trong danh sách lớp và đặt giá trị tương ứng thành 1
        if labels == None:
           
            return one_hot_vector
        class_index = classes.index(labels)
        one_hot_vector[class_index] = 1
        
        return one_hot_vector

    def _check_csv_file(self, ann_file):
        """
        Kiểm tra tệp tin có tồn tại và có đuôi mở rộng là .csv hay không.
        Nếu tệp tin không tồn tại hoặc không phải là tệp tin csv, sẽ báo lỗi.

        :param file_path: Đường dẫn tới tệp tin cần kiểm tra
        """

        # Kiểm tra tệp tin có tồn tại hay không
        if not os.path.exists(ann_file):
            raise ValueError('Tệp tin không tồn tại')

        # Kiểm tra đuôi mở rộng của tệp tin
        if not ann_file.endswith('.csv'):
            raise ValueError('Tệp tin không có định dạng csv')
            
    def _save_info_from_csv_annotation(self, dest_folder, gt_labels, img_id2idx, idx2img_id, condition_prob, index_dic, co_labels):
        ''' Save gt_labels, img_id2idx, idx2img_id'''

        save_data = dict(gt_labels=gt_labels, img_id2idx=img_id2idx, idx2img_id=idx2img_id)
        # path = 'mllt/appendix/chestdevkit/terse_gt_2023_part0.pkl'
        path = os.path.join(dest_folder, "terse_gt_2023.pkl")
        if not osp.exists(path):
            mmcv.dump(save_data, path)
            print('key info saved at {}!'.format(path))
        else:
            print('already exist, wont\'t overwrite!')

        ''' save long tail information '''
        class_freq = np.sum(gt_labels, axis=0)
        # print(np.mean(class_freq), np.var(class_freq/len(gt_labels)))
        neg_class_freq = np.shape(gt_labels)[0] - class_freq
        save_data = dict(gt_labels=gt_labels, class_freq=class_freq, neg_class_freq=neg_class_freq
                        , condition_prob=condition_prob)
        # path = 'mllt/appendix/chestdevkit/longtail/class_freq_part0.pkl'
        path = os.path.join(dest_folder, "class_freq.pkl")
        if not osp.exists(path):
            mmcv.dump(save_data, path)
            print('key info saved at {}!'.format(path))
        else:
            print('already exist, wont\'t overwrite!')

        '''Save index dic and co labels'''
        save_data = dict(index_dic=index_dic, co_labels= co_labels)
        # path = "/content/drive/MyDrive/git/DistributionBalancedLoss/index_dic_or_co_labels/index_dic_or_co_labels_part0.pkl"
        path = os.path.join(dest_folder, "index_dic_or_co_labels.pkl")
        if not osp.exists(path):
            mmcv.dump(save_data, path)
            print('index_dic and co_labels at {}!'.format(path))
        else:
            print('already exist, wont\'t overwrite!')


        ''' comment the code above iff you already run the code above at once'''

        return