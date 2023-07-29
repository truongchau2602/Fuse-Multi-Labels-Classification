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

import torchxrayvision as xrv
import torchvision, torchvision.transforms

@DATASETS.register_module
class XChestDataset2(CustomDataset):
  CLASSES = ('Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 
             'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 
             'Fibrosis', 'Fracture', 'Hernia', 'Infiltration', 'Lung Lesion', 'Lung Opacity', 
             'Mass', 'No Finding', 'Nodule', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 
             'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 'Subcutaneous Emphysema', 
             'Support Devices', 'Tortuous Aorta')
  def __init__(self, **kwargs):
    super(XChestDataset2, self).__init__(**kwargs)
    self.df = pd.read_csv(self.ann_file)
    
    self.xchest_aug = False
    self.data_aug_rot = 45
    self.data_aug_trans = 0.15
    self.data_aug_scale = 0.15

    
    print("\n\n XChestDataset __init__")
    print("ann_file=",self.ann_file)
    self.index_dic = self.get_index_dic()

  def load_annotations(self, ann_file, LT_ann_file=None):
    """Trả về list các tên ảnh để training"""
    self._check_csv_file(ann_file)

    if 'path' in self.df.columns:
        img_infos = self.df['path'].tolist()
        
    else:
        raise ValueError("Khong ton tai column 'path' trong file csv.")
    
    return img_infos

  def get_ann_info(self, idx):
    img_path = self.img_infos[idx]
    row = self.df.loc[self.df["path"]== img_path]
    # print(f"\n\n\n XChest2 get_ann_info:")
    label_value = row.iloc[0][6+3:32+3].values
    # print(f"label_value={label_value}\n\n\n")
    ann = dict(labels=label_value)

    return ann

  def split_csv_file(self, dataframe, n):
    
    row_count = len(dataframe)

    # Tính kích thước của từng phần bằng nhau
    chunk_size = int(row_count / n)
    root_name, file_name = os.path.split(self.ann_file)
    name, ext = os.path.splitext(file_name)
    path = os.path.join(root_name, name)
    # path = "/content/drive/MyDrive/official_data_iccv_2/label"
    # Chia file thành n phần bằng nhau và lưu chúng thành các file riêng biệt
    for i in range(n):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        if i == n - 1:
            end_idx = row_count
        chunk = dataframe.iloc[start_idx:end_idx]
        print(f"part_{i} size={chunk.shape}")
        chunk.to_csv(f'{path}_part_{i}.csv', index=False)
        print(f"Save at '{path}_part_{i}.csv'")
    print(f"Done!")

  def get_index_dic(self, list=False, get_labels=False):
    """ build a dict with class as key and img_ids as values
    :return: dict()
    """
    print("\n\nXChest2.py ----- get_index_dic")
    if self.single_label:
        return


    """Load index_dic in .pkl file if exist"""
    # pkl_path = '/content/drive/MyDrive/git/DistributionBalancedLoss/index_dic_or_co_labels/index_dic_or_co_labels_part0.pkl'
    save_folder = "./mllt/appendix/chestdevkit_" + os.path.split(self.ann_file)[-1].split('.')[0]
    print(f"save_folder={save_folder}\n")
    pkl_path = os.path.join(save_folder, "index_dic_or_co_labels.pkl")
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
    print("line 115  num_classes=", num_classes)
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
    
    for i, img_path in tqdm(enumerate(self.img_infos)):

        my_df = self.df.loc[self.df["path"] == img_path]
        img_id = my_df.iloc[0]["dicom_id"]
        # print(img_id)
        label = self.get_ann_info(i)['labels']
        #rint(f"{img_id} ---- {label}")
        gt_labels.append(label)
        idx2img_id.append(img_id)
        img_id2idx[img_id] = i
        # print(np.where(np.asarray(label) == 1)[0])
        # print(f"len {len(index_dic)}  index_dic = {index_dic}")
        # print(f"len {len(co_labels)}  index_dic = {co_labels}")
        for idx in np.where(np.asarray(label) == 1)[0]:
            index_dic[idx].append(i)
            co_labels[idx].append(label)
        # print()
        # print("index_dic = ",index_dic)
        # print("co_labels = ",co_labels)
    # sys.exit()
    
    # for cla in range(11,13):
    for cla in tqdm(range(num_classes)):
        cls_labels = co_labels[cla]
        # print(f"{cla} --- {cls_labels}")
        num = len(cls_labels)
        condition_prob[cla] = np.sum(np.asarray(cls_labels), axis=0) / num
        
    print("line 108 XChest2 OUT LOOP!")
    # sys.exit()
    
    self._save_info_from_csv_annotation(save_folder, gt_labels, img_id2idx, idx2img_id, condition_prob, index_dic, co_labels)
    
    if get_labels:
        return index_dic, co_labels
    else:
        return index_dic

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


  def augment_img(self, img, gt_labels):
    img_height, img_width = img.shape[:2] 
    data_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(self.data_aug_rot, 
                                            translate=(self.data_aug_trans, self.data_aug_trans), 
                                            scale=(1.0-self.data_aug_scale, 1.0+self.data_aug_scale)),
        torchvision.transforms.ToTensor()
    ])

    transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
    
    
    # Áp dụng các biến đổi khác cho ảnh
    transformed_img = transforms(img)
    augmented_img = data_aug(transformed_img)
    # Chuyển đổi ảnh thành tensor PyTorch
    img_tensor = to_tensor(augmented_img)

    # Tạo dictionary img_meta
    ori_shape = (img_height, img_width , 1)
    img_shape = transformed_img.shape
    pad_shape = augmented_img.shape
    scale_factor = (img_shape[0] / img.shape[0], img_shape[1] / img.shape[1])
    flip = False
    img_meta = dict(
        ori_shape=ori_shape,
        img_shape=img_shape,
        pad_shape=pad_shape,
        scale_factor=scale_factor,
        flip=flip
    )

    # Tạo dictionary data
    data = dict(
        img=DC(img_tensor, stack=True),
        img_meta=DC(img_meta, cpu_only=True),
        gt_labels=to_tensor(gt_labels)
    )
    return data


  def prepare_train_img(self, idx):
    img_path = self.img_infos[idx]
     
    # load image
    img = mmcv.imread(osp.join(self.img_prefix, img_path))

    img_height,img_width = img.shape[:2] 

    ann = self.get_ann_info(idx)
    gt_labels = np.asarray(ann['labels']).astype(np.float32)
   
    # extra augmentation
    if self.xchest_aug == True:
        data = self.augment_img(img, gt_labels)
        return data
    
    # extra augmentation
    if self.extra_aug is not None:
        img, gt_labels = self.extra_aug(img, gt_labels)

    flip = True if np.random.rand() < self.flip_ratio else False
    # randomly sample a scale
    img_scale = random_scale(self.img_scales, self.multiscale_mode)
    img, img_shape, pad_shape, scale_factor = self.img_transform(
        img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
    img = img.copy()

    ori_shape = (img_height, img_width, 3)
    img_meta = dict(
        ori_shape=ori_shape,
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
    img = mmcv.imread(osp.join(self.img_prefix, img_path))

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
    
  def prepare_raw_img(self, idx):
    img_path = self.img_infos[idx]
    img_info = self.img_infos[idx]
    img = mmcv.imread(osp.join(self.img_prefix, img_path))
    img_height,img_width = img.shape[:2] 
    _img, img_shape, pad_shape, scale_factor = self.img_transform(
        img, self.img_scales[0], flip=False, keep_ratio=self.resize_keep_ratio)
    img_meta = dict(
        ori_shape=(img_height, img_width, 3),
        img_shape=img_shape,
        pad_shape=pad_shape,
        scale_factor=scale_factor,
        flip=False)

    data = dict(img=img, img_meta=img_meta)
    return data

  def _save_info(self, gt_labels, img_id2idx, idx2img_id, condition_prob):
      '''save info for later training'''
      ''' save original gt_labels '''
      save_data = dict(gt_labels=gt_labels, img_id2idx=img_id2idx, idx2img_id=idx2img_id)
      if 'coco' in self.ann_file:
          # path = 'mllt/appendix/coco/terse_gt_2017_test.pkl'
          path = 'mllt/appendix/coco/terse_gt_2017.pkl'
      elif 'VOC' in self.ann_file:
          path = 'mllt/appendix/VOCdevkit/terse_gt_2012.pkl'
          # path = 'mllt/appendix/VOCdevkit/terse_gt_2007_test.pkl'
      elif "xchest" in self.ann_file:
          path = 'mllt/appendix/chestdevkit/terse_gt_2023.pkl'
      else:
          raise NameError

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
      if 'coco' in self.ann_file:
          # long-tail coco
          path = 'mllt/appendix/coco/longtail2017/class_freq.pkl'
          # full coco
          # path = 'mllt/appendix/coco/class_freq.pkl'
      elif 'VOC' in self.ann_file:
          # long-tail VOC
          path = 'mllt/appendix/VOCdevkit/longtail2012/class_freq.pkl'
          # full VOC
          # path = 'mllt/appendix/VOCdevkit/class_freq.pkl'
      else:
          raise NameError

      if not osp.exists(path):
          mmcv.dump(save_data, path)
          print('key info saved at {}!'.format(path))
      else:
          print('already exist, wont\'t overwrite!')
      exit()

  # def _get_col_in_csv(self, ann_file, col_num):
  #   # Truy cập vào cột thứ col_num của dataframe và chuyển thành list
  #   list_col = self.dataframe.iloc[:, col_num].tolist()

  #   return list_col

  def get_col_in_csv(self, col_num):
      # Truy cập vào cột thứ col_num của dataframe và chuyển thành list
      list_col = self.df.iloc[:, col_num].tolist()

      return list_col

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
