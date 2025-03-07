from collections import defaultdict
from time import time
from data import BaseDataset
# from data.utils import create_modal_mask, permute_modal_names
from data.nii_data_loader import nii_slides_loader, load_set, normalize_nii, seg_transform, findSegBox
import os
import os.path
import numpy as np
import cv2
import torch
import pickle
from PIL import Image
from loguru import logger

class BratsDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        logger.info("No dataset-related parser options changed.")
        return parser

    def dataset_config(self):
        msg = "dataset config\n \
            Train Mode\n \
            Img Normalize: img / max * 2 - 1\n \
            Seg Normalize: seg[>0] = 1\n \
            loader: get 2D slice and normalize (and crop to augment)\n \
            slice number: layer with biggest lesion\n \
            resize: 256\n \
            modality: seg, t1, t1ce, t2, flair\n"
        msg = msg + "Example T1 attributes: type %s, shape %s, min %s, max %s \n" % \
            (type(self.data_dict['t1'][0]).__name__, str(self.data_dict['t1'][0].shape), str(self.data_dict['t1'][0].min().item()), str(self.data_dict['t1'][0].max().item()))
        msg = msg + "Example Seg attributes: type %s, shape %s, min %s, max %s \n" % \
            (type(self.data_dict['seg'][0]).__name__, str(self.data_dict['seg'][0].shape), str(self.data_dict['seg'][0].min().item()), str(self.data_dict['seg'][0].max().item()))
        msg = msg + "Example Img Path: %s" % str(self.data_dict['img_path'][0])
        logger.info(msg)

    def __init__(self, opt):
        # 1. load form nii file
        if opt.isTrain:
            data_root = opt.dataroot
        else:
            data_root = opt.test_dataroot
        self.mode = opt.dataset_mode  # which dataset to use
        transform = normalize_nii     # how to normalize image
        segTrans = seg_transform      # how to normalize segmentation
        loader = nii_slides_loader    # how to process image
        choose_slice_num = 78         # slice layer
        resize = 256                  # resize

        flair_path = os.path.join(data_root, 'flair')
        t1_path = os.path.join(data_root, 't1')
        t1ce_path = os.path.join(data_root, 't1ce')
        t2_path = os.path.join(data_root, 't2')
        seg_path = os.path.join(data_root, 'seg')

        self.flair_set = load_set(flair_path)
        self.t1_set = load_set(t1_path)
        self.t1ce_set = load_set(t1ce_path)
        self.t2_set = load_set(t2_path)
        self.seg_set = load_set(seg_path)

        self.n_data = len(self.flair_set)

        # save_path = '/home/jiamingzhao/BraTS2020/img/train/'

        # 2. load_all modal into memory
        print('Loading BraTS Dataset ...')
        start = time()
        cache_path = os.path.join(data_root, 'cache.pkl')
        if os.path.exists(cache_path):     # load pkl
            print('load data cache from: ', cache_path)
            with open(cache_path, 'rb') as fin:
                self.data_dict = pickle.load(fin)
        else:
            print('load data from raw')    # make pkl
            self.data_dict = defaultdict(list)
            for index in range(self.n_data):
                choose_slice_num = 78
                for modal in ['seg', 't1', 't1ce', 't2', 'flair']:
                    modal_path, modal_target = getattr(self, modal+'_set')[index]  # get image path
                    modal_img = None
                    if modal == 'seg':
                        choose_slice_num = findSegBox(modal_path)                  # find layer with biggest tumor
                        modal_img = loader(modal_path, num=choose_slice_num, transform=segTrans)  # load image from memory
                    else:
                        modal_img = loader(modal_path, num=choose_slice_num, transform=transform) # load image from memory
                    modal_img = cv2.resize(modal_img, (resize, resize))            # resize image
                    self.data_dict[modal].append(modal_img)

                    # savetmp = save_path + modal + '/' + str(index+1) + '.png'
                    # imgtmp = Image.fromarray((modal_img+1)/2*255)
                    # imgtmp = imgtmp.convert('L')
                    # imgtmp.save(savetmp)

                self.data_dict['img_path'].append(modal_path)                      # append image path
                    # print(choose_slice_num)
            with open(cache_path, 'wb') as fin:
                pickle.dump(self.data_dict, fin)                                   # save as pkl
        end = time()
        print('Finish Loading, cost {:.1f}s'.format(end - start))
        self.dataset_config()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        input = {}
        for modal in ['t1', 't1ce', 't2', 'flair', 'seg']:
            modal_tmp = self.data_dict[modal][index]
            modal_tmp = torch.tensor(modal_tmp[None], dtype=torch.float)
            input[modal] = modal_tmp
        input['img_path'] = self.data_dict['img_path'][index]
        return input  # ['t1', 't1ce', 't2', 'flair', 'seg', 'img_path']

    def __len__(self):
        return self.n_data

    def get_modal_names(self):
        return ['t1', 't1ce', 't2', 'flair', 'seg']

