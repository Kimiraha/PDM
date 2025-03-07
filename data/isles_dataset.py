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
import imageio
from PIL import Image

class IslesDataset(BaseDataset):
    def __init__(self, opt):
        # 1. load form nii file
        if opt.isTrain:
            data_root = opt.dataroot
        else:
            data_root = opt.test_dataroot
        self.mode = opt.dataset_mode
        transform = normalize_nii
        segTrans = seg_transform
        loader = nii_slides_loader
        choose_slice_num = 78  # change to maxSeg
        resize = 256        

        flair_path = os.path.join(data_root, 'flair')
        t1_path = os.path.join(data_root, 't1')
        t1ce_path = os.path.join(data_root, 'dwi')
        t2_path = os.path.join(data_root, 't2')
        seg_path = os.path.join(data_root, 'seg')

        self.flair_set = load_set(flair_path)
        self.t1_set = load_set(t1_path)
        self.t1ce_set = load_set(t1ce_path)
        self.t2_set = load_set(t2_path)
        self.seg_set = load_set(seg_path)
        # print(self.flair_set)

        self.n_data = len(self.flair_set)

        # 2. load_all modal into memory
        print('Loading Dataset ...')
        start = time()
        cache_path = os.path.join(data_root, 'cache2D.pkl')
        if os.path.exists(cache_path):
            print('load data cache from: ', cache_path)
            with open(cache_path, 'rb') as fin:
                self.data_dict = pickle.load(fin)
        else:
            print('load data from raw')
            self.data_dict = defaultdict(list)
            for index in range(self.n_data):
                choose_slice_num = 78
                for modal in ['seg', 't1', 't1ce', 't2', 'flair']:
                    modal_path, modal_target = getattr(self, modal+'_set')[index]
                    modal_img = None
                    if modal == 'seg':
                        choose_slice_num = findSegBox(modal_path)
                        modal_img = loader(modal_path, num=choose_slice_num, transform=segTrans, crop=None)
                    else:
                        modal_img = loader(modal_path, num=choose_slice_num, transform=transform, crop=None)
                    modal_img = cv2.resize(modal_img, (resize, resize))
                    self.data_dict[modal].append(modal_img)
                self.data_dict['img_path'].append(modal_path)
                    # print(choose_slice_num)
            with open(cache_path, 'wb') as fin:
                pickle.dump(self.data_dict, fin)
        end = time()
        print('Finish Loading, cost {:.1f}s'.format(end - start))
        # self.get_img_test()


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
        return input

    def __len__(self):
        return len(self.data_dict['t2'])

    def get_modal_names(self):
        return ['t1', 'dwi', 't2', 'flair', 'seg']

    def get_img_test(self):
        for e in range(1,5+1):
            imgs = self.__getitem__(e)
            # print(imgs['t2'].shape)
            save_path = '/home/jiamingzhao/PDM-Net-Remake/watch/Augdata/index_' + str(e)
            self.check_paths(save_path)
            save_name = os.path.join(save_path, 't2'+'.png')
            self.save_image(save_name, (imgs['t2'][0]+1)/2)
            save_name = os.path.join(save_path, 't1'+'.png')
            self.save_image(save_name, (imgs['t1'][0]+1)/2)
            save_name = os.path.join(save_path, 'flair'+'.png')
            self.save_image(save_name, (imgs['flair'][0]+1)/2)
            save_name = os.path.join(save_path, 'dwi'+'.png')
            self.save_image(save_name, (imgs['t1ce'][0]+1)/2)
            save_name = os.path.join(save_path, 'seg'+'.png')
            self.save_image(save_name, (imgs['seg'][0]+1)/2)

    def save_image(self, filename, data):
        img = data.clone().clamp(0, 255).numpy()
        imageio.imwrite(filename, img)

    def check_paths(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError as e:
            print(e)
            sys.exit(1)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from options.config import load_config, load_config_single
    opt = load_config_single('/home/jiaxu/home/multi-modal-image-translation/options/multi_modal_brats/flair-t2_pix2pix.yaml')
    dataset = BratsDataset(opt)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=16, shuffle=True)
    start = time()
    for input in dataloader:
        print(input['A'].shape)
    end = time()
    print('dataset traverse time {:.1f}'.format(end-start))
    # print(data)
