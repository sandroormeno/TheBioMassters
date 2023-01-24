import torch
#from skimage.io import imread
import numpy as np
#from einops import rearrange
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, s1, s2, s3, months=None, labels=None, trans=None ,chip_ids=None):
        self.images = images
        self.labels = labels
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.chip_ids = chip_ids
        self.months= months or ['September'] #['September', 'July', 'August']
        self.trans=trans
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        s1s, s2s, s3s = [], [], []
        paths = self.images[ix]
        if self.s1 is not None:
            for month in self.months:
                path = paths['S1'][month]
                #path = paths['S1'] # this is for only one month
                #print(path)
                if path is None:
                    s1s.append(np.zeros((3,256, 256)))
                else:
                    s1 = cv2.imread(path)
                    s1_ = cv2.cvtColor(s1, cv2.COLOR_BGR2RGB)
                    img_ = torch.from_numpy(s1_).permute(2,0,1)
                    img_OUT = img_/255
                    s1s.append(img_OUT)
        if self.s2 is not None:
            for month in self.months:
                path = paths['S2'][month]
                #path = paths['S2'] # this is for only one month
                if path is None:
                    s2s.append(np.zeros((3,256, 256)))
                else:
                    s2 = cv2.imread(str(path))
                    s2_ = cv2.cvtColor(s2, cv2.COLOR_BGR2RGB)
                    img_ = torch.from_numpy(s2_).permute(2,0,1)
                    img_OUT = img_/255
                    s2s.append(img_OUT)
        if self.s3 is not None:
            for month in self.months:
                path = paths['S3'][month]
                #path = paths['S3'] # this is for only one month
                if path is None:
                    s3s.append(np.zeros((3,256, 256)))
                else:
                    s3 = cv2.imread(str(path))
                    s3_ = cv2.cvtColor(s3, cv2.COLOR_BGR2RGB)
                    img_ = torch.from_numpy(s3_).permute(2,0,1)
                    img_OUT = img_/255
                    s3s.append(img_OUT)
                    
        s1s = np.stack([img for img in s1s]).astype(np.float32) if len(s1s) > 0 else None
        s2s = np.stack([img for img in s2s]).astype(np.float32) if len(s2s) > 0 else None
        s3s = np.stack([img for img in s3s]).astype(np.float32) if len(s3s) > 0 else None

        if self.labels is not None:
            segmentation_npz = np.load(self.labels[ix])
            segmentation_map = segmentation_npz['arr_0']

            # ver: https://albumentations.ai/docs/getting_started/mask_augmentation/
            # https://github.com/juansensio/competis/blob/master/TheBioMassters/bck/src/ds.py

            if self.trans:     
                t = self.trans( image = s1s.squeeze(0).transpose(1, 2, 0), 
                               image0 = s2s.squeeze(0).transpose(1, 2, 0), 
                               image1 = s3s.squeeze(0).transpose(1, 2, 0), 
                               mask = segmentation_map)
                # 3 extraer las imágenes y unirlas con np.stack 
                s1s = np.expand_dims(t['image'].transpose(2, 0, 1), axis=0)
                s2s = np.expand_dims(t['image0'].transpose(2, 0, 1), axis=0)
                s3s = np.expand_dims(t['image1'].transpose(2, 0, 1), axis=0)
                segmentation = t['mask']

            else:
                segmentation = segmentation_map
           
            return s1s, s2s, s3s, segmentation

        assert self.chip_ids is not None

        return s1s, s2s, s3s, self.chip_ids[ix]
