from torch.utils.data import Dataset
import numpy as np
import cv2

import utilities as UT

class DCADataset(Dataset):
    def __init__(self, 
                 label_file='../data/train_list.csv',
                 resize=True,
                 augmentation=True):
        
        self.resize = resize        
        self.augmentation = augmentation
        self.files = UT.read_csv(label_file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        temp = self.files[idx]
        full_path = temp[0]
        label = temp[1]
        
        im = cv2.imread(full_path)/255. 
        if(self.resize):
            im = cv2.resize(im,(832,832))        
                
        if(self.augmentation):
            # apply random flip
            flip = np.random.randint(4, size=3)
            if(flip[0]):
                im = cv2.flip(im,0) # flip horizontally
            elif(flip[1]):
                im = cv2.flip(im,1) # flip vertically
            elif(flip[2]):
                im = cv2.flip(im,-1) # flip both horizontally and vertically

            # random rotate
            rotate = np.random.randint(4)
            (h, w) = im.shape[:2]
            # calculate the center of the image
            center = (w / 2, h / 2)
            angle90 = 90
            angle180 = 180
            angle270 = 270
            scale = 1.0

            # Perform the counter clockwise rotation holding at the center
            if(rotate==0):
                M = cv2.getRotationMatrix2D(center, angle90, scale)           
                im = cv2.warpAffine(im, M, (h, w))
            elif(rotate==1):
                M = cv2.getRotationMatrix2D(center, angle180, scale)           
                im = cv2.warpAffine(im, M, (h, w))
            elif(rotate==2):
                M = cv2.getRotationMatrix2D(center, angle270, scale)           
                im = cv2.warpAffine(im, M, (h, w))
                
        return np.transpose(im, [2,0,1]), np.asarray(int(label)), full_path