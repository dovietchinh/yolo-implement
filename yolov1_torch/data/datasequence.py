import torch
import cv2
import numpy as np

class DataSeuqnce(torch.utils.data.Dataset):

    def __init__(self,data):
        self.data = data 

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        """ yield sigle training sample in dataset.
            Convert data txt to 7*7*30 torch.tensor
        Args:
            index (int): 
        Return:
            img, tensor_label
        """
        img_file = self.data[index].get('img')
        txt_file = self.data[index].get('txt')
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (448,448))
        img = img.astype('uint8')/255.
        label_matrix = np.zeros((7,7,30))
        #30 = 20 + 4 +1 + 4 +1
        with open(txt_file) as f:
            lines = f.readlines()
        for line in lines:
            cl,x,y,w,h = line.strip().split()
            x = float(x) 
            y= float(y)
            w = float(w)
            h = float(h)
            cl = int(cl)
            loc = [7*x, 7*y]
            loc_i = int(loc[0])
            loc_j = int(loc[1])
            y = loc[1] - loc_j
            x = loc[0] - loc_i
            label_matrix[loc_j,loc_i,24] = 1
            label_matrix[loc_j,loc_i,29] = 1
            label_matrix[loc_j,loc_i,cl] = 1
            label_matrix[loc_j,loc_i,20:24] = [x,y,w,h]
            label_matrix[loc_j,loc_i,25:29] = [x,y,w,h]
        return img,label_matrix

            
    