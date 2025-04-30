import cv2
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import time
import torch
import torch.nn as nn
import pickle

# convert to RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    

# normalization
def preprocess(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def rand(a=0, b=1):
        return np.random.rand()*(b-a) + a

class seqDataset(Dataset):
    def __init__(self, dataset_path, image_size, num_frame=5 ,type='train'):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_idx = []
        self.anno_idx = []
        self.type = type
        self.image_size = image_size
        self.num_frame = num_frame
        if type == 'train':
            self.txt_path = dataset_path
            self.aug = True
        else:
            self.txt_path = dataset_path
            self.aug = False
        with open(self.txt_path) as f: 
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip('\n').split()
                self.img_idx.append(line[0])
                self.anno_idx.append(np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]]))

        ###############################
        description = pickle.load(open('/home/chenshengjia/MoPKL/emb_train_IRDST-H.pkl', 'rb'))
        embeddings = np.array(list(description.values()))
        self.cap_idx =list(description.keys())
        self.motion_cap_idx = np.array(list(description.values()))

        relation = pickle.load(open('/home/chenshengjia/MoPKL/motion_relation_IRDST-H.pkl', 'rb'))
        relations = np.array(list(relation.values()))
        self.re_idx = list(relation.keys())
        self.motion_re_idx = np.array(list(relation.values()))
        ###############################

        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        images, box = self.get_data(index)
        images = np.transpose(preprocess(images),(3, 0, 1, 2))
        
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + ( box[:, 2:4] / 2 )
        

        if self.type == 'train':
            caption_data,relation = self.get_caption_data(index) 
            multi_box = self.get_boxes_data(index)
            for box in multi_box: 
                if len(box) != 0:
                    box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
                    box[:, 0:2] = box[:, 0:2] + ( box[:, 2:4] / 2 )
        else:
            caption_data = None
            multi_box = None
            relation = None
        caption_data = np.array(caption_data)
        relation = np.array(relation)
        # caption_data = np.array(caption)
        # print(caption_data)
        return images, box, caption_data, multi_box, relation 

    def get_caption_data(self, index):
        file_name = self.img_idx[index]
        caption_frames = []
        relation = []
        cap_index = self.cap_idx.index(file_name)
        relation.append(self.motion_re_idx[cap_index])
        for id in range(0, self.num_frame):
            idx = max(cap_index - id, 0)
            caption_frames.append(self.motion_cap_idx[idx]) 
        return caption_frames[::-1], relation


    def get_boxes_data(self, index):
        file_name = self.img_idx[index]
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], '')
        img = Image.open(image_path + '%d.bmp' % image_id)
        h, w = self.image_size, self.image_size
        iw, ih = img.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
    
        boxes_frames = []
        with open(self.txt_path) as f:
            data_lines = f.readlines()
    
        for id in range(self.num_frame):
            idx = max(index - id, 0)
            line = data_lines[idx].strip().split()
            if len(line) > 1:
                label_data = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
            else:
                label_data = np.empty((0, 5))
            
            
            if label_data.size > 0:
                label_data[:, [0, 2]] = label_data[:, [0, 2]] * nw / iw + dx
                label_data[:, [1, 3]] = label_data[:, [1, 3]] * nh / ih + dy
            
                label_data[:, 0:2][label_data[:, 0:2] < 0] = 0
                label_data[:, 2][label_data[:, 2] > w] = w
                label_data[:, 3][label_data[:, 3] > h] = h
            
            boxes_frames.append(label_data)
        
        boxes_frames = [np.array(b, dtype=np.float32) for b in boxes_frames[::-1]]
        return boxes_frames

    
    def get_data(self, index):
        image_data = []
        multi_frame_label = [] 
        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index]
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], '')
        label_data = self.anno_idx[index]  # 4+1

        
        for id in range(0, self.num_frame):

            
            # with open(image_path + '%d.txt' % max(image_id - id, 0), 'r') as f:
            #     lines = f.readlines()
 
            #     labels_box = []
            #     for line in lines:
            #         if len(line.split(" "))>1:
            #             for i in range(2):
            #                 labels_box = [[int(num) for num in line.split(" ")[0].split(',')]] 
            #         else:
            #             labels_box = [[int(num) for num in line.strip().split(',')]]
            #     labels = np.array(labels_box)

            
            img = Image.open(image_path +'%d.bmp' % max(image_id - id, 0))
            img = cvtColor(img)
            iw, ih = img.size
            
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            
            img = img.resize((nw, nh), Image.Resampling.BICUBIC)  # 原图等比列缩放
            new_img = Image.new('RGB', (w,h), (128, 128, 128))  # 预期大小的灰色图
            new_img.paste(img, (dx, dy))  # 缩放图片放在正中
            image_data.append(np.array(new_img, np.float32))
            
            if len(label_data) > 0 and id == 0:
                np.random.shuffle(label_data)
                label_data[:, [0, 2]] = label_data[:, [0, 2]]*nw/iw + dx
                label_data[:, [1, 3]] = label_data[:, [1, 3]]*nh/ih + dy
                
                label_data[:, 0:2][label_data[:, 0:2]<0] = 0
                label_data[:, 2][label_data[:, 2]>w] = w
                label_data[:, 3][label_data[:, 3]>h] = h
                # discard invalid box
                box_w = label_data[:, 2] - label_data[:, 0]
                box_h = label_data[:, 3] - label_data[:, 1]
                label_data = label_data[np.logical_and(box_w>1, box_h>1)] 
                
        #     multi_frame_label.append(label_data)
        # multi_frame_label = np.array(multi_frame_label[::-1], dtype=np.float32)

        # print(multi_frame_label.shape)

        
        image_data = np.array(image_data[::-1]) # 关键帧在后 # [5,w,h,3]
        label_data = np.array(label_data, dtype=np.float32) # [:,5]
        
        # print('!!!!!!!!!!!!!!!!!', file_name)
        return image_data, label_data#, multi_frame_label
                    
def dataset_collate(batch):
    images = []
    bboxes = []
    captions = [] #####################
    multi_boxes = []
    relations = []
    for img, box, caption, multi_box, relation in batch:
        images.append(img)
        bboxes.append(box)
        captions.append(caption) ###############################
        multi_boxes.append(multi_box) ###############################
        relations.append(relation)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    # multi_boxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in multi_boxes]
    
    return images, bboxes, captions, multi_boxes, relations
