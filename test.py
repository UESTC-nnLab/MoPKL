import json
import os
import colorsys
from nets.MoPKL import MoPKL
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_outputs, non_max_suppression

map_mode            = 0
cocoGt_path         = '/home/public/ITSDT/instances_test2017.json'
dataset_img_path    = '/home/public/ITSDT/'
# cocoGt_path         = '/home/public/DAUB/test.json'
# dataset_img_path    = '/home/public/DAUB/'
# cocoGt_path         = '/home/public/IRDST-H/test.json'
# dataset_img_path    = '/home/public/IRDST-H/'

temp_save_path      = 'map_out/coco_eval'

class MAP_vid(object):
    _defaults = {
        
        "model_path"        : '/home/chenshengjia/iMoPKL/logs/ITSDT_80.67_90.35',
        "classes_path"      : 'model_data/classes.txt',
        "input_shape"       : [512, 512],
        "phi"               : 's',
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "letterbox_image"   : True, 
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        show_config(**self._defaults)

    def generate(self, onnx=False):
        
        self.net    = MoPKL(self.num_classes, num_frame=2) 
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image_id, images, results):
        
        image_shape = np.array(np.shape(images[0])[0:2])
        images       = [cvtColor(image) for image in images]
        image_data  = [resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image) for image in images]
        image_data = [np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)) for image in image_data]
        # (3, 640, 640) -> (3, 16, 640, 640)
        image_data = np.stack(image_data, axis=1)
        
        image_data  = np.expand_dims(image_data, 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            outputs = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if outputs[0] is None: 
                return results

            top_label   = np.array(outputs[0][:, 6], dtype = 'int32')
            top_conf    = outputs[0][:, 4] * outputs[0][:, 5]
            top_boxes   = outputs[0][:, :4]

        for i, c in enumerate(top_label):
            result                      = {}
            top, left, bottom, right    = top_boxes[i]

            result["image_id"]      = int(image_id)
            result["category_id"]   = clsid2catid[c]
            result["bbox"]          = [float(left),float(top),float(right-left),float(bottom-top)]
            result["score"]         = float(top_conf[i])
            results.append(result)
        return results

def get_history_imgs(line):
    dir_path = line.replace(line.split('/')[-1],'')
    file_type = line.split('.')[-1]
    index = int(line.split('/')[-1][:-4])
     
    return [os.path.join(dir_path,  "%d.%s" % (max(id, 0),file_type)) for id in range(index - 1, index + 1)] 
    
    

if __name__ == "__main__":
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    cocoGt      = COCO(cocoGt_path)
    ids         = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()

    if map_mode == 0 or map_mode == 1:
        yolo = MAP_vid(confidence = 0.001, nms_iou = 0.65) 

        with open(os.path.join(temp_save_path, 'eval_results.json'),"w") as f:
            results = []
            for image_id in tqdm(ids):
                image_path  = os.path.join(dataset_img_path, cocoGt.loadImgs(image_id)[0]['file_name'])
                images = get_history_imgs(image_path)
                images = [Image.open(item) for item in images]
                # image       = Image.open(image_path)
                results     = yolo.detect_image(image_id, images, results)
            json.dump(results, f)

    if map_mode == 0 or map_mode == 2:
        cocoDt      = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))
        cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox') 
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        """
        T:iouThrs [0.5:0.05:0.95] T=10 IoU thresholds for evaluation
        R:recThrs [0:0.01:100] R=101 recall thresholds for evaluation
        K: category ids 
        A: [all, small, meduim, large] A=4 
        M: maxDets [1, 10, 100] M=3 max detections per image
        """
        precisions = cocoEval.eval['precision']
        precision_50 = precisions[0,:,0,0,-1]  
        recalls = cocoEval.eval['recall']
        recall_50 = recalls[0,0,0,-1] 
                
        with open("pr_results.txt", 'w') as f: 
            for pred in precision_50:
                f.writelines(str(pred)+'\t')
        
        print("Precision: %.4f, Recall: %.4f, F1: %.4f" %(np.mean(precision_50[:int(recall_50*100)]), recall_50, 2*recall_50*np.mean(precision_50[:int(recall_50*100)])/( recall_50+np.mean(precision_50[:int(recall_50*100)]))))
        print("Get map done.")
        
        import matplotlib.pyplot as plt
        plt.figure(1) 
        plt.title('PR Curve')# give plot a title
        plt.xlabel('Recall')# make axis labels
        plt.ylabel('Precision')
        
        x_axis = plt.xlim(0,105)
        y_axis = plt.ylim(0,1.05)
        plt.figure(1)
        plt.plot(precision_50)
        plt.show()
        plt.savefig('p-r.png')

