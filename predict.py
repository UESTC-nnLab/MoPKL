import time
import cv2
import numpy as np
from PIL import Image
from test import get_history_imgs
import colorsys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from nets.MoPKL import MoPKL
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_outputs, non_max_suppression


class Pred_vid(object):
    _defaults = {
        
        "model_path"        : '/home/chenshengjia/iMoPKL/logs/ITSDT_80.67_90.35.pth',
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
        self.net    = MoPKL(self.num_classes, num_frame=5)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
 
    def detect_image(self, images, crop = False, count = False):
  
        image_shape = np.array(np.shape(images[0])[0:2])
       
        images       = [cvtColor(image) for image in images]

        c_image = images[-1]
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
                return c_image

            top_label   = np.array(outputs[0][:, 6], dtype = 'int32')
            top_conf    = outputs[0][:, 4] * outputs[0][:, 5]
            top_boxes   = outputs[0][:, :4]
            
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * c_image.size[1] + 15).astype('int32'))  #######
        thickness   = int(max((c_image.size[0] + c_image.size[1]) // np.mean(self.input_shape), 1))
        
        if count:
            print("top_label:", len(top_label))
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        
        print(len(top_label))
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(c_image.size[1], np.floor(bottom).astype('int32'))
                right   = min(c_image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = c_image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
       
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(c_image.size[1], np.floor(bottom).astype('int32'))
            right   = min(c_image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(c_image)
            label_size = draw.textbbox((125, 20),label, font)
            label = label.encode('utf-8')
           
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            del draw

        return c_image
    
if __name__ == "__main__":
    yolo = Pred_vid()
    
    # mode = "video"
    mode = "predict"
    
   
    crop            = False
    count           = False

    if mode == "predict":
        
        while True:
            img = input('Input image filename:')
            try:
                img = get_history_imgs(img)
                images = [Image.open(item) for item in img]
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(images, crop = crop, count=count)
                r_image.save("pred.png")
    if mode == "video":
        import numpy as np
        from tqdm import tqdm
        dir_path = '/home/public/ITSDT/images/58'
        images = os.listdir(dir_path)
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.ipynb_checkpoints'):
                images.remove('.ipynb_checkpoints') 
        images = [fn for fn in images if fn.endswith("bmp")]
        images.sort(key=lambda x:int(x[:-4]))
        list_img = []
        for image in tqdm(images):
            image = dir_path+image
            img = get_history_imgs(image)
            imgs = [Image.open(item) for item in img]
            r_image = yolo.detect_image(imgs, crop = crop, count=count)
            list_img.append(cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR))
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')# *'XVID'    
        outfile = cv2.VideoWriter("./output.avi", fourcc, 24, (256, 256), True) 
        
        for i in list_img: 
            outfile.write(i) 
            if cv2.waitKey(1) == 27: 
                break 
        outfile.release()
        cv2.destroyAllWindows()
    

    
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
