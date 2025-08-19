## ***Language-driven Motion Prior Knowledge Learning for Moving Infrared Small Target Detection***

This project is a substantial improvement and extension version of our
**MoPKL** ***(Motion Prior Knowledge Learning with Homogeneous Language Descriptions for Moving Infrared Small Target Detection)***, published in the proceedings of the **39th AAAI Conference on Artificial Intelligence (AAAI’25)**.


## Datasets (bounding box-based)
- Datasets are available at [`ITSDT-15K`](https://drive.google.com/file/d/149HdOo8078My1FDiI8mmkH-KXJB0dvXj/view?usp=sharing), [`DAUB-R`](https://pan.baidu.com/s/1Bay8pNlDJIKCD75O3tdhuA?pwd=jya7)(code: jya7) and [`IRDST-H`](https://pan.baidu.com/s/1rLNyw1sIitSVU1yzzCQnyA?pwd=c4ar)(code: c4ar). 
DAUB-R is a reconstructed version of DAUB, split into training, validation, and test sets.
IRDST-H it is a hard version of IRDST.


- You need to reorganize these datasets in a format similar to the `coco_train_ITSDT.txt` and `coco_val_ITSDT.txt` files we provided (`.txt files` are used in training).  We provide the `.txt files` for ITSDT-15K, DAUB-R and IRDST-H.
For example:
```python
train_annotation_path = '/home/ITSDT/coco_train_ITSDT.txt'
val_annotation_path = '/home/ITSDT/coco_val_ITSDT.txt'
```
- Or you can generate a new `txt file` based on the path of your datasets. `.txt files` (e.g., `coco_train_ITSDT.txt`) can be generated from `.json files` (e.g., `instances_train2017.json`). We also provide all `.json files` for [`ITSDT-15K`](https://drive.google.com/file/d/149HdOo8078My1FDiI8mmkH-KXJB0dvXj/view?usp=sharing), [`DAUB-R`](https://pan.baidu.com/s/1Bay8pNlDJIKCD75O3tdhuA?pwd=jya7)(code: jya7) and [`IRDST-H`](https://pan.baidu.com/s/1rLNyw1sIitSVU1yzzCQnyA?pwd=c4ar)(code: c4ar). 

``` python 
python utils_coco/coco_to_txt.py
```

- The folder structure should look like this:
```
ITSDT
├─instances_train2017.json
├─instances_test2017.json
├─coco_train_ITSDT.txt
├─coco_val_ITSDT.txt
├─images
│   ├─1
│   │   ├─0.bmp
│   │   ├─1.bmp
│   │   ├─2.bmp
│   │   ├─ ...
│   ├─2
│   │   ├─0.bmp
│   │   ├─1.bmp
│   │   ├─2.bmp
│   │   ├─ ...
│   ├─3
│   │   ├─ ...
```


## Prerequisite

* python==3.11.8
* pytorch==2.1.1
* torchvision==0.16.1
* numpy==1.26.4
* opencv-python==4.9.0.80
* scipy==1.13
* Tested on Ubuntu 20.04, with CUDA 11.8, and 1x NVIDIA 3090.


## Usage of MoPKL

### Language Descriptions

- We provide the encoded [embedding representations](https://pan.baidu.com/s/18gA0735vQO_vnFHvuQmD3Q?pwd=fmag)(code: fmag) of the language descriptions for `ITSDT-15K`, `DAUB-R` and `IRDST-H` datasets. 
There are three embedded representations in this file: `emb_train_ITSDT.pkl`, `emb_train_DAUB.pkl` and `emb_train_IRDST-H.pkl`.

- We also provide initial the language description [text files](https://pan.baidu.com/s/1-IIlc527SMDXrHQ-RMqZig?pwd=yuy3)(code: yuy3) that you can explore further with vision-language models.

- Take the ITSDT-15K dataset as an example, modify the path of the `dataloader_for_ITSDT` for language description embedding representations:
```python
# Path to your emb_train_ITSDT.pkl

description = pickle.load(open('/home/MoPKL/emb_train_ITSDT.pkl', 'rb'))
embeddings = np.array(list(description.values()))
self.cap_idx =list(description.keys())
self.motion_cap_idx = np.array(list(description.values()))
```
- In addition, you need to modify the dimension of `text_input_dim` in the network file `MoPKL.py`:  
```python
# ITSDT: 130 * 300
# DAUB-R: 20 * 300
# IRDST-H: 20 * 300

self.motion = MotionModel(text_input_dim=130*300, latent_dim=128, hidden_dim=1024)
```

- We provide the encoded [tensor](https://pan.baidu.com/s/1BpSeFZQjR3KbcLKwuD7dgg?pwd=45c6)(code: 45c6) of the `motion relations` for `ITSDT-15K`, `DAUB-R` and `IRDST-H` datasets. 
There are three embedded representations in this file: `motion_relation_ITSDT.pkl`, `motion_relation_DAUB.pkl` and `motion_relation_IRDST-H.pkl`.

- Take the ITSDT-15K dataset as an example, modify the path of the `dataloader_for_ITSDT` for language description embedding representations:
```python
# Path to your motion_relation_ITSDT.pkl

description = pickle.load(open('/home/MoPKL/motion_relation_ITSDT.pkl', 'rb'))
relations = np.array(list(relation.values()))
self.re_idx = list(relation.keys())
self.motion_re_idx = np.array(list(relation.values()))
```


### Training
- Note: Please use different `dataloader` for different datasets. For example, to train the model on ITSDT dataset, enter the following command: 
```python
CUDA_VISIBLE_DEVICES=0 python train_ITSDT.py 
```

### Test
- Usually `model_best.pth` is not necessarily the best model. The best model may have a lower val_loss or a higher AP50 during verification.
```python
"model_path": '/home/MoPKL/logs/model.pth'
```
- You need to change the path of the `json file` of test sets. For example:
```python
# Use ITSDT-15K dataset for test

cocoGt_path         = '/home/public/ITSDT-15K/instances_test2017.json'
dataset_img_path    = '/home/public/ITSDT-15K/'
```
```python
python test.py
```

### Visulization
- We support `video` and `single-frame image` prediction.
```python
# mode = "video" (predict a sequence)

mode = "predict"  # Predict a single-frame image 
```
```python
python predict.py
```

### Parameters and FLOPs Calculation
```python
python summary.py
```



## Results
- For bounding box detection, we use COCO's evaluation metrics:

<table>
  <tr>
    <th>Method</th>
    <th>Dataset</th>
    <th>mAP50 (%)</th>
    <th>Precision (%)</th>
    <th>Recall (%)</th>
    <th>F1 (%)</th>
    <th>Download</th>
  </tr>
  <tr>
    <td align="center">iMoPKL</td>
    <td align="center">ITSDT-15K</td>
    <td align="center">80.67</td>
    <td align="center">92.28</td>
    <td align="center">88.50</td>
    <td align="center">90.35</td>
    <td rowspan="3" align="center">
      <a href="https://pan.baidu.com/s/18O7gEwr-QvMxrckCJrRH_Q?pwd=2u4k">Baidu</a> (code: 2u4k)
      <br>
    </td>
  </tr>
  <tr>
    <td align="center">iMoPKL</td>
    <td align="center">DAUB-R</td>
    <td align="center">88.57</td>
    <td align="center">92.94</td>
    <td align="center">96.94</td>
    <td align="center">94.90</td>
  </tr>
  <tr>
    <td align="center">iMoPKL</td>
    <td align="center">IRDST-H</td>
    <td align="center">43.95</td>
    <td align="center">59.82</td>
    <td align="center">74.48</td>
    <td align="center">66.35</td>
  </tr>
 </table>



- PR curves on `ITSDT-15K`, `DAUB-R` and `IRDST-H` datasets in this paper. 
- We also provided the result [files](https://pan.baidu.com/s/1Z8YSNDS0iHAg10UT9FpZ3w?pwd=2544)(code:2544) for these PR curves, so you can directly plot curves yourself.


<img src="PR.png" width="800px">


## Contact
If any questions, kindly contact with Shengjia Chen via e-mail: csj_uestc@126.com.

## References
1. S. Chen, L. Ji, J. Zhu, M. Ye and X. Yao, "SSTNet: Sliced Spatio-Temporal Network With Cross-Slice ConvLSTM for Moving Infrared Dim-Small Target Detection," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-12, 2024, Art no. 5000912, doi: 10.1109/TGRS.2024.3350024. 
2. Bingwei Hui, Zhiyong Song, Hongqi Fan, et al. A dataset for infrared image dim-small aircraft target detection and tracking under ground / air background[DS/OL]. V1. Science Data Bank, 2019[2024-12-10]. https://cstr.cn/31253.11.sciencedb.902. CSTR:31253.11.sciencedb.902.
3. Ruigang Fu, Hongqi Fan, Yongfeng Zhu, et al. A dataset for infrared time-sensitive target detection and tracking for air-ground application[DS/OL]. V2. Science Data Bank, 2022[2024-12-10]. https://cstr.cn/31253.11.sciencedb.j00001.00331. CSTR:31253.11.sciencedb.j00001.00331.


## Citation

If you find this repo useful, please cite our paper. 

```
@ARTICLE{CheniMoPKL2025,
  author={Chen, Shengjia and Ji, Luping and Peng, Shuang and Zhu, Sicheng and Ye, Mao and Sang, Yongsheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Language-Driven Motion Prior Knowledge Learning for Moving Infrared Small Target Detection}, 
  year={2025},
  volume={63},
  pages={1-14},
  doi={10.1109/TGRS.2025.3596902}}

@inproceedings{ChenMoPKL2025,
  title={{Motion Prior Knowledge Learning with Homogeneous Language Descriptions for Moving Infrared Small Target Detection}},
  author={Chen, Shengjia and Ji, Luping and Duan, Weiwei and Peng, Shuang and Ye, Mao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={2},
  pages={2186--2194},
  year={2025}
}
```