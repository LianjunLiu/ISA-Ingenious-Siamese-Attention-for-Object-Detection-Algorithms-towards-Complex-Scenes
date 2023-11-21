# ISA-Ingenious-Siamese-Attention-for-Object-Detection-Algorithms-towards-Complex-Scenes

This is the official repository for paper "ISA: Ingenious Siamese Attention for Object Detection Algorithms towards Complex Scenes".

---

Our paper has been accepted by ***ISA Transactions***.

Link: https://www.sciencedirect.com/science/article/abs/pii/S0019057823004007

A Chinese translation of our paper has been published on ***Zhihu***.

Link: https://zhuanlan.zhihu.com/p/659753274


---

## Introduction

The interference of complex environments on object detection tasks dramatically limits the application of object detection algorithms. Improving the detection accuracy of the object detection algorithms is able to effectively enhance the stability and reliability of the object detection algorithm-based tasks in complex environments. In order to ameliorate the detection accuracy of object detection algorithms under various complex environment transformations, this work proposes the Siamese Attention YOLO (SAYOLO) object detection algorithm based on ingenious siamese attention structure. The ingenious siamese attention structure includes three aspects: Attention Neck YOLOv4 (ANYOLOv4), siamese neural network structure and special designed network scoring module. In the Complex Mini VOC dataset, the detection accuracy of SAYOLO algorithm is 12.31%, 48.93%, 17.80%, 10.12%, 18.79% and 1.12% higher than Faster-RCNN (Resnet50), SSD (Mobilenetv2), YOLOv3, YOLOv4, YOLOv5-l and YOLOX-x, respectively. Compared with traditional object detection algorithms based on image preprocessing, the detection accuracy of SAYOLO is 4.88%, 11.51%, 1.73%, 23.27%, 18.12%, and 5.76% higher than Image-Adaptive YOLO, MSBDN-DFF + YOLOv4, Dark Channel Prior + YOLOv4, Zero-DCE + YOLOv4, MSBDN-DFF + Zero-DCE + YOLOv4, and Dark Channel Prior + Zero-DCE + YOLOv4, respectively.

### Overall Network Structure

![](./assets/SiameseAttentionYOLO-eps-converted-to_00.png)

### Attention Neck YOLOv4 (ANYOLOv4)

![](./assets/AttentionNeckYOLO-eps-converted-to_00.png)

### Special Designed Network Scoring Module

![](./assets/NetworkScoringModule-eps-converted-to_00.png)

### Detection Results Under Real Limit Conditions

![](./assets/ComparewithSOTAAlgorithms2-eps-converted-to_00.png)

![](./assets/ComparewithSOTAAlgorithms1-eps-converted-to_00.png)

## Usage

### Environment Install

```python
pip install -r requirements.txt
```

### Weights

Download the SAYOLO Network Weights and place them in the "model_data" folder.

**SAYOLO Network Weights**:

Link: https://pan.baidu.com/s/1xu-TkFKaDyfVU6-sars9vQ?pwd=Lisa

### Detection

```python
python predict.py
```

### Dataset

Download the Complex Mini VOC Dataset and place them in the "ISA-Ingenious-Siamese-Attention-for-Object-Detection-Algorithms-towards-Complex-Scenes" folder.

**Complex Mini VOC Dataset**:

Link: https://pan.baidu.com/s/1sxbn3gvr0pdES-dURVro6g?pwd=Lisa

### Train

```python
python train.py
```

## Paper Content Correction

- The "CBAB" in the third row and fifth column of Table 5 should be "CBAM".


## Acknowledgement

This work was supported by the National Natural Science Foundation of China (No. 62003296), the Natural Science Foundation of Hebei (No. F2020203031), the Science and Technology Project of Hebei Education Department (No. QN2020225), the National Undergraduate Training Program for Innovation and Entrepreneurship of China (No. 202210216001).

## Citation

```latex
@article{LIU2023,
title = {ISA: Ingenious Siamese Attention for object detection algorithms towards complex scenes},
journal = {ISA Transactions},
year = {2023},
issn = {0019-0578},
doi = {https://doi.org/10.1016/j.isatra.2023.09.001},
url = {https://www.sciencedirect.com/science/article/pii/S0019057823004007},
author = {Lianjun Liu and Ziyu Hu and Yan Dai and Xuemin Ma and Pengwei Deng},
keywords = {Complex scenes, Object detection, Siamese network, YOLO}
}
```

## Reference

https://github.com/bubbliiiing/yolov4-pytorch
