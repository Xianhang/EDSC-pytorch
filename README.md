# EDSC-pytorch
Code for Multiple Video Frame Interpolation via Enhanced Deformable Separable Convolution

Pre-trained models
---
Google Drive

[Baidu Cloud](https://pan.baidu.com/s/1kC7dEN2ZsMS7IdOLXVuDGQ) :bdfu

Usage
---
We provide two versions of our model. The `EDSC_s` model was trained to generate the midpoint in time of the input frames. We are good to run

```
python run.py --model EDSC_s --model_state EDSC_s_l1.ckpt
```
