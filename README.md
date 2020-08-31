# EDSC-pytorch
Code for Multiple Video Frame Interpolation via Enhanced Deformable Separable Convolution

Pre-trained models
---
Google Drive

[Baidu Cloud](https://pan.baidu.com/s/1kC7dEN2ZsMS7IdOLXVuDGQ) :bdfu

Usage
---
We provide two versions of our model. The `EDSC_s` model was trained to generate the midpoint (in time) of the two input frames. And you can either choose the `l1` or the `lf` model.

We are good to run

```
python run.py --model EDSC_s --model_state EDSC_s_l1.ckpt --out out.png
```

The `EDSC_m` model is able to generate a frame at an arbirary time position. For instance, to generate an intermediate frame at `t=0.1`, we are good to run

```
python run.py --model EDSC_m --model_state EDSC_m.ckpt --time 0.1 --out.png
```
