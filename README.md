# EDSC-pytorch
Code for Multiple Video Frame Interpolation via Enhanced Deformable Separable Convolution [[arXiv](https://arxiv.org/abs/2006.08070)] .

Pre-trained models
---
[Google Drive](https://drive.google.com/drive/folders/1iv4_34clpnjM2af34_qdoEpEzblMMNrW?usp=sharing)

[Baidu Cloud](https://pan.baidu.com/s/1kC7dEN2ZsMS7IdOLXVuDGQ) : bdfu

Environment
---
We are good in the environment:

python 3.7

CUDA 10.1

Pytorch 1.0.0

opencv-python 4.2.0

numpy 1.18.1

cupy 6.0.0

Usage
---
We provide two versions of our model. The `EDSC_s` model was trained to generate the midpoint (in time) of the two input frames. And you can either choose the `l1` or the `lf` model for distortion and perceptual quality, respectively.

We are good to run

```
python run.py --model EDSC_s --model_state EDSC_s_l1.ckpt --out out.png
```

The `EDSC_m` model is able to generate a frame at an arbirary time position. For instance, to generate an intermediate frame at `t=0.1`, we are good to run

```
python run.py --model EDSC_m --model_state EDSC_m.ckpt --time 0.1 --out out.png
```

Please see the paper for more details.

Citation
---

```
@article{EDSC,
    author={Cheng, Xianhang and Chen, Zhenzhong},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={Multiple Video Frame Interpolation via Enhanced Deformable Separable Convolution}, 
    year={2021},
    doi={10.1109/TPAMI.2021.3100714}
}
```

Acknowledgement
---
Part of the code was adapted from [sepconv-slomo](https://github.com/sniklaus/sepconv-slomo). A huge thanks to the authors!
