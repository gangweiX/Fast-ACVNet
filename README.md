# Fast-ACVNet
This is the implementation of the paper: [Accurate and Efficient Stereo Matching via Attention Concatenation Volume](https://arxiv.org/pdf/2209.12699.pdf), Gangwei Xu, Yun Wang, Junda Cheng, Jinhui Tang, Xin Yang

# Demo on KITTI raw data

A demo result on our RTX 3090 (Ubuntu 20.04).

<p align="center">
  <img width="844" height="446" src="./demo/kittiraw_demo.gif" data-zoomable>
</p>

# Introduction

We design a fast version of ACV to enable real-time performance, named Fast-ACV,  which generates high likelihood disparity hypotheses and the  corresponding attention weights from low-resolution correlation clues to significantly reduce computational and memory cost and meanwhile maintain a satisfactory accuracy. The core idea of our Fast-ACV is volume attention propagation (VAP) which can automatically select accurate correlation values from an upsampled correlation volume and propagate these accurate values to the surroundings pixels with ambiguous correlation clues. In addition, we propose a “Fine-to-Important” sampling strategy which generates a set of disparity hypotheses with high likelihood and the corresponding attention weights to significantly suppress impossible disparities in the concatenation volume and in turn reduce time and memory cost.

![image](https://github.com/gangweiX/Fast-ACVNet/blob/main/imgs/Fast-ACV.png)

# How to use

## Environment
* Python 3.8
* Pytorch 1.10

## Install

### Create a virtual environment and activate it.

```
conda create -n fast_acv python=3.8
conda activate fast_acv
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm==0.5.4
```

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

## Train
Use the following command to train Fast-ACVNet+ or Fast-ACVNet on Scene Flow

Firstly, train attention weights generation network for 24 epochs,
```
python main_sceneflow.py --attention_weights_only True
```
Secondly, train complete network for another 24 epochs,
```
python main_sceneflow.py
```
Use the following command to train Fast-ACVNet+ or Fast-ACVNet on KITTI,
```
python main_kitti.py
```


Use the following command to train ACVNet on KITTI (using pretrained model on Scene Flow)
```
python main_kitti.py
```

## Submitted to KITTI benchmarks
```
python save_disp.py
```

# Evaluation on Scene Flow and KITTI

| Method | Scene Flow <br> (EPE) | KITTI 2012 <br> (3-all) | KITTI 2015 <br> (D1-all) | Runtime (ms) |
|---|---|---|---|---|
| Fast-ACVNet+ | 0.59 | 1.85 % | 2.01 % | 45 |
| HITNet | - | 1.89 % |1.98 % | 54 |
| CoEx | 0.69 | 1.93 % | 2.13 % | 33 |
| BGNet+ |  - | 2.03 % | 2.19 % | 35 |
| AANet |  0.87 | 2.42 % | 2.55 % | 62 |
| DeepPrunerFast | 0.97 | - | 2.59 % | 50 |

Our Fast-ACVNet+ achieves comparable accuracy with HITNet on KITTI 2012 and KITTI 2015

### Pretrained Model

[Fast-ACVNet](https://drive.google.com/drive/folders/1vLt_9W3F2K-MciV8Pmv8iRpU24lkMXGo?usp=share_link)

[Fast-ACVNet+](https://drive.google.com/drive/folders/1lcyzoKlkYoDL3tiPGCR6nob9WsusaTI8?usp=share_link)

# Qualitative results on Scene Flow.

![image](https://github.com/gangweiX/Fast-ACVNet/blob/main/imgs/sceneflow.png)

# Qualitative results on KITTI.

![image](https://github.com/gangweiX/Fast-ACVNet/blob/main/imgs/kitti.png)

# Generalization performance on the Middlebury 2014 dataset. All the comparison methods are only trained on Scene Flow without data augmentation.

![image](https://github.com/gangweiX/Fast-ACVNet/blob/main/imgs/middlebury.png)

# Citation

If you find this project helpful in your research, welcome to cite the paper.

```
@article{xu2022accurate,
  title={Accurate and Efficient Stereo Matching via Attention Concatenation Volume},
  author={Xu, Gangwei and Wang, Yun and Cheng, Junda and Tang, Jinhui and Yang, Xin},
  journal={arXiv preprint arXiv:2209.12699},
  year={2022}
}

```

# Acknowledgements

Thanks to Antyanta Bangunharcana for opening source of his excellent work [Correlate-and-Excite](https://github.com/antabangun/coex). Thanks to Xiaoyang Guo for opening source of his excellent work [GwcNet](https://github.com/xy-guo/GwcNet).
