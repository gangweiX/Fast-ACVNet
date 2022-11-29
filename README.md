# Fast-ACVNet

Paper:[[arxiv]](https://arxiv.org/pdf/2209.12699.pdf)

Part of the code has already been uploaded.

# Demo on KITTI raw data

A demo result on our RTX 3090 (Ubuntu 20.04).

<p align="center">
  <img width="844" height="446" src="./demo/kittiraw_demo.gif" data-zoomable>
</p>

# Evaluation on Scene Flow and KITTI

| Method | Scene Flow <br> (EPE) | KITTI 2012 <br> (3-all) | KITTI 2015 <br> (D1-all) | Runtime (ms) |
|---|---|---|---|---|
| Fast-ACVNet+ | 0.59 | 1.85 % | 1.90 % | 45 |
| HITNet | - | 1.89 % |1.98 % | 54 |
| CoEx | 0.69 | 1.93 % | 2.13 % | 33 |
| BGNet+ |  - | 2.03 % | 2.19 % | 35 |
| AANet |  0.87 | 2.42 % | 2.55 % | 62 |
| DeepPrunerFast | 0.97 | - | 2.59 % | 50 |

Our Fast-ACVNet+ outperforms all the published real-time methods on Scene Flow, KITTI 2012 and KITTI 2015

### Pretrained Model

[Scene Flow](https://drive.google.com/file/d/1FWr4QCDe0e325OOviorvZ7lpORmnqTUM/view?usp=share_link)
[KITTI 2012](https://drive.google.com/file/d/1Cq_TQpuT0JF6-jq8DUjLvF0qQqxTZkJ-/view?usp=share_link) 
[KITTI 2015](https://drive.google.com/file/d/17X5hUzRWpxbHdlMDiIAJyG-VByCH59Sz/view?usp=share_link) 
[Generalization](https://drive.google.com/file/d/1VUJR11tVgYiLpUsDYET6NCTgU1QGWtn7/view?usp=share_link) 

# Qualitative results on Scene Flow.

![image](https://github.com/gangweiX/Fast-ACVNet/blob/main/imgs/sceneflow.png)

# Qualitative results on KITTI.

![image](https://github.com/gangweiX/Fast-ACVNet/blob/main/imgs/kitti.png)

# Generalization performance on the Middlebury 2014 dataset. All the comparison methods are only trained on Scene Flow.

![image](https://github.com/gangweiX/Fast-ACVNet/blob/main/imgs/middlebury.png)
