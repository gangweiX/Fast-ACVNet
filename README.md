# Fast-ACVNet

We will release the source code soon

# Demo on KITTI raw data

<p align="center">
  <img width="666" height="333" src="./demo/kittiraw_demo.gif" data-zoomable>
</p>

# Evaluation on Scene Flow and KITTI

| Method | Scene Flow <br> (EPE) | KITTI 2012 <br> (3-all) | KITTI 2015 <br> (D1-all) | Runtime (ms) |
|---|---|---|---|---|
| Fast-ACVNet+ | 0.59 | 1.85 % | 2.01 % | 45 |
| HITNet | - | 1.89 % |1.98 % | 54 |
| CoEx | 0.69 | 1.93 % | 2.13 % | 33 |
| BGNet+ |  - | 2.03 % | 2.19 % | 35 |
| AANet |  0.87 | 2.42 % | 2.55 % | 62 |
| DeepPrunerFast | 0.97 | - | 2.59 % | 50 |

Our Fast-ACVNet+ outperforms all the published real-time methods on Scene Flow, KITTI 2012 and KITTI 2015

# Qualitative results on Scene Flow.

![image](https://github.com/gangweiX/Fast-ACVNet/blob/main/imgs/sceneflow.png)

# Qualitative results on KITTI.

![image](https://github.com/gangweiX/Fast-ACVNet/blob/main/imgs/kitti.png)

# Generalization performance on the Middlebury 2014 dataset. All the comparison methods are only trained on Scene Flow.

![image](https://github.com/gangweiX/Fast-ACVNet/blob/main/imgs/middlebury.png)
