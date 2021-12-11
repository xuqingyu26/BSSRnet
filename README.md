# BSSRnet
**PyTorch implementation of "*Deep Bilateral Learning for Stereo Image Super-Resolution*", IEEE Signal Processing Letters.**
<br><br>
## *Highlights:*
#### 1. *We develop a bilateral dynamic network, which conduct space-variable filter on stereo images.*
 <p align="center"> <img src="https://github.com/xuqingyu26/BSSRnet/blob/main/Figs/Overview.png" width="100%"></p>
 
#### 2. *Details of the Refinement Part.*
<p align="center"><img src="https://github.com/xuqingyu26/BSSRnet/blob/main/Figs/Refinement.png" width="100%"></p>
  
#### 3. *Illustration of several kernels in bilateral filters.*
<p align="center"><img src="https://github.com/xuqingyu26/BSSRnet/blob/main/Figs/filter.png" width="100%"></p>

#### 4. *Our BSSR significantly outperforms PASSRnet with a comparable model size.*
<p align="center"><img src="https://github.com/xuqingyu26/BSSRnet/blob/main/Figs/quantatitive.png" width="100%"></p>

## Requirement
* **PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=11.0.**
* **Matlab (For training/test data generation and performance evaluation)**
* **Prepare the train and test data following [this](https://github.com/YingqianWang/iPASSR).**

## Train
* **Download the training sets from [Baidu Drive](https://pan.baidu.com/s/173UGmmN0rtOUghIT40oy8w) (Key: NUDT) and unzip them to `./data/train/`.** 
* **Run `train.py` to perform training. Checkpoint will be saved to  `./log/`.**

## Test
* **Download the test sets and unzip them to `./data/test/`. Here, we provide the full test sets used in our paper on [Google Drive](https://drive.google.com/file/d/1LQDUclNtNZWTT41NndISLGvjvuBbxeUs/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1SIYGcMBEDDZ0wYrkxL9bnQ) (Key: NUDT).** 
* **Run `demo_test.py` to perform a demo inference. Results (`.png` files) will be saved to `./results`.**

## Citiation
**We hope this work can facilitate the future research in stereo image SR. If you find this work helpful, please consider citing:**
```
@article{xu2021deep,
  title={Deep Bilateral Learning for Stereo Image Super-Resolution},
  author={Xu, Qingyu and Wang, Longguang and Wang, Yingqian and Sheng, Weidong and Deng, Xinpu},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={613--617},
  year={2021},
  publisher={IEEE}
}

```
