# Improved Trainable Calibration Method for Neural Networks on Medical Imaging Classification

## Introduction
This repository contains the source code and demonstration of ***Improved Trainable Calibration Method for Neural Networks on Medical Imaging Classification*** ([project page](http://www.gb-liang.com/dca.html)), which is accepted to [BMVC2020](https://bmvc2020.github.io). 

In this work, we propose to use DCA as an auxiliary loss term for classification network calibration. DCA integrates network calibration into the classification training stage. Thus, no explicit training round for calibration is required. 

## Requirements 
We recommended the following dependencies.
- Python 3.7
- [PyTorch](https://pytorch.org) 1.4

## Code
- *[demo_dca.ipynb](./demo_dca.ipynb)* shows a demonstartion of the proposed method using the AlexNet backbone and Mendeley V2. 
- *[loss_fn.py](./loss_fn.py)* defines the proposed classificaiton loss, i.e., cross-entropy loss + DCA auxiliary loss
- *[demo_uncalibrated.ipynb](./demo_uncalibrated.ipynb)* shows a demonstartion of the uncalibrated method using the AlexNet backbone and Mendeley V2.

## Reference
If you find this paper or code helpful, please cite this paper:
<br/> 
<br/> 
@inproceedings{liang2020imporved,  
&nbsp;&nbsp;title={Improved Trainable Calibration Method for Neural Networks on Medical Imaging Classification},  
&nbsp;&nbsp;author={Liang, Gongbo and Zhang, Yu and Wang, Xiaoqin and Jacobs, Nathan},  
&nbsp;&nbsp;booktitle={British Machine Vision Conference (BMVC)},  
&nbsp;&nbsp;year={2020} <br/>
}

## Acknowledgements and Disclaimers
*The code is provided for academic purposes only without any guarantees.* <br />
*The Mendeley V2 dataset can be downloaded [here](https://www.kaggle.com/andrewmvd/pediatric-pneumonia-chest-xray).* <br />
*Part of the code that is used in this repo. is based on [temperature_scaling](https://github.com/gpleiss/temperature_scaling)* <br />
*For more detail of temperature scaling, pleae visit their [project page](https://geoffpleiss.com/nn_calibration)* <br />
