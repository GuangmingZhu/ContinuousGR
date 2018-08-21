# ContinuousGR

## Prerequisites
1) Tensorflow-1.2 <br/>
#### The files about the proposed balanced squared hinge loss function is in the dir tfkeras, replace the original files in contrib/keras/python/keras/ of TF-1.2 with the files in the dir tfkeras. <br/> <br/>
   
## Get the pretrained models
The trained models can be obtained from the below link:  <br/>
    Link: https://pan.baidu.com/s/1pKGwBAb Password: ci7j <br/>

## How to use the code
### Prepare the data
1) Convert each video files into images.
2) Replace the path "/ssd/dataset" in the files under "dataset_splits" 
### Training 
1) Use training_*.py to finetune the networks for different modalities. <br/>
### Testing 
1) Use testing_*.py to extract features or segmentation probability scores. <br/>

## Citation
Please cite the following paper if you feel this repository useful. <br/>
http://ieeexplore.ieee.org/abstract/document/7880648/
http://openaccess.thecvf.com/content_ICCV_2017_workshops/w44/html/Zhang_Learning_Spatiotemporal_Features_ICCV_2017_paper.html
```
@article{ZhuTMM2018,
  title={Continuous Gesture Segmentation and Recognition using 3DCNN and Convolutional LSTM},
  author={Liang Zhang and Guangming Zhu and Peiyi Shen and Juan Song and Syed Afaq Shah and Mohammed Bennamoun},
  journal={IEEE Transactions on Multimedia},
  year={2018}
}
@article{ZhuICCV2017,
  title={Learning Spatiotemporal Features using 3DCNN and Convolutional LSTM for Gesture Recognition},
  author={Liang Zhang and Guangming Zhu and Peiyi Shen and Juan Song and Syed Afaq Shah and Mohammed Bennamoun},
  journal={ICCV},
  year={2017}
}
@article{Zhu2017MultimodalGR,
  title={Multimodal Gesture Recognition Using 3-D Convolution and Convolutional LSTM},
  author={Guangming Zhu and Liang Zhang and Peiyi Shen and Juan Song},
  journal={IEEE Access},
  year={2017},
  volume={5},
  pages={4517-4524}
}
```

## Contact
For any question, please contact
```
  gmzhu@xidian.edu.cn
```

