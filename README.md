# Deep color calibration
Pytorch implementation of [Object-based attention mechanism for color calibration of UAV remote sensing images in precision agriculture](https://ieeexplore.ieee.org/abstract/document/9963677).

This project aims to build a generic framework for color calibration in agricultural UAV remote sensing 
with unsupervised object-based attention.

------

## Requirements

- Python 3.6
- PyTorch 1.5.1
- TorchVision
- Anaconda environment recommended here!
- GPU environment is required for training and testing


## Usage
------
## Dataset preparation
The dataset used in this study is the public dataset: CropUAV, which includes the UAV imagery of rice, bean, and cotton(CropUAVDataset). The imagery were collected at different sites and different dates, where the ortho-mosaics present siginifcant color cast and color inconsistency. The dataset is available at https://drive.google.com/drive/folders/15TETGMJxQvuBOqjCTGQPu5OTCizF4pPK.
After downloading, the dataset should be placed at the "data" folder under the project, where the training and testing samples were defined in the responding text files.
For more details on our dataset, you can refer to our previous paper [Deep Color Calibration for UAV Imagery in Crop Monitoring Using Semantic Style Transfer with Local to Global Attention](https://www.sciencedirect.com/science/article/pii/S030324342100297X).

During our experiments, one single reference image was selected for the image sequences of one UAV flight. For the experimental results in our paper, the reference images were selected as follows:

2017-9-30:  DJI_0426_4_4.png

2018-10-8:  DJI_0481_3_3.png

2017-7-15-field1:  DJI_0068_4_1.png

2017-7-15-field2:  DJI_0258_1_0.png

## Testing
The script of transfer5_rice.py is used to calibrate the color cast of rice.
The script of transfer5_bean.py is used to calibrate the color cast of bean.
The script of transfer5_cotton.py is used to calibrate the color cast of cotton.
The only difference of these scripts on different crops is the thershold to split the representation into
strong correspondences and weak correspondences, where the threshold value ranges from 150 to 200 under all occasions.

## Training
The proposed object-based attention mechanism directly applied the COCO pretrained weights to all the involved research senarios, which required no extra training.


## References
- [Photorealistic Style Transfer via Wavelet Transforms](https://github.com/clovaai/WCT2)
- [Style Mixer: Semantic-aware Multi-Style Transfer Network](https://github.com/zxhuang97/Official-implementation-StyleMixer)

