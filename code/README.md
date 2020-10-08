## Semi-supervised Learning for Medical Image Segmentation (**SSL4MIS**)

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone the repo:
```
git clone https://https://github.com/Luoxd1996/SSL4MIS.git 
cd SSL4MIS
```
2. Download the processed data and put the data in ../data/BraTS2019, please read and follow the [README](https://github.com/Luoxd1996/SSL4MIS/tree/master/data/BraTS2019).

3. Train the model
```
cd code
python train_unet_3D_XXXXX.py
```

4. Test the model
```
python test.py
```

## Acknowledgement
* This code is adapted from [UA-MT](https://github.com/yulequan/UA-MT), [Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks). 
