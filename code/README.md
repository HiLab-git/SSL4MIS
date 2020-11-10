## Semi-supervised Learning for Medical Image Segmentation (**SSL4MIS**)

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone the repo:
```
git clone https://https://github.com/HiLab-git/SSL4MIS.git 
cd SSL4MIS
```
2. Download the processed data and put the data in `../data/BraTS2019` or `../data/ACDC`, please read and follow the [README](https://github.com/Luoxd1996/SSL4MIS/tree/master/data/).

3. Train the model
```
cd code
python train_unet_3D_XXXXX.py or python train_unet_2D_XXXXX.py or bash train_acdc.sh
```

4. Test the model
```
python test_XXXXX.py
```
# Reimplemented methods
* [Mean Teacher](https://papers.nips.cc/paper/6719-mean-teachers-are-better-role-models-weight-averaged-consistency-targets-improve-semi-supervised-deep-learning-results.pdf)
* [Entropy Minimization](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf)
* [Deep Adversarial Networks](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_47)
* [Uncertainty Aware Mean Teacher](https://arxiv.org/pdf/1907.07034.pdf)
* [Interpolation Consistency Training](https://arxiv.org/pdf/1903.03825.pdf)
## Acknowledgement
* Part of the code is adapted from open-source codebase and original implementations of algorithms, we thank these author for their fantastic and efficient codebase, such as, [UA-MT](https://github.com/yulequan/UA-MT), [Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks) and [segmentatic_segmentation.pytorch](https://github.com/qubvel/segmentation_models.pytorch) . 
