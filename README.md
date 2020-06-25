# SiamPolar: Video Object Segmentation with Polar Representation

This is an implement of Asymmetric SiamPolarMask on [mmdetection](https://github.com/open-mmlab/mmdetection). 

![siam_polarmask_pipeline](./imgs/siam_polarmask_pipeline.png)

Figure1.SiamPolar

## Highlights

- **Polar Representation:** We introduced the polar coordinate represented mask into Video Object Segmentation and proposed SiamPolar. It is also an anchor-free object track method. In DAVIS2016, SiamPolar reaches 69.1% J(mean) and 59.2fps. 
- **Asymmetric Siamese Network**: Most Siamese Networks use the same backbone for both search images and template images leading to the unaligned spatial problems in different scales. To solve this problem, we proposed Asymmetric Siamese Network using similar backbone in different depth, which also the make Siamese Network could use a deeper backbone performing better. 
- **Re-cross Correlation**: We designed a new correlation module using repeated cross-correlation other than depth-wised correlation to reduce parameters and fusion the search features improving the results of segmentation.  
- **Semi-FPN:** To make use of different scales of features, we use Semi-FPN(Feature Pyramid Network). We adapted FPN into Semi-FPN delaying the feature fusion step into SiamPolarHead. In this way, FPN features for classification, centerness, bounding box, and mask are separated, and the antagonism between branches in SiamPolarHead is improved. 

## Performances

<img src="./imgs/car.gif" alt="demo" style="zoom:50%;" />



**Results on DAVIS-2016**

![siam_polar_performance](./imgs/siam_polar_performance.png)

## Setup Environment

Our SiamPolarMask is based on [mmdetection](https://github.com/open-mmlab/mmdetection). It can be installed easily as following, and more details can be seen in `./INSTALL.md`.

```shell
git clone ...
cd SiamPolar
conda create -n open_mmlab python=3.7 -y
source activate open_mmlab

pip install --upgrade pip
pip install cython torch==1.4.0 torchvision==0.5.0 mmcv
pip install -r requirements.txt # ignore the errors
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"

python setup.py develop
# or "pip install -v -e ."
```

## Prepare DAVIS Dataset

1. Download DAVIS from [kaggle-DAVIS480p](https://www.kaggle.com/mrjb166/davis480p).

2. Convert DAVIS to coco format by `/tools/conver_datasets/davis2coco.py` and organized it as following

```shell
SiamPolar
├── mmdet
├── tools
├── configs
├── data
│  ├── DAVIS
│  │  ├── Annotations
|  |  |  ├── 480p_train.json
|  |  |  ├── 480p_trainval.json
|  |  |  ├── 480p_val.json
|  |  |  ├── db_info.yml
│  │  ├── Imageset
│  │  ├── JPEGImages
```

## Train & Test

It can be trained and test as other mmdetection models. For more details, you can read [mmdetection-manual](https://mmdetection.readthedocs.io/en/latest/INSTALL.html) and [mmcv-manual](https://mmcv.readthedocs.io/en/latest/image.html). This is an example of SiamPolarMask(ResNet50 Backbone). 

```shell
python tools/train.py ./configs/siampolar/siampolar_r101.py --gpus 1

python tools/test.py ./configs/siampolar/siampolar_r101.py \./work_dirs/siam_polarmask_r50/epoch_12.pth \
--out ./work_dirs/siam_polarmask_r50/res.pkl \
--eval vos
```

We also add some configures for SiamPolar like backbone type, polar points number and so on, which can be easily set in `./configs/siampolar/`.

## Demo

```
cd ./demo
python visualize_vos.py
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 
