# Asymmetric SiamPolar: Single Shot Video Object Segmentation with Polar Representation

This is an implement of Asymmetric SiamPolarMask on [mmdetection](https://github.com/open-mmlab/mmdetection). 

![siam_polarmask_pipeline](./imgs/siam_polarmask_pipeline.png)

Figure1.SiamPolar

## Highlights

- **Polar Representation:** We introduce polar coordinate represented mask into Video Object Segmentation, which reaches 67.1% J(mean) and 53.1fps in DAVIS2016. Besides, we design a correlation module using cross-correlation to reduce the number of network parameters other than depth-correlation. 
- **Asymmetric Siamese Network**: Most Siamese Networks use the same shadow backbone for both research images and template images, which lead to the unaligned scaled problems. However, research images and reference images are usually not the same sizes leading the backbone cannot pay attention to the same field on them. 
- **FPN:** To make use of different scales of features, we use FPN(Feature Pyramid Network). According to our experiments, FPN is more efficient than other popular feature fusion methods in SiamPolar. 

## Performances

<img src="./imgs/car.gif" alt="demo" style="zoom:50%;" />

<img src="./imgs/bear.gif" alt="demo" style="zoom:50%;" />

On DAVIS-2016: 

| Backbone             | J(M) | J(O) | J(D) | F(M) | F(O) | F(D) | Speed/fps |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | --------- |
| ResNet-50            | 60.5 | 77.3 | 1.7  | 44.0 | 35.3 | 13.9 | 33.20     |
| ResNet-101           | 53.3 | 65.8 | -2.5 | 36.2 | 23.8 | 15.5 | 26.40     |
| Asymmetric ResNet101 | 67.4 | 90.3 | 0.08 | 50.2 | 43.3 | 12.3 | 52.40     |

## Setup Environment

Our SiamPolarMask is based on [mmdetection](https://github.com/open-mmlab/mmdetection). It can be installed easily as following, and more details can be seen in `./INSTALL.md`.

```shell
git clone ...
cd SiamPolarMask
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
SiamPolarMask
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
python tools/train.py ./configs/siam_polarmask/siampolar_r50.py --gpus 1

python tools/test.py ./configs/siam_polarmask/siampolar_r50.py \
./work_dirs/siam_polarmask_r50/epoch_12.pth \
--out ./work_dirs/siam_polarmask_r50/res.pkl \
--eval vos
```

It is very convenient to change the backbones in `./mmdet/model/backbones/siamese.py`.

## Demo

```
cd ./demo
python visualize_vos.py
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 
