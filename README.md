# Asymmetric SiamPolarMask: Single Shot Video Object Segmentation with Polar Representation

This is an implement of Asymmetric SiamPolarMask on [mmdetection](https://github.com/open-mmlab/mmdetection). 

![siam_polarmask_pipeline](./imgs/siam_polarmask_pipeline.png)

## Highlights

- **Accuracy**: For Siamese Networks, such as SiamFC, SiamMask and so on, they use the same backbone for both research images and refer images. However, research images and reference images are usually not the same sizes, which leads to the backbone cannot pay attention to the same field on them. Thus, we use an **Asymmetric Siamese Network** to solve this problem and get a more accuracy segmentation results. 
- **Speed**: It is a fast method for VOS because of the distance regression of key points in polar coordinates.
- **One-step**: It is a concise method that we track and segment the objects with points, which reduce a great amount of computing. 
- **Initiate without Mask**: The inputs is the bounding boxes of first frame, but not the masks. 

## Performances

<img src="./imgs/car.gif" alt="demo" style="zoom:50%;" />

<img src="./imgs/bear.gif" alt="demo" style="zoom:50%;" />

On DAVIS-2016: 

| Backbone             | J(M) | J(O) | J(D) | F(M) | F(O) | F(D) | Speed/fps |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | --------- |
| ResNet-50            | 60.5 | 77.3 | 1.7  | 44.0 | 35.3 | 13.9 | 33.20     |
| ResNet-101           | 53.3 | 65.8 | -2.5 | 36.2 | 23.8 | 15.5 | 26.40     |
| Asymmetric ResNet101 | 63.3 | 80.6 | 1.9  | 49.1 | 45.1 | 18.6 | 41.60     |

## Setup Environment

Our SiamPolarMask is based on [mmdetection](https://github.com/open-mmlab/mmdetection). It can be installed easily as following. 

```shell
git clone ...
cd SiamPolarMask
conda create -n open_mmlab python=3.7 -y
conda activate open_mmlab

pip install --upgrade pip
pip install cython torch==1.4.0 torchvision==0.5.0 mmcv
pip install -r requirements.txt # ignore the errors
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e . 
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
