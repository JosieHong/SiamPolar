# SiamPolarMask: Single Shot Video Object Segmentation with Polar Representation

This is an implement of SiamPolarMask on [mmdetection](https://github.com/open-mmlab/mmdetection). 

![siam_polarmask_pipeline](./imgs/siam_polarmask_pipeline.png)

## Highlights

- **One-step**: It is a concise method that we track and segment the objects with points, which reduce a great amount of computing. 
- **Initiate without Mask**: The inputs is the bounding boxes of first frame, but not the masks. 
- **Speed**: It is a fast method for VOS because of the distance regression of key points in polar coordinates.

## Performances

<img src="./imgs/car.gif" alt="demo" style="zoom:50%;" />

<img src="./imgs/bear.gif" alt="demo" style="zoom:50%;" />

On DAVIS-2016: 

| Backbone   | J(M) | J(O) | J(D) | F(M) | F(O) | F(D) | Speed/fps |
| ---------- | ---- | ---- | ---- | ---- | ---- | ---- | --------- |
| ResNet-50  | 48.6 | 56.0 | 20.8 | 34.3 | 21.2 | 20.0 | 48.0      |
| ResNet-101 | 51.1 | 57.9 | 7.0  | 34.6 | 21.4 | 19.2 | 37.6      |

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
mmdetection
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

It can be trained and test as other mmdetection models. For more details, you can read [mmdetection-manual](https://mmdetection.readthedocs.io/en/latest/INSTALL.html) and [mmcv-manual](https://mmcv.readthedocs.io/en/latest/image.html).

```shell
python tools/train.py ./configs/siam_polarmask/siampolar_r50.py --gpus 1

python tools/test.py ./configs/siam_polarmask/siampolar_r50.py \
./work_dirs/siam_polarmask_r50/epoch_12.pth \
--out ./work_dirs/siam_polarmask_r50/res.pkl \
--eval segm
```

## Demo

```
cd ./demo
python visualize_vos.py
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 
