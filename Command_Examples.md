# Commands Examples

Here are some command examples. 

Please set them to your own configures. 

```bash
# DAVIS2016
python tools/train.py ./configs/siampolar/siampolar_r101.py --gpus 1 --resume_from ./work_dirs/polar_r101/epoch_24.pth
python tools/test.py ./configs/siampolar/siampolar_r101.py ./work_dirs/polar_r101/epoch_36.pth \
--out ./work_dirs/trash/res.pkl \
--eval vos

# DAVIS2016_GCN
python tools/train.py ./configs/siampolar/siampolar_r101_gcn.py --gpus 1 --resume_from ./work_dirs/trash/epoch_2.pth
python tools/test.py ./configs/siampolar/siampolar_r101_gcn.py ./work_dirs/polar_gcn/epoch_36.pth \
--out ./work_dirs/trash/res.pkl \
--eval vos

# DAVIS2016_light
python tools/train.py ./configs/siampolar/siampolar_r101_light.py --gpus 1
python tools/test.py ./configs/siampolar/siampolar_r101_light.py ./work_dirs/light/epoch_24.pth \
--out ./work_dirs/light/res.pkl \
--eval vos

# TSD-max
python tools/train.py ./configs/siampolar/siampolar_r101_tsd-max.py --gpus 1
python tools/test.py ./configs/siampolar/siampolar_r101_tsd-max.py ./work_dirs/tsd_max/epoch_36.pth \
--out ./work_dirs/tsd_max/res.pkl \
--eval vos

# SegTrack
python tools/train.py ./configs/siampolar/siampolar_r101_segtrack.py --gpus 1
python tools/test.py ./configs/siampolar/siampolar_r101_segtrack.py ./work_dirs/segtrack/epoch_36.pth \
--out ./work_dirs/segtrack/res.pkl \
--eval vos

# SegTrack v2
python tools/train.py ./configs/siampolar/siampolar_r101_segtrackv2.py --gpus 1
python tools/test.py ./configs/siampolar/siampolar_r101_segtrackv2.py ./work_dirs/segtrackv2/epoch_36.pth \
--out ./work_dirs/segtrackv2/res.pkl \
--eval vos
```

