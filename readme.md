# PGCN for gait recognition
## Introduction
This repository holds the codebase, dataset and models for the paper:

**Gait Identification based on human skeleton with pariwise graph convolutional network** Ke Xu, Xinghao Jiang, Tanfeng Sun, ICME 2021.

1. Download and install st-gcn structure from https://github.com/yysijie/st-gcn
We made revisions based on their structures.
**Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition** Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)

2. Download our files and put them into the right folder

1) Download the casia-skeleton folder and put it into $STGCN PATH$/config/st_gcn/

$STGCN PATH$/config/st_gcn/
  -casia-skeleton/
    -train.yaml
    -test.yaml

2) Download and put $STGCN PATH$/tools/casia_gendata.py
3) Download and put $STGCN PATH$/feeder/feeder_casia.py
4) Download and put $STGCN PATH$/net/st_gcn_casia.py
5) Download and replace $STGCN PATH$/processor/recognition.py
6) Download and replace $STGCN PATH$/processor/processor.py


3. Make data for CASIA-B
1) Download CASIA-B dataset and extract skeleton with OpenPose. The file structure is (view 000 for example):
#train set
-train62-000
    -001-bg-01-000
        -001-bg-01-000_000000000000_keypoints.json
        -001-bg-01-000_000000000001_keypoints.json
        -001-bg-01-000_000000000002_keypoints.json
        ...
    -001-bg-02-000
        -001-bg-02-000_000000000000_keypoints.json
        -001-bg-02-000_000000000001_keypoints.json
        -001-bg-02-000_000000000002_keypoints.json
        ...
    -001-cl-01-000
    ...
#gallery set
-gallery62-000
    -063-nm-01-000
        -063-nm-01-000_000000000000_keypoints.json
        -063-nm-01-000_000000000001_keypoints.json
        ...
    -063-nm-02-000
    -063-nm-03-000
    -063-nm-04-000
    ...
#probe set
-probe62-000bg
    -063-bg-01-000
        -063-bg-01-000_000000000000_keypoints.json
        -063-bg-01-000_000000000001_keypoints.json
        ...
    -063-bg-02-000
    -064-bg-01-000
    ...
-probe62-000cl
    -063-cl-01-000
    -063-cl-02-000
    -064-cl-01-000
    ...
-probe62-000cm
    -063-nm-05-000
    -063-nm-06-000
    -064-nm-05-000
    ...

2)Modify the input path and output path in ./tools/casia_gendata.py. Run python ./tools/casia_gendata.py
3) Modify $STGCN PATH$/config/st_gcn/casia-skeleton/train.yaml
4)Train model
python main.py recognition --phase train -c $STGCN PATH$/config/st_gcn/casia-skeleton/train.yaml
5) Modify $STGCN PATH$/config/st_gcn/casia-skeleton/test.yaml
6) Test
python main.py recognition  --phase test -c $STGCN PATH$/config/st_gcn/casia-skeleton/test.yaml

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{ke2021icme,
  title     = {Gait Identification based on human skeleton with pariwise graph convolutional network},
  author    = {Ke Xu, Xinghao Jiang, Tanfeng Sun},
  booktitle = {ICME},
  year      = {2021},
}