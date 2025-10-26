# ReID with Deep Cosine Metric Learning

## Introduction

This repository contains code for training a metric feature representation in BEV space directly from images. The approach is described in

    @INPROCEEDINGS{Oliveira2025, 
    author={De Oliveira, Bruno Borges and Bispo, Ruan and Grassi, Valdir}, 
    booktitle={2025 Brazilian Conference on Robotics (CROS)}, 
    title={Vehicle Re-identification in BEV Space with Deep Cosine Metric Learning},
    year={2025}, 
    volume={1}, 
    number={}, 
    pages={1-6}, 
    keywords={Space vehicles; Visualization;Shape;Spaceborne radar;Sensor fusion;Feature extraction;Robot sensing systems;Radar tracking;Reliability;Autonomous vehicles;autonomous vehicles;re-identification;adverse conditions;convolutional neural networks}, 
    doi={10.1109/CROS66186.2025.11064866}}

## Creating a docker container
Build image (in this case, the image created will be named pytorch-gpu-cuda11-6):
```
docker build --build-arg uid=$UID --build-arg user=$USER -t pytorch-gpu-cuda11-6 . -f Dockerfile_pytorch
```
Run image with access to external dataset folder:
```
docker run --name creating-reid-dataset --gpus all -it --rm -v $(pwd):/$USER/bev_cosine_metric_learning -v PATH_TO_DATASET_FOLDER:/$USER/bev_cosine_metric_learning/downloads/datasets pytorch-gpu-cuda11-6

```

## Preparing environment
These steps describes how to prepare the environment to run the camera-to-BEV training.

Run tools/setup.py file, that was modified from [BEVFusion](https://github.com/mit-han-lab/bevfusion) to generate .so files from the modules in C language bev_pool and voxel_layer, needed to use [LSS](https://github.com/nv-tlabs/lift-splat-shoot) to convert images to BEV (use install_dir as the path to the folder python3.8/site-packages contained in $PATH in the running docker container):

```
python3 tools/setup.py develop --install-dir ./.local/lib/python3.8/site-packages
```

Install mmcv and mmdet (use openmim):
```
mmcv==2.1.0 
mmdet==3.2.0
```

Install efficientnet-pytorch:
```
pip install efficientnet-pytorch
```

Install [BEVFusion](https://github.com/mit-han-lab/bevfusion) for MMDET3D (follow instructions from github) and copy file builder.py from folder 
```/bevfusion-mmdet/venv/lib/python3.8/site-packages/google/protobuf/internal/builder.py``` to folder ```bev_cosine_metric_learning/.local/lib/python3.8/site-packages/google/protobuf/internal``` in docker container, because the ```protobuf``` version compatible with the libraries in docker is not compatible with the version that contains the file builder.py, but [LSSTransform](https://github.com/nv-tlabs/lift-splat-shoot) needs this library. 

## Training model for camera-to-BEV transformation
The dataset used is extracted from [Nuscenes](https://github.com/nutonomy/nuscenes-devkit) dataset.
To change the category to be used to train the model, change the line 184 in the file ```/tools/lss_transform/src/data.py```. If no category is filtered, all categories will be used in training.
```
            #Filter for vehicle's category
            if ((not inst['category_name'].split('.')[0] == 'vehicle')):
                continue
```

Train the model:
```
python3 tools/lss_transform/main.py train trainval  --dataroot=PATH_TO_DATASET/nuscenes --logdir=./runs/dual_loss_64channels_all --gpuid=0 --nworkers=2 --bsz=2
```

To run the validation for a specific model number for n times, change the model number in from the ones in folder runs (modelxxx.pt) and nvals in the command bellow:
```
python3 tools/lss_transform/main.py val trainval  --dataroot=PATH_TO_DATASET/nuscenes --logdir=./runs/dual_loss_64channels_all modelf=bev_cosine_metric_learning/runs/dual_loss_64channels_all/model290000.pt  --gpuid=0 --nworkers=2 --bsz=2 --nvals=10
```

## Creating ReID dataset from [Nuscenes](https://github.com/nutonomy/nuscenes-devkit)

Uncomment lines 515 and 516 in ``create_nuscenes_reid_dataset.py`` file to ignore iou, if necessary. Adjust ``destination_path`` in line 27. Adjust line 52 to set the desired model trained in first step ``modelf = '/bev_cosine_metric_learning/tools/lss_transform/models/dual_loss_64channel_all/model290000.pt' ``.
Run the command:
```
python3 tools/create_nuscenes_reid_dataset.py
```


## Training on Nusc-ReID dataset

Set in file ``datasets/nuscenes_reid.py`` the MAX_LABEL parameter to the value of the last ID obtained from ``object_data.json`` in the generated dataset folder.

The following command starts training
using the cosine-softmax classifier described in the above paper:
```
python3 train_nuscenes_reid.py \
    --dataset_dir=./downloads/datasets/nuscenes_reid_human_iou20  \
    --loss_mode=cosine-softmax \
    --log_dir=./output/nusc-reid/ \
    --run_id=cosine-softmax
```
This will create a directory `./output/nusc-reid/cosine-softmax` where
TensorFlow checkpoints are stored and which can be monitored using
``tensorboard``:
```
tensorboard --logdir ./output/nusc-reid/cosine-softmax --port 6006
```
The code splits off 10% of the training data for validation.
Concurrently to training, run the following command to run CMC evaluation
metrics on the validation set:
```
python3 train_nuscenes_reid.py \
    --mode=eval \
    --dataset_dir=./downloads/datasets/nuscenes_reid \
    --loss_mode=cosine-softmax \
    --log_dir=./output/nusc-reid/ \
    --run_id=cosine-softmax \
    --eval_log_dir=./eval_output/nusc-reid-human
```
The command will block indefinitely to monitor the training directory for saved
checkpoints and each stored checkpoint in the training directory is evaluated on
the validation set. The results of this evaluation are stored in
``./eval_output/nusc-reid/cosine-softmax`` to be monitored using
``tensorboard``:
```
tensorboard --logdir ./eval_output/nusc-reid/cosine-softmax --port 6007
```
