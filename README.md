# GazeDetectionSim

## Gaze Detection System

The image below shows the architecture of the gaze detection model.

[Architecture](imgs/model_architecture.jpg)

The model is based on the [gaze 350](http://gaze360.csail.mit.edu/) but we replaced the backbone with the [convNext](https://github.com/facebookresearch/ConvNeXt) models. We replace the backbone and retrain the model.
The retrained model has the **Accuracy of 11.6 degrees** on test dataset provided in gaze360 dataset.

## Evaluating the model on frontal faces.

Test involved asking people  to walk in front of the camera at distance of `1m, 1.5m and 2m` while looking at the either one of the three objects present in fron tof them for 1 minute. They were randomly asked to look at different object every 5 seconds.

Refere the image below

![experimental setup](imgs/frontFaceTest.jpg)

Total of 3 subjects were each asked to walk at all three distances and the average of all 3 is recorded below

**Accuracy for each distance**
```
1.0m --- 2.12 %
1.5m --- 2.12 %
2.0m --- 2.12 %
```

## Downloading the weights of the model

The weights of the model can be downloaded with gdown

gdown --id 1pv_kSHkVqar8E3WEqVCZAdoPlFmusHZl on linux
or use this link
[dataset](https://drive.google.com/file/d/1pv_kSHkVqar8E3WEqVCZAdoPlFmusHZl/view?usp=drive_link) 
