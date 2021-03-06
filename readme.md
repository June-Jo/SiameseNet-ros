# The ROS Package of SiameseNet

This is a ROS package of [Siamesenet](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) algorithm.

The package contains ROS node of SiameseNet with service-based ROS interface.

Some codes are from [here](https://github.com/adambielski/siamese-triplet).

## Requirements
* ROS kinetic/Melodic
* Pytorch

## ROS Interfaces
 
### Services

* `siameseNet`

    Containing 2 images to compare. After comparing, it returns similarity.

## Getting Started

1. Clone this repository to your catkin workspace, build workspace and source devel environment 
2. Download the [trained model](https://koreaoffice-my.sharepoint.com/:u:/g/personal/jhj0630_korea_edu/Eedowb8mGyRDgFAJ3_a__JcB4WfatFdCVYlb7o2QyvNz7A?e=HYn5NO) and locate it at proper path.
3. Launch siameseNet.launch
4. Run imgClient.py

## Example

$ roslaunch siamesenet siameseNet.launch

$ rosrun siamesenet imgClient.py

![example1](doc/example.PNG)
