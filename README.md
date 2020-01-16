# MobileUnet is a lightweight UNet implementation

MobileUnet is an architecture that uses depth-wise separable convolutions to build lightweight UNet, using Keras API. It's inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/), [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) and the very clean [Unet Implementation](https://github.com/zhixuhao/unet).

---
## How to use

### Dependencies

It depends on the following libraries:

* Tensorflow
* Keras >= 2.2

and it has depnedency on UNET data generator and skimage if you want to run the demo. 
### Clone the repo with submodules

git clone --recurse-submodules https://github.com/iamyb/mobileunet.git  


## Performance  
The inference latency comparison between MobileUnet and Unet, which are tested only on Intel Xeon CPU E5-2680.   

| Model |Total Parameters| 1 CPU Cores | 2 CPU Cores | 4 CPU Cores | 8 CPU Cores | 16 CPU Cores |  
|---|---|---|---|---|---|---|  
|mobileunet|9,488,462|1320ms|846ms|632ms|417ms|292ms|  
|unet|31,031,685|2790ms|1830ms|1290ms|829ms|554ms|  


### Open Issues

There is a discussion about the performance on GPU https://github.com/tensorflow/tensorflow/issues/12132


