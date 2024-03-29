# Chromosome Semantic Segmentation

<details>
    <summary>Citation</summary>
    
If you use the proposed model in your research or wish to refer to the results published, please use the following BibTeX entry.

    @article{altinsoy2021improved,
        title={An improved denoising of G-banding chromosome images using cascaded CNN and binary classification network},
        author={Altinsoy, Emrecan and Yang, Jie and Tu, Enmei},
        journal={The Visual Computer},
        pages={1--14},
        year={2021},
        publisher={Springer}
    }
</details>
<!--  -->

<details>
<summary>Setup Environment</summary>

## Cuda and cuDNN Version

- Cuda 10.2
- Cudnn v8.1.0.77

## Create Conda Environment and Install Libraries 
```bash
conda create -n <your_env_name> python=3.8
```

```bash
activate <your_env_name>
```

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

```bash
conda install -c anaconda pillow scikit-image scikit-learn imageio pandas seaborn
```

```bash
conda install -c conda-forge pytorch-model-summary tensorflow==1.15
```

```bash
pip install ptflops
```

## Cloud TPU Usage
To use cloud TPU run the code below to install torch_xla before running the tpu_train.py and tpu_inference.py files
```bash
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl
```
</details>
<!--  -->

<details>
<summary>Dataset Preparation</summary>

### Classification data preparation
Before running classification_train.py, classification data needs to be created. To create the classification dataset, you need to run classification_data_prep.py file. Run the following command to see the details of the arguments.

```
python3 ./src/classification_data_prep.py -h 
```

### Example
```
python3 ./src/classification_data_prep.py --model=preactivation_resunet --weight-path=preactivation_resunet-20211012T1603 --weight-num=40 --images='datasets/raw_chromosome_data/'
```

</details>
<!--  -->

<details>
<summary>Training</summary>


### Segmentation arguments
```
usage: ./src/segmentation_train.py [-h] [--model MODEL] [--pretrained PRETRAINED]
                               [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                               [--lr LR] [--device DEVICE] [--workers WORKERS]
                               [--weights WEIGHTS] [--test TEST] [--logs LOGS]
                               [--images IMAGES] [--image-size IMAGE_SIZE]
                               [--init-features INIT_FEATURES]

Semantic segmentation of G-banding chromosome Images

optional arguments:
  -h, --help                        show this help message and exit
  --model MODEL                     choose model to train
  --pretrained PRETRAINE            is the backbone pretrained or not
  --batch-size BATCH_SIZ            input batch size for training (default: 2)
  --epochs EPOCHS                   number of epochs to train (default: 40)
  --lr LR                           initial learning rate (default: 0.0001)
  --device DEVICE                   device for training (default: cuda:0)
  --workers WORKERS                 number of workers for data loading (default: 0)
  --weights WEIGHTS                 folder to save weights
  --test TEST                       folder to save weights
  --logs LOGS                       folder to save logs
  --images IMAGES                   dataset folder directory
  --image-size IMAGE_SIZE           target input image size (default: 480x640)
  --init-features INIT_FEATURES     init features for unet, resunet, preact-resunet

available models:
    unet, resunet, preactivation_resunet, cenet, segnet, nested_unet, 
    attention_unet, fcn_resnet101, deeplabv3_resnet101, pspnet
```

### Classification arguments
```
usage: classification_train.py [-h] [--batch-size BATCH_SIZE]
                                      [--epochs EPOCHS] [--lr LR]
                                      [--device DEVICE] [--workers WORKERS]
                                      [--weights WEIGHTS] [--logs LOGS]
                                      [--train-data TRAIN_DATA]
                                      [--validation-data VALIDATION_DATA]

Classification of chromosome and non-chromosome objects

optional arguments:
  -h, --help                            show this help message and exit
  --batch-size BATCH_SIZE               input batch size for training (default: 20)
  --epochs EPOCHS                       number of epochs to train (default: 100)
  --lr LR                               initial learning rate (default: 0.0001)
  --device DEVICE                       device for training (default: cuda:0)
  --workers WORKERS                     number of workers for data loading (default: 0)
  --weights WEIGHTS                     folder to save weights
  --logs LOGS                           folder to save logs
  --train-data TRAIN_DATA               directory of training data
  --validation-data VALIDATION_DATA     directory of validation data
```

### Segmentation training example
```
python3 ./src/segmentation_train.py --images='datasets/raw_chromosome_data/' --model='unet' --epochs=50 --batch-size=4
```
### Classification training example
```
python3 ./src/classification_train.py --train-data=datasets/binary_classification_data/train_data.csv --validation-data=datasets/binary_classification_data/valid_data.csv --epochs=100 --batch-size=40
```
</details>
<!--  -->

<details>
<summary>Results</summary>

|  |  |
|:-----------------------:|:-----------------------:|
| ![loss](./assets/losses/loss.png) | ![val_loss](./assets/losses/val_loss.png) |

## Overall Evaluation Metrics
|          Model          |  Params |     MACs    | Dice (%) |  Se (%) |  Sp (%) | Pre (%) | Acc (%) |
|:-----------------------:|:-------:|:-----------:|:--------:|:-------:|:-------:|:-------:|:-------:|
|          U-Net          |  7.76 M |  64.24 GMac |  99.4998 | 99.5379 | 99.7306 | 99.4616 | 99.6664 |
|      Residual U-Net     |  8.11 M |  67.39 GMac |  99.7657 | 99.7649 | 99.8833 | 99.7666 | 99.8438 |
|         U-net++         |  9.05 M | 158.91 GMac |  98.9878 | 98.1763 | 99.9079 | 99.8128 | 99.3307 |
|     Attention U-net     | 34.88 M | 312.04 GMac |  99.6272 | 99.6367 | 99.8088 | 99.6177 | 99.7515 |
|          CE-Net         |  29.0 M |  41.89 GMac |  99.7414 | 99.7416 | 99.8706 | 99.7412 | 99.8276 |
|          SegNet         | 29.44 M | 188.33 GMac |  99.7415 | 99.7511 | 99.8659 | 99.7319 | 99.8277 |
|    FCN8s (Resnet-101)   | 51.94 M | 253.85 GMac |  99.3271 | 99.3324 | 99.6609 | 99.3218 | 99.5514 |
| DeepLab v3 (Resnet-101) | 58.63 M | 283.44 GMac |  99.2612 | 99.2848 | 99.6186 | 99.2377 | 99.5074 |
|   PSPNet (Resnet-101)   | 72.31 M | 327.25 GMac |  99.3541 | 99.3847 | 99.6615 | 99.3234 | 99.5693 |
|       Proposed CNN      |  8.11 M |  67.44 GMac |  99.7836 |  99.787 | 99.8901 | 99.7802 | 99.8557 |

## Threshold DSC Scores (Grid Search, Search Best)

|  |  |
|:-----------------------:|:-----------------------:|
| ![](./assets/threshold_figures/unet.png) | ![](./assets/threshold_figures/resunet.png) |
| ![](./assets/threshold_figures/nested_unet.png) | ![](./assets/threshold_figures/attention_unet.png) |
| ![](./assets/threshold_figures/cenet.png) | ![](./assets/threshold_figures/segnet.png) |
| ![](./assets/threshold_figures/fcn_resnet101.png) | ![](./assets/threshold_figures/deeplabv3_resnet101.png) |
| ![](./assets/threshold_figures/pspnet.png) | ![](./assets/threshold_figures/proposed_cnn.png) |

## Final Evaluation Metrics
|            Model            | Dice (%) |  Se (%) |  Sp (%) | Pre (%) | Acc (%) |
|:---------------------------:|:--------:|:-------:|:-------:|:-------:|:-------:|
| Local Adaptive Thresholding |  76.1106 | 71.8094 | 99.3036 | 84.3697 | 97.5273 |
|      Histogram Analysis     |  81.3935 | 79.8560 | 99.3562 | 85.4999 | 98.1877 |
|         U-Net (t=18)        |  98.1012 | 98.1355 | 99.9144 |  98.072 | 99.8397 |
|    Residual U-Net (t=187)   |  97.9031 | 97.8826 |  99.908 | 97.9306 | 99.8227 |
|       U-Net++ (t=171)       |  98.084  | 98.1604 | 99.9115 | 98.0139 | 99.8378 |
|   Attention U-Net (t=252)   |  97.8263 | 97.6727 | 99.9116 | 97.9905 | 99.8163 |
|        CE-Net (t=190)       |  97.677  | 97.6342 |  99.899 | 97.7274 | 99.8039 |
|        SegNet (t=173)       |  97.6706 | 97.6965 | 99.8959 | 97.6528 | 99.8033 |
|        FCN8s (t=109)        |  93.3623 | 92.2031 | 99.7601 | 94.5663 | 99.4357 |
|      DeepLab v3 (t=124)     |  93.1646 | 91.8936 |  99.757 | 94.4835 | 99.4186 |
|        PSPNet (t=180)       |  95.2566 | 95.1396 | 99.7966 |  95.386 | 99.6006 |
|     Proposed CNN (t=131)    |  98.0006 | 97.9824 | 99.9121 | 98.0316 |  99.832 |
|      Proposed CNN + BCN     |  98.735  | 98.6783 | 99.9467 | 98.7918 | 99.8931 |

## Comparisons
| Method |          Image 1          |  Image 2 |     Image 3    |  Image 4 |
|:-----------------------:|:-----------------------:|:-------:|:-----------:|:--------:|
| Original Image | ![](./assets/comparison/original_images/0_image.png) | ![](./assets/comparison/original_images/10_image.png) | ![](./assets/comparison/original_images/12_image.png) | ![](./assets/comparison/original_images/18_image.png) |
| Local Adaptive Thresholding |![](./assets/comparison/adaptive_thresholding/0_adaptive_gaussian.png) | ![](./assets/comparison/adaptive_thresholding/10_adaptive_gaussian.png) | ![](./assets/comparison/adaptive_thresholding/12_adaptive_gaussian.png) | ![](./assets/comparison/adaptive_thresholding/18_adaptive_gaussian.png) |
| Histogram Analysis |![](./assets/comparison/histogram_analysis/0_chromosome.png) | ![](./assets/comparison/histogram_analysis/10_chromosome.png) | ![](./assets/comparison/histogram_analysis/12_chromosome.png) | ![](./assets/comparison/histogram_analysis/18_chromosome.png) |
| U-Net (t=18) |![](./assets/comparison/unet/0_unet.png) | ![](./assets/comparison/unet/10_unet.png) | ![](./assets/comparison/unet/12_unet.png) | ![](./assets/comparison/unet/18_unet.png) |
| Residual U-Net (t=187) |![](./assets/comparison/resunet/0_resunet.png) | ![](./assets/comparison/resunet/10_resunet.png) | ![](./assets/comparison/resunet/12_resunet.png) | ![](./assets/comparison/resunet/18_resunet.png) |
| U-Net++ (t=171) |![](./assets/comparison/nested_unet/0_nested_unet.png) | ![](./assets/comparison/nested_unet/10_nested_unet.png) | ![](./assets/comparison/nested_unet/12_nested_unet.png) | ![](./assets/comparison/nested_unet/18_nested_unet.png) |
| Attention U-Net (t=252) |![](./assets/comparison/attention_unet/0_attention_unet.png) | ![](./assets/comparison/attention_unet/10_attention_unet.png) | ![](./assets/comparison/attention_unet/12_attention_unet.png) | ![](./assets/comparison/attention_unet/18_attention_unet.png) |
| CE-Net (t=190) |![](./assets/comparison/cenet/0_cenet.png) | ![](./assets/comparison/cenet/10_cenet.png) | ![](./assets/comparison/cenet/12_cenet.png) | ![](./assets/comparison/cenet/18_cenet.png) |
| SegNet (t=173) |![](./assets/comparison/segnet/0_segnet.png) | ![](./assets/comparison/segnet/10_segnet.png) | ![](./assets/comparison/segnet/12_segnet.png) | ![](./assets/comparison/segnet/18_segnet.png) |
| FCN8s (t=109) |![](./assets/comparison/fcn_resnet101/0_fcn_resnet101.png) | ![](./assets/comparison/fcn_resnet101/10_fcn_resnet101.png) | ![](./assets/comparison/fcn_resnet101/12_fcn_resnet101.png) | ![](./assets/comparison/fcn_resnet101/18_fcn_resnet101.png) |
| DeepLab v3 (t=124) |![](./assets/comparison/deeplabv3_resnet101/0_deeplabv3_resnet101.png) | ![](./assets/comparison/deeplabv3_resnet101/10_deeplabv3_resnet101.png) | ![](./assets/comparison/deeplabv3_resnet101/12_deeplabv3_resnet101.png) | ![](./assets/comparison/deeplabv3_resnet101/18_deeplabv3_resnet101.png) |
| PSPNet (t=180) |![](./assets/comparison/pspnet/0_pspnet.png) | ![](./assets/comparison/pspnet/10_pspnet.png) | ![](./assets/comparison/pspnet/12_pspnet.png) | ![](./assets/comparison/pspnet/18_pspnet.png) |
| Proposed CNN (t=131) |![](./assets/comparison/proposed_cnn/0_proposed_cnn.png) | ![](./assets/comparison/proposed_cnn/10_proposed_cnn.png) | ![](./assets/comparison/proposed_cnn/12_proposed_cnn.png) | ![](./assets/comparison/proposed_cnn/18_proposed_cnn.png) |
| Proposed CNN+BCN |![](./assets/comparison/proposed_cnn_bcn/0_proposed_cnn_bcn.png) | ![](./assets/comparison/proposed_cnn_bcn/10_proposed_cnn_bcn.png) | ![](./assets/comparison/proposed_cnn_bcn/12_proposed_cnn_bcn.png) | ![](./assets/comparison/proposed_cnn_bcn/18_proposed_cnn_bcn.png) |

</details>

