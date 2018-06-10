## PyTorch Face Recognizer based on 'VGGFace2: A dataset for recognising faces across pose and age'.

This repo implements training and testing models, and feature extractor based on models for VGGFace2 [1].
Pretrained models for PyTorch are converted from [Caffe models](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) 
authors of [1] provide.

### Dataset

To download VGGFace2 dataset, see [authors' site](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).

### Pretrained models

The followings are PyTorch models converted from [Caffe models](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) authors of [1] provide.

|arch_type|download link|
| :--- | :---: |
|`resnet50_ft`|[link](https://drive.google.com/open?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU)|
|`senet50_ft`|[link](https://drive.google.com/open?id=1YtAtL7Amsm-fZoPQGF4hJBC9ijjjwiMk)|
|`resnet50_scratch`|[link](https://drive.google.com/open?id=1gy9OJlVfBulWkIEnZhGpOLu084RgHw39)|
|`senet50_scratch`|[link](https://drive.google.com/open?id=11Xo4tKir1KF8GdaTCMSbEQ9N4LhshJNP)|

### Extracting features

Usage: 
```bash
python demo.py extract <options>
```

#### Options

* `--arch_type` network architecture type (default: `resnet50_ft`): 
    - `resnet50_ft` ResNet-50 which are first pre-trained on MS1M, and then fine-tuned on VGGFace2
    - `senet50_ft` SE-ResNet-50 trained like `resnet50_ft`
    - `resnet50_scratch` ResNet-50 trained from scratch on VGGFace2
    - `senet50_scratch` SE-ResNet-50 trained like `resnet50_scratch`
* `--weight_file` weight file converted from Caffe model(see [here](#pretrained-models))
* `--resume` checkpoint file used in feature extraction (default: None). If set, `--weight_file` is ignored.
* `--dataset_dir` dataset directory
* `--feature_dir` directory where extracted features are saved
* `--test_img_list_file` image file for which features are extracted
* `--log_file` log file
* `--meta_file` Meta information file for VGGFace2, `identity_meta.csv` in [Meta.tar.gz](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
* `--batch_size` batch size (default: 32)
* `--gpu` GPU devide id (default: 0)
* `--workers` number of data loading workers (default: 4)
* `--horizontal_flip` horizontally flip images specified in `--test_img_list_file`

### Testing

Usage: 
```bash
python demo.py test <options>
```

#### Options

* `--arch_type` network architecture type (default: `resnet50_ft`): 
    - `resnet50_ft` ResNet-50 which are first pre-trained on MS1M, and then fine-tuned on VGGFace2
    - `senet50_ft` SE-ResNet-50 trained like `resnet50_ft`
    - `resnet50_scratch` ResNet-50 trained from scratch on VGGFace2
    - `senet50_scratch` SE-ResNet-50 trained like `resnet50_scratch`
* `--weight_file` weight file converted from Caffe model(see [here](#pretrained-models))
* `--resume` checkpoint file used in test (default: None). If set, `--weight_file` is ignored.
* `--dataset_dir` dataset directory
* `--test_img_list_file` text file containing image files used for validation, test or feature extraction
* `--log_file` log file
* `--meta_file` Meta information file for VGGFace2, `identity_meta.csv` in [Meta.tar.gz](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
* `--batch_size` batch size (default: 32)
* `--gpu` GPU devide id (default: 0)
* `--workers` number of data loading workers (default: 4)


### Training

Usage: 
```bash
python demo.py train <options>
```

#### Options

* `--arch_type` network architecture type (default: `resnet50_ft`): 
    - `resnet50_ft` ResNet-50 which are first pre-trained on MS1M, and then fine-tuned on VGGFace2
    - `senet50_ft` SE-ResNet-50 trained like `resnet50_ft`
    - `resnet50_scratch` ResNet-50 trained from scratch on VGGFace2
    - `senet50_scratch` SE-ResNet-50 trained like `resnet50_scratch`
* `--weight_file` weight file converted from Caffe model(see [here](#pretrained-models)), and used for fine-tuning
* `--resume` checkpoint file used to resume training (default: None). If set, `--weight_file` is ignored.
* `--dataset_dir` dataset directory
* `--train_img_list_file` text file containing image files used for training
* `--test_img_list_file` text file containing image files used for validation, test or feature extraction
* `--log_file` log file
* `--meta_file` Meta information file for VGGFace2, `identity_meta.csv` in [Meta.tar.gz](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
* `--checkpoint_dir` checkpoint output directory
* `--config` number of settings and hyperparameters used in training
* `--batch_size` batch size (default: 32)
* `--gpu` GPU devide id (default: 0)
* `--workers` number of data loading workers (default: 4)

## Note

VGG-Face dataset, described in [2], is not planned to be supported in this repo.
If you are interested in models for VGG-Face, see [keras-vggface](https://github.com/rcmalli/keras-vggface).

## References

1. ZQ. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman,
    VGGFace2: A dataset for recognising faces across pose and age.   
    [site](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/), [arXiv](https://arxiv.org/abs/1710.08092)
    
2. Parkhi, O. M. and Vedaldi, A. and Zisserman, A.,
    Deep Face Recognition.   
    [site](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)