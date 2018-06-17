## PyTorch Face Recognizer based on 'VGGFace2: A dataset for recognising faces across pose and age'.

This repo implements training and testing models, and feature extractor based on models for VGGFace2 [1].

[Pretrained models](#pretrained-models) for PyTorch are converted from [Caffe models](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) 
authors of [1] provide.

### Dataset

To download VGGFace2 dataset, see [authors' site](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).

### Preprocessing images

Faces should be detected and cropped from images before face images are fed to this face recognizer(`demo.py`).

There are several face detection programs based on MTCNN [3].

* PyTorch version: [mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch)
* MXNet version: [mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
* Matlab version: [MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

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
    VGGFace2: A dataset for recognising faces across pose and age, 2018.   
    [site](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/), [arXiv](https://arxiv.org/abs/1710.08092)
    
2. Parkhi, O. M. and Vedaldi, A. and Zisserman, A.,
    Deep Face Recognition, British Machine Vision Conference, 2015.
    [site](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
    
3. K. Zhang and Z. Zhang and Z. Li and Y. Qiao,
   Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks,
   IEEE Signal Processing Letters, 2016. 
   [arXiv](https://arxiv.org/abs/1604.02878)