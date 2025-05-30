Summary: This tutorial provides implementation guidance for preparing the Pascal VOC dataset for object detection tasks using AutoMM. It covers two main implementation approaches: using Python CLI (cross-platform) and Bash scripts (Unix), with specific commands for downloading VOC2007 and VOC2012 datasets. The tutorial helps with dataset setup tasks, including proper directory structure organization and format conversion. Key features include dataset download automation, proper directory structure setup (~8.4GB), and handling both VOC2007 and VOC2012 versions with 20 object categories. It emphasizes COCO format recommendation over VOC format and includes essential information about dataset splits for training and validation.

# AutoMM Detection - Prepare Pascal VOC Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/data_preparation/prepare_voc.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/data_preparation/prepare_voc.ipynb)



[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) is a collection of datasets for object detection. 
The most commonly combination for benchmarking is using VOC2007 trainval and VOC2012 trainval for training and VOC2007 test for validation.
Both VOC2007 and VOC2012 have the same 20 classes, and they have 16551 training images in total.
This tutorial will walk through the steps of preparing both VOC2007 and VOC2012 for Autogluon MultiModalPredictor.

You need 8.4 GB disk space to download and extract this dataset. SSD is preferred over HDD because of its better performance.
The total time to prepare the dataset depends on your Internet speed and disk performance. For example, it often takes 10 min on AWS EC2 with EBS.

VOC has an [official webpage](http://host.robots.ox.ac.uk/pascal/VOC/) to download the data, 
but it's always easier to perform a one-step setup.
We prepared a script to download both VOC2007 and VOC2012 in our examples: 
[download_voc0712.sh](https://raw.githubusercontent.com/autogluon/autogluon/master/examples/automm/object_detection/download_voc0712.sh).
You can also download them separately:
[download_voc07.sh](https://raw.githubusercontent.com/autogluon/autogluon/master/examples/automm/object_detection/download_voc07.sh),
[download_voc12.sh](https://raw.githubusercontent.com/autogluon/autogluon/master/examples/automm/object_detection/download_voc12.sh).
Or you can also use our cli tool `prepare_detection_dataset` that can download all datasets mentioned in our tutorials.
This python script is in our code: 
[prepare_detection_dataset.py](https://raw.githubusercontent.com/autogluon/autogluon/master/multimodal/src/autogluon/multimodal/cli/prepare_detection_dataset.py),
and you can also run it as a cli: `python3 -m autogluon.multimodal.cli.prepare_detection_dataset`.

## Download with Python Script

The python script does not show progress bar, but is promised to work on all major platforms.
If you are working on a Unix system and needs a progress bar, try the bash script!

You could either extract it under current directory by running:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712
```


or extract it under a provided output path:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712 --output_path ~/data
```


or make it shorter:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc -o ~/data
```


or download them separately

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc07 -o ~/data
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc12 -o ~/data
```


## Download with Bash Script

You could either extract it under current directory by running:

```
bash download_voc0712.sh
```


or extract it under a provided output path:

```
bash download_voc0712.sh ~/data
```


The command line output will show the progress bar:

```
extract data in current directory
Downloading VOC2007 trainval ...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  438M  100  438M    0     0  92.3M      0  0:00:04  0:00:04 --:--:-- 95.5M
Downloading VOC2007 test data ...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  430M  100  430M    0     0  96.5M      0  0:00:04  0:00:04 --:--:-- 99.1M
Downloading VOC2012 trainval ...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
 73 1907M   73 1401M    0     0   108M      0  0:00:17  0:00:12  0:00:05  118M

```


And after it finished, VOC datasets are extracted in folder `VOCdevkit`, it contains

```
VOC2007  VOC2012
```


And both of them contains:

```
Annotations  ImageSets  JPEGImages  SegmentationClass  SegmentationObject
```


## The VOC Format
VOC also refers to the specific format (in `.xml` file) the VOC dataset is using.

**In Autogluon MultiModalPredictor, we strongly recommend using COCO as your data format instead.
Check [AutoMM Detection - Prepare COCO2017 Dataset](prepare_coco17.ipynb) and [Convert Data to COCO Format](convert_data_to_coco_format.ipynb) for more information
about COCO dataset and how to convert a VOC dataset to COCO.**

However, for fast proof testing we also have limit support for VOC format.
While using VOC format dataset, the input is the root path of the dataset, and contains at least:

```
Annotations  ImageSets  JPEGImages
```


## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../../advanced_topics/customization.ipynb).

## Citation
```
@Article{Everingham10,
   author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
   title = "The Pascal Visual Object Classes (VOC) Challenge",
   journal = "International Journal of Computer Vision",
   volume = "88",
   year = "2010",
   number = "2",
   month = jun,
   pages = "303--338",
}
```
