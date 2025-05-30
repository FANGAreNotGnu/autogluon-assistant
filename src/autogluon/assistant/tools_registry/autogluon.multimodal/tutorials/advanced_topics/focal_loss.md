Summary: This tutorial demonstrates implementing Focal Loss in AutoGluon's MultiModalPredictor to handle class imbalance problems. It provides specific code for configuring Focal Loss parameters (alpha, gamma, reduction) and calculating class weights from imbalanced data distributions. The tutorial helps with tasks involving imbalanced classification, particularly multiclass problems, by showing how to properly weight classes and focus on hard samples. Key features covered include custom loss function configuration, class weight calculation, and integration with AutoGluon's MultiModalPredictor, making it valuable for developers working on imbalanced dataset challenges.

# Handling Class Imbalance with AutoMM - Focal Loss

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/advanced_topics/focal_loss.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/advanced_topics/focal_loss.ipynb)


In this tutorial, we introduce how to use focal loss with the AutoMM package for balanced training.
Focal loss is first introduced in this [Paper](https://arxiv.org/abs/1708.02002)
and can be used for balancing hard/easy samples as well as un-even sample distribution among classes. This tutorial demonstrates how to use focal loss.

## Create Dataset
We use the shopee dataset for demonstration in this tutorial. Shopee dataset contains 4 classes and has 200 samples each in the training set.


```python
!pip install autogluon.multimodal

```


```python
from autogluon.multimodal.utils.misc import shopee_dataset

download_dir = "./ag_automm_tutorial_imgcls_focalloss"
train_data, test_data = shopee_dataset(download_dir)
```

For the purpose of demonstrating the effectiveness of Focal Loss on imbalanced training data, we artificially downsampled the shopee 
training data to form an imbalanced distribution.


```python
import numpy as np
import pandas as pd

ds = 1

imbalanced_train_data = []
for lb in range(4):
    class_data = train_data[train_data.label == lb]
    sample_index = np.random.choice(np.arange(len(class_data)), size=int(len(class_data) * ds), replace=False)
    ds /= 3  # downsample 1/3 each time for each class
    imbalanced_train_data.append(class_data.iloc[sample_index])
imbalanced_train_data = pd.concat(imbalanced_train_data)
print(imbalanced_train_data)

weights = []
for lb in range(4):
    class_data = imbalanced_train_data[imbalanced_train_data.label == lb]
    weights.append(1 / (class_data.shape[0] / imbalanced_train_data.shape[0]))
    print(f"class {lb}: num samples {len(class_data)}")
weights = list(np.array(weights) / np.sum(weights))
print(weights)
```

## Create and train `MultiModalPredictor`

### Train with Focal Loss
We specify the model to use focal loss by setting the `"optimization.loss_function"` to `"focal_loss"`.
There are also three other optional parameters you can set.

`optimization.focal_loss.alpha` - a list of floats which is the per-class loss weight that can be used to balance un-even sample distribution across classes.
Note that the `len` of the list ***must*** match the total number of classes in the training dataset. A good way to compute `alpha` for each class is to use the inverse of its percentage number of samples.

`optimization.focal_loss.gamma` - float which controls how much to focus on the hard samples. Larger value means more focus on the hard samples.

`optimization.focal_loss.reduction` - how to aggregate the loss value. Can only take `"mean"` or `"sum"` for now.


```python
import uuid
from autogluon.multimodal import MultiModalPredictor

model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee_focal"

predictor = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)

predictor.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optimization.loss_function": "focal_loss",
        "optimization.focal_loss.alpha": weights,  # shopee dataset has 4 classes.
        "optimization.focal_loss.gamma": 1.0,
        "optimization.focal_loss.reduction": "sum",
        "optimization.max_epochs": 10,
    },
    train_data=imbalanced_train_data,
) 

predictor.evaluate(test_data, metrics=["acc"])
```

### Train without Focal Loss


```python
import uuid
from autogluon.multimodal import MultiModalPredictor

model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee_non_focal"

predictor2 = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)

predictor2.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optimization.max_epochs": 10,
    },
    train_data=imbalanced_train_data,
)

predictor2.evaluate(test_data, metrics=["acc"])
```

As we can see that the model with focal loss is able to achieve a much better performance compared to the model without focal loss.
When your data is imbalanced, try out focal loss to see if it brings improvements to the performance!

## Citations

```
@misc{https://doi.org/10.48550/arxiv.1708.02002,
  doi = {10.48550/ARXIV.1708.02002},
  
  url = {https://arxiv.org/abs/1708.02002},
  
  author = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Dollár, Piotr},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Focal Loss for Dense Object Detection},
  
  publisher = {arXiv},
  
  year = {2017},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```


```python

```
