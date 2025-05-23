Summary: This tutorial covers AutoGluon's tabular feature engineering implementation, focusing on automatic data type detection and processing for boolean, categorical, numerical, datetime, and text columns. It demonstrates how to implement custom feature processing pipelines, configure data type overrides, and handle automated feature engineering for datetime (extracting year, month, day components) and text data (using either Transformer networks or n-gram generation). Key functionalities include automatic column type detection rules, missing value handling, and categorical encoding. The tutorial helps with tasks like building custom feature pipelines, optimizing datetime processing, and implementing text feature generation, while highlighting best practices for data type management and column preprocessing.

# AutoGluon Tabular - Feature Engineering

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/tabular-feature-engineering.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/tabular-feature-engineering.ipynb)



## Introduction ##

Feature engineering involves taking raw tabular data and 

1. converting it into a format ready for the machine learning model to read
2. trying to enhance some columns ('features' in ML jargon) to give the ML models more information, hoping to get more accurate results.

AutoGluon does some of this for you.  This document describes how that works, and how you can extend it.  We describe here the default behaviour, much of which is configurable, as well as pointers to how to alter the behaviour from the default.

## Column Types ##

AutoGluon Tabular recognises the following types of features, and has separate processing for them:

| Feature Type        | Example Values           |
| :------------------ | :----------------------- |
| boolean             | A, B                     |
| numerical           | 1.3, 2.0, -1.6           |
| categorical         | Red, Blue, Yellow        |
| datetime            | 1/31/2021, Mar-31        |
| text                | Mary had a little lamb   |

In addition, other AutoGluon prediction modules recognise additional feature types, these can also be enabled in AutoGluon Tabular by using the [MultiModal](tabular-multimodal.ipynb) option. 

| Feature Type        | Example Values           |
| :------------------ | :----------------------- |
| image               | path/image123.png        |

## Column Type Detection ##

- Boolean columns are any columns with only 2 unique values.

- Any string columns are deemed categorical unless they are text (see below).  Some models perform better if you tell them which columns are categorical and which are continuous. 

- Numeric columns are passed through without change, except to identify them as `float` or `int`.  Currently, numeric columns are not tested to determine if they are likely to be categorical.  You can force them to be treated as categorical with the Pandas syntax `.astype("category")`, see below.

- Text columns are detected by firstly checking that most rows are unique.  If they are, and there are multiple separate words detected in most rows, the row is a text column.  For details see `common/features/infer_types.py` in the source.

- Datetime columns are detected by trying to convert them to Pandas datetimes.  Pandas detects a wide range of datetime formats.  If many of the values in a column are successfully converted, they are datetimes.  Currently datetimes that appear to be purely numeric (e.g. 20210530) are not correctly detected.  Any NaN values are set to the column mean.  For details see `common/features/infer_types.py`.


## Problem Type Detection ##

If the user does not specify whether the problem is a classification problem or a regression problem, the 'label' column is examined to try to guess.  Several things point towards a regression problem : the values are floating point non-integers, and there are a large amount of unique values.  Within classification, both multiclass and binary (n=2 categories) are detected.  For details see `utils/utils.py`.

To override the automatic inference, explicitly pass the problem_type (one of 'binary', 'regression', 'multiclass') to `TabularPredictor()`.  For example:

```
predictor = TabularPredictor(label='class', problem_type='multiclass').fit(train_data)
```


## Automatic Feature Engineering ##

## Numerical Columns ##

Numeric columns, both integer and floating point, currently have no automated feature engineering.

## Categorical Columns ##

Since many downstream models require categories to be encoded as integers, each categorical feature is mapped to monotonically increasing integers.

## Datetime Columns ##

Columns recognised as datetime, are converted into several features:

- a numerical Pandas datetime.  Note this has maximum and minimum values specified at [pandas.Timestamp.min](https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.min.html) and [pandas.Timestamp.max](https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.min.html) respectively, which may affect extremely dates very far into the future or past.
- several extracted columns, the default is `[year, month, day, dayofweek]`.  This is configrable via the [DatetimeFeatureGenerator](../../api/autogluon.features.rst)

Note that missing, invalid and out-of-range features generated by the above logic will be converted to the mean value across all valid rows.


## Text Columns ##

If the [MultiModal](tabular-multimodal.ipynb) option is enabled, then text columns are processed using a full Transformer neural network model with pretrained NLP models.

Otherwise, they are processed in two more simple ways:

- an n-gram feature generator extracts n-grams (short strings) from the text feature, adding many additional columns, one for each n-gram feature.  These columns are 'n-hot' encoded, containing 1 or more if the original feature contains the n-gram 1 or more times, and 0 otherwise.  By default, all text columns are concatenated before applying this stage, and the n-grams are individual words, not substrings of words.  You can configure this via the [TextNgramFeatureGenerator](../../api/autogluon.features.rst) class. The n-gram generation is done in `generators/text_ngram.py`
- Some additional numerical features are calculated, such as word counts, character counts, proportion of uppercase characters, etc.  This is configurable via the [TextSpecialFeatureGenerator](../../api/autogluon.features.rst).  This is done in `generators/text_special.py`

## Additional Processing ##

- Columns containing only 1 value are dropped before passing to models.
- Columns containing duplicates of other columns are removed before passing to models.

## Feature Engineering Example ##

By default a feature generator called [AutoMLPipelineFeatureGenerator](../../api/autogluon.features.rst) is used.  Let's see this in action.  We'll create a dataframe containing a floating point column, an integer column, a datetime column,  a categorical column.  We'll first take a look at the raw data we created.


```python
!pip install autogluon.tabular[all]

```


```python
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_regression
from datetime import datetime

x, y = make_regression(n_samples = 100,n_features = 5,n_targets = 1, random_state = 1)
dfx = pd.DataFrame(x, columns=['A','B','C','D','E'])
dfy = pd.DataFrame(y, columns=['label'])

# Create an integer column, a datetime column, a categorical column and a string column to demonstrate how they are processed.
dfx['B'] = (dfx['B']).astype(int)
dfx['C'] = datetime(2000,1,1) + pd.to_timedelta(dfx['C'].astype(int), unit='D')
dfx['D'] = pd.cut(dfx['D'] * 10, [-np.inf,-5,0,5,np.inf],labels=['v','w','x','y'])
dfx['E'] = pd.Series(list(' '.join(random.choice(["abc", "d", "ef", "ghi", "jkl"]) for i in range(4)) for j in range(100)))
dataset=TabularDataset(dfx)
print(dfx)
```

Now let's call the default feature generator AutoMLPipeLineFeatureGenerator with no parameters and see what it does.


```python
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=dfx)
```

We can see that:

- The floating point and integer columns 'A' and 'B' are unchanged.
- The datetime column 'C' has been converted to a raw value (in nanoseconds), as well as parsed into additional columns for the year, month, day and dayofweek.
- The string categorical column 'D' has been mapped 1:1 to integers - a lot of models only accept numerical input.
- The freeform text column has been mapped into some summary features ('char_count' etc) as well as a N-hot matrix saying whether each text contained each word.

To get more details, we should call the pipeline as part of `TabularPredictor.fit()`.  We need to combine the `dfx` and `dfy` DataFrames since fit() expects a single dataframe.


```python
df = pd.concat([dfx, dfy], axis=1)
predictor = TabularPredictor(label='label')
predictor.fit(df, hyperparameters={'GBM' : {}}, feature_generator=auto_ml_pipeline_feature_generator)
```

Reading the output, note that:

- the string-categorical column 'D', despite being mapped to integers, is still recognised as categorical. 
- the integer column 'B' has not been identified as categorical, even though it only has a few unique values:


```python
print(len(set(dfx['B'])))
```

To mark it as categorical, we can explicitly mark it as categorical in the original dataframe:


```python
dfx["B"] = dfx["B"].astype("category")
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=dfx)
```

## Missing Value Handling ##
To illustrate missing value handling, let's set the first row to all NaNs:


```python
dfx.iloc[0] = np.nan
dfx.head()
```

Now if we reprocess:


```python
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=dfx)
```

We see that the floating point, integer, categorical and text fields 'A', 'B', 'D', and 'E' have retained the NaNs, but the datetime column 'C' has been set to the mean of the non-NaN values.


## Customization of Feature Engineering ##
To customize your feature generation pipeline, it is recommended to call [PipelineFeatureGenerator](../../api/autogluon.features.rst), passing in non-default parameters to other feature generators as required.  For example, if we think downstream models would benefit from removing rare categorical values and replacing with NaN, we can supply the parameter maximum_num_cat to CategoryFeatureGenerator, as below:


```python
from autogluon.features.generators import PipelineFeatureGenerator, CategoryFeatureGenerator, IdentityFeatureGenerator
from autogluon.common.features.types import R_INT, R_FLOAT
mypipeline = PipelineFeatureGenerator(
    generators = [[        
        CategoryFeatureGenerator(maximum_num_cat=10),  # Overridden from default.
        IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),
    ]]
)
```

If we then dump out the transformed data, we can see that all columns have been converted to numeric, because that's what most models require, and the rare categorical values have been replaced with NaN:


```python
mypipeline.fit_transform(X=dfx)
```

For more on custom feature engineering, see the detailed notebook `examples/tabular/example_custom_feature_generator.py`.
