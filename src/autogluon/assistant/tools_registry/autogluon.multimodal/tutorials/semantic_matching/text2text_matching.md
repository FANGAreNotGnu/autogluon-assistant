Summary: This tutorial demonstrates implementing text-to-text semantic matching using AutoGluon's MultiModalPredictor. It covers techniques for computing similarity between text pairs using BERT embeddings, specifically useful for tasks like web search, QA, and document deduplication. Key functionalities include data preparation with SNLI dataset, model training configuration with text similarity settings, making predictions on text pairs, and extracting embeddings. The tutorial shows how to set up binary classification for semantic matching, configure match labels, and evaluate model performance using AUC metrics, all through AutoMM's streamlined interface.

# Text-to-Text Semantic Matching with AutoMM 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/semantic_matching/text2text_matching.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/semantic_matching/text2text_matching.ipynb)



Computing the similarity between two sentences/passages is a common task in NLP, with several practical applications such as web search, question answering, documents deduplication, plagiarism comparison, natural language inference, recommendation engines, etc. In general, text similarity models will take two sentences/passages as input and transform them into vectors, and then similarity scores calculated using cosine similarity, dot product, or Euclidean distances are used to measure how alike or different of the two text pieces. 

## Prepare your Data
In this tutorial, we will demonstrate how to use AutoMM for text-to-text semantic matching with the Stanford Natural Language Inference ([SNLI](https://nlp.stanford.edu/projects/snli/)) corpus. SNLI is a corpus contains around 570k human-written sentence pairs labeled with *entailment*, *contradiction*, and *neutral*. It is a widely used benchmark for evaluating the representation and inference capbility of machine learning methods. The following table contains three examples taken from this corpus.

| Premise                                                   | Hypothesis                                                           | Label         |
|-----------------------------------------------------------|----------------------------------------------------------------------|---------------|
| A black race car starts up in front of a crowd of people. | A man is driving down a lonely road.                                 | contradiction |
|  An older and younger man smiling.                        | Two men are smiling and laughing at the cats playing on the   floor. | neutral       |
| A soccer game with multiple males playing.                | Some men are playing a sport.                                        | entailment    |

Here, we consider sentence pairs with label *entailment* as positive pairs (labeled as 1) and those with label *contradiction* as negative pairs (labeled as 0). Sentence pairs with neural relationship are discarded. The following code downloads and loads the corpus into dataframes.


```python
!pip install autogluon.multimodal

```


```python
from autogluon.core.utils.loaders import load_pd
import pandas as pd

snli_train = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_train.csv', delimiter="|")
snli_test = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_test.csv', delimiter="|")
snli_train.head()
```

## Train your Model

Ideally, we want to obtain a model that can return high/low scores for positive/negative text pairs. Traditional text similarity methods only work on a lexical level without taking the semantic aspect into account, for example, using term frequency or tf-idf vectors. With AutoMM, we can easily train a model that captures the semantic relationship between sentences. Basically, it uses [BERT](https://arxiv.org/abs/1810.04805) to project each sentence into a high-dimensional vector and treat the matching problem as a classification problem following the design in [sentence transformers](https://www.sbert.net/).
With AutoMM, you just need to specify the query, response, and label column names and fit the model on the training dataset without worrying the implementation details. Note that the labels should be binary, and we need to specify the `match_label`, which means two sentences have the same semantic meaning. In practice, your tasks may have different labels, e.g., duplicate or not duplicate. You may need to define the `match_label` by considering your specific task contexts.


```python
from autogluon.multimodal import MultiModalPredictor

# Initialize the model
predictor = MultiModalPredictor(
        problem_type="text_similarity",
        query="premise", # the column name of the first sentence
        response="hypothesis", # the column name of the second sentence
        label="label", # the label column name
        match_label=1, # the label indicating that query and response have the same semantic meanings.
        eval_metric='auc', # the evaluation metric
    )

# Fit the model
predictor.fit(
    train_data=snli_train,
    time_limit=180,
)
```

## Evaluate on Test Dataset
You can evaluate the macther on the test dataset to see how it performs with the roc_auc score:


```python
score = predictor.evaluate(snli_test)
print("evaluation score: ", score)
```

## Predict on a New Sentence Pair
We create a new sentence pair with similar meaning (expected to be predicted as $1$) and make predictions using the trained model.


```python
pred_data = pd.DataFrame.from_dict({"premise":["The teacher gave his speech to an empty room."], 
                                    "hypothesis":["There was almost nobody when the professor was talking."]})

predictions = predictor.predict(pred_data)
print('Predicted entities:', predictions[0])
```

## Predict Matching Probabilities
We can also compute the matching probabilities of sentence pairs.


```python
probabilities = predictor.predict_proba(pred_data)
print(probabilities)
```

## Extract Embeddings
Moreover, we support extracting embeddings separately for two sentence groups.


```python
embeddings_1 = predictor.extract_embedding({"premise":["The teacher gave his speech to an empty room."]})
print(embeddings_1.shape)
embeddings_2 = predictor.extract_embedding({"hypothesis":["There was almost nobody when the professor was talking."]})
print(embeddings_2.shape)
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.


## Customization

To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
