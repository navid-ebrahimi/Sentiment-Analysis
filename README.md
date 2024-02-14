# Sentiment-Analysis

---
author:
- Navid Ebrahimi, Baktash Ansari

date: 2024-02-06

title: Sentiment-Analysis
---

## Project Proposal

### Overview
Sentiment analysis, also known as opinion mining, is a field of study that analyzes people's opinions, feelings, experiences, and emotions expressed through written language. This field is a subset of natural language processing, data mining, web mining, and text mining.
When it comes to Persian text, sentiment analysis faces unique challenges. Persian language has special features and requires unique methods and models to analyze emotions.

### Goals
1. In the first method, the purpose of dividing the emotions of the dataset data into 7 groups is defined.
2. In the second method, the purpose of dividing the emotions of the data, which includes images and text, is defined into 7 groups.

#### Data Collection:
The ArmanEmo dataset is a collection of human-tagged emotions that contains more than 7000 Persian sentences labeled for seven categories. This dataset was introduced by Mirzaei et al. In their study, they introduced it as "ArmanEmo: Persian dataset for text-based emotion recognition".

The sentences in the ArmanEmo dataset are collected from various sources, including Twitter comments, Instagram and DigiKala. Labels are based on the six main emotions (anger, fear, happiness, disgust, sadness, surprise) and another category (other) to account for any other emotions not included in the Ekman model.

#### Data Preprocessing:
I will divide the implementation we did in this part into two parts:
**1- Displaying the most important data information:** In this part, with pandas and matplotlib tools, we displayed tips such as the most repeated words in the text, the number of unique words in each label, and 5 frequently used words in each label. These statistics will definitely come in handy at the end of the work when we start evaluating the models.
**2- Dataset cleaning:** In this step, we used the HAZM library from Roshan. First, we normalized the data. One of the examples modified by this is the removal of extra spaces between words. Next, we used Stemmer. With this, we were able to understand the main root of the words. For example, if the word "books" exists, then this word becomes "book". With these transformations, we can train the model more accurately.
Another method we explored was the lemmatizer. At this stage, by choosing each word, we turn it into its root. But with this method, because the word becomes its root, it may lose its emotional meaning. After training the model with this method and poor input accuracy, we removed this method. Regarding the preprocessing methods, we can refer to the main site of the HAZM library.

For embedding the words of our dataset, we also used the tokenizers of the used models.

#### Train model:
We used 2 models for the fine tuning process:

1. Bert:
As one of the base models, we used a pre-trained model for Persian language, ParsBERT. ParsBERT is a monolingual model implemented based on the BERT architecture. The designers of this model have shown that the ParsBERT model outperforms multilingual BERT and previous models in several Persian NLP tasks, including text classification and sentiment analysis. Compared to the multilingual BERT model, ParsBERT is trained on a larger and more diverse (in terms of topic range and writing style) pre-trained Persian dataset.

2. XLM-RoBERTa:
We also use a model called XLM-RoBERTa as our other model. XLM-RoBERTa is another transformer-based masked language model that has already been trained on texts in 100 languages of the world. This multilingual language model has resulted in improved performance in cross-linguistic classification, sequence labeling, and question answering, outperforming multilingual BERT (mBERT) on various multilingual metrics.

Although it is clear that ParsBERT as a monolingual linguistic model performs better than multilingual BERT on various tasks in Farsi, we decided to compare the performance of XLM-RoBERTa variations against ParsBERT in emotion recognition as well.

## Actions taken in the first method of the project
**1. Statistical information:** First, we reported a set of statistical information, such as the most frequent words, etc., which we talked about earlier.

**2. Pre-processing:** We then pre-processed and cleaned the data using the digest library we talked about.

**3. Teaching the PARSBERT model:**
We used the Transformers Hugging Face library to train and evaluate the models in question. We started this work by defining the "compute_metrics" function to calculate accuracy, recall, F1-Score and precision. Then, we specified the model name and loaded the Tokenizer and model and prepared the dataset for train and test. We defined training hyperparameters such as output directory, number of epochs, batch sizes, learning rate and other parameters.
We use Trainer class with specified arguments and data set for training. After training, we test the model on the test dataset and print the evaluation results. The model used here is a pre-trained BERT model for Persian language and it is set for a sequence classification task with seven labels. The data is encoded using the same tokenizer that was used for pre-training the model. Training and evaluation datasets are created from these encodings and corresponding labels. The training process is managed by the "Trainer" class, which also calculates the specified criteria during evaluation. The best model is saved at the end of training based on evaluation loss. The final evaluation results are printed.

**3. XLM-RoBERTa model training:**
We did this training using pytorch and tqdm library. In this way, like the previous model, we loaded the tokenizer and model from hugging face and prepared the test and train data set using DataLoader. And we did the model training process. In order to observe the progress of the model in the training process, we used the tqdm library.

**5. Model evaluation using the predict_label function:**
Finally, we load the models and evaluate the models on 5 samples using the predict_label function located at the end of the notebook.

Note: Pay attention that the training and evaluation steps have been done several times with different hyperparameters in order to reach an acceptable and good result.

#### Evaluation:
![header_img](
https://github.com/navid-ebrahimi/Sentiment-Analysis/blob/main/images/firstMethod_Evaluation.png)


## Actions taken in the second method of the project
For the scoring part, we first found a data set that is used to analyze positive, negative, and neutral emotions. This data can be seen in [this link](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/). This data called MVSA includes a collection of photos and texts collected from the social network Twitter (X) in such a way that each sample of this data set contains a photo and a tweet related to it, and their corresponding label is also with three Neutral, positive and negative values can be displayed. We loaded this dataset into a separate notebook and showed a sample of it. For the training model, we used a combination of Resnet50 and BERT models. In this way, for the photo modality, we gave the photo input to the Resnet model and removed the last layer of this model to reach a feature vector. For the text data, we gave the text input to the BERT model to extract a feature vector from it. Next, using pytorch tools, we concat these two vectors together to have a unique feature vector, and we entered this vector as input to a four-layer MLP model for training, and finally we trained the model. Finally, we tested the model on a sample of data.
The accuracy and loss of the model can be seen on the train and test data:

![header_img](
https://github.com/navid-ebrahimi/Sentiment-Analysis/blob/main/images/secondMethod_Evaluation.png)

![header_img](
https://github.com/navid-ebrahimi/Sentiment-Analysis/blob/main/images/secondMethod_Evaluation2.png)
