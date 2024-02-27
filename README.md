# Creation-of-a-synthetic-dataset-for-French-NER-in-clinical-trial-texts
Final project of Hands-On NLP course at Université Paris-Saclay M1-AI

**Authors:**

1. Carlos Cuevas Villarmín
2. José Felipe Espinosa Orjuela
3. Javier Alejandro Lopetegui González

*keywords:* machine translation, NER, multilingual, medical trials
### Introduction:

The objective of this work is to create a dataset in French for Named Entity Recognition (NER) in the context of medical trials eligibility criteria. We are going to use a cross-lingual approach to make it possible based on the idea from this [blog](https://pradeepundefned.medium.com/how-to-do-named-entity-recognition-for-languages-other-than-english-bac58898ad33).

As starting point we are going to use a dataset built from the original [CHIA dataset](https://figshare.com/articles/dataset/Chia_Annotated_Datasets/11855817) but just considering non-overlaping entities. This dataset is annotated in [BIO](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) format. Then, the process for creating this French version of the dataset consists in:

1. Translate sentences from English to French using a Neural Machine Translation (NMT) model.
2. Fine-tune a transformers-based multilingual model for NER over the English version of the corpus.
3. Use the fine-tuned model in English data to annotate the sentences in French.

In order to validate this pipeline for French annotations we are going to evaluate the performance of the selected model over a dataset already annotated in English and French: [multiNERD](https://huggingface.co/datasets/Babelscape/multinerd). As in our pipeline, we are going to train the model in English and evaluate it using the french evaluation data.

Now we are going to explain in details the process and the results obtained.

### Model Selection for NER:

For the model implementation we are going to use the huggingface version for TokenClassification of the [XML-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base) model. This model is a multilingual version of [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta). It is pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages (French and English among them).

We are able to load the model directly from hugginface as follow:

```
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=len(label_list), label2id=labels_vocab, id2label=labels_vocab_reverse)
```

where the rest of the inputs are going to be explained afterward.

### Evaluation of XLM-RoBERTa performance over MultiNERD dataset in the cross-lingua NER task.

As we explained before, in order to validate the idea implemented for the corpus annotation we used the MultiNERD corpus. We fine-tuned the **XML-roBERTa** model for NER using the english version of the corpus and evaluated over the French test subset. This implementation is accessible in the [multilingual_ner_model](https://github.com/jlopetegui98/Creation-of-a-synthetic-dataset-for-French-NER-in-clinical-trial-texts/blob/main/Multilingual-NER-Model/multilingual_ner_model.ipynb) notebook.

In the following image we show the distribution of languages in the dataset:

<img src="./images/dataset_languages.png"
     alt="dataset_languages"
     style="float: left; margin-right: 10px;" />

SPACE TO EXPLAIN THE RESULTS

If we compare the results obtained after training just in english with the results obtained in the training 


### section
