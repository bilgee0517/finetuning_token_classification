# Fine Tuning Distilbert for word classification in user-specific task

## Overview 

This repository contains a fine-tuned dslim/distilbert-NER model specifically designed for token classification tasks within sentences. The project demonstrates the application of a DistilBERT model, a lighter version of BERT, token classification tasks. The fine-tuning and application are tailored to efficiently and effectively classify words into categories in a given text, making it a valuable tool for natural language processing tasks.


## Model Description

The model used in this project is dslim/distilbert-NER, a distilled version of BERT (Bidirectional Encoder Representations from Transformers) fine-tuned for token classification for user specific task. DistilBERT offers a lighter and faster alternative to BERT while maintaining competitive performance, particularly in token classification tasks.

## Dataset

The model has been fine-tuned on a dataset specifically curated for token classification tasks. The dataset includes various sentence structures with annotated labels, enabling the model to learn and predict a wide range of entity types.

## Usage

The repository provides easy-to-use functions to tokenize input text, align labels, predict entities in new sentences, and visually represent these predictions. Users can input any sentence into the predict_sentence() function to receive token classifications, which can be further visualized using colorize_text().


## Requirements
- Python 3.x
- PyTorch
- Transformers library by Hugging Face
- Other dependencies listed in requirements.txt

To use without training [Download the Model](https://drive.google.com/file/d/15bHPPHiLYpT-1iVs4bz02XWkMk9T2ler/view?usp=drive_link)

## Installation
To use this project, clone the repository and install the required packages:

```bash
git clone https://github.com/[your-username]/distilbert-ner-token-classification.git
cd distilbert-ner-token-classification
pip install -r requirements.txt
```

## Example Usage 

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from functions import remove_punct, predict_sentence, colorize_text

#Loading model
model = BertModel()
tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

#Prediction
text = "Enter text"
predictions = predict_sentence(model, remove_punct(text))
colorize_text(text, predictions, color_map)
```

## Contact Information

[Linkedin](https://www.linkedin.com/in/bilegjargal-altangerel-6335ab25b/)|[Email](bilegjargal@uni.minerva.edu)| [Portfolio](https://obtainable-dart-e03.notion.site/Bilegjargal-Altangerel-Portfolio-f27a387c84d74f589e3cac8cce8d0d47?pvs=4) 