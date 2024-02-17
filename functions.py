import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
from IPython.display import HTML

tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
label_all_tokens = False

def align_label(texts, labels):
    """
    Aligns labels with tokenized input.

    This function takes raw texts and their corresponding labels, tokenizes the texts,
    and aligns the labels with the tokenized inputs. Labels for non-first subwords and special tokens are set to -100.

    Args:
        texts (str): The text to be tokenized.
        labels (list of int): The labels corresponding to each word in `texts`.

    Returns:
        list of int: The list of aligned labels.
    """
    # Tokenize the input text
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=64, truncation=True)
    
    # Get the word IDs for each token
    word_ids = tokenized_inputs.word_ids()
    
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            # Set label to -100 for special tokens
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            # Set label for the first subword of each word
            label_ids.append(labels[word_idx] if word_idx < len(labels) else -100)
        else:
            # Set label for non-first subwords
            label_ids.append(labels[word_idx] if (label_all_tokens and word_idx < len(labels)) else -100)
        previous_word_idx = word_idx

    return label_ids


def align_word_ids(texts):
    """
    Aligns word IDs for given texts.

    This function tokenizes the given texts and aligns the word IDs, setting a value
    for each token indicating whether it's the first subword of a word or not.

    Args:
        texts (str): The text to be tokenized.

    Returns:
        list of int: The list of aligned word IDs.
    """
    # Tokenize the input text
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=64, truncation=True)

    # Get the word IDs for each token
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            # Set label to -100 for special tokens
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            # Set label to 1 for the first subword of each word
            label_ids.append(1)
        else:
            # Set label for non-first subwords
            label_ids.append(1 if label_all_tokens else -100)
        previous_word_idx = word_idx

    return label_ids


class DataSequence(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for tokenized text and aligned labels.

    This class is used to create a dataset from a DataFrame containing text and labels,
    which is suitable for training or evaluating machine learning models in PyTorch.

    Args:
        df (pandas.DataFrame): A DataFrame containing the columns 'labels' and 'text'.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer to tokenize the text.
    """
    def __init__(self, df, tokenizer):
        self.texts = [tokenizer(str(i), padding='max_length', max_length=64, truncation=True, return_tensors="pt") for i in df['text'].values]
        self.labels = [align_label(i, j) for i, j in zip(df['text'].values, df['labels'].values)]

    def __len__(self):
        # Returns the length of the dataset
        return len(self.labels)

    def get_batch_data(self, idx):
        # Retrieves batch data by index
        return self.texts[idx]

    def get_batch_labels(self, idx):
        # Retrieves batch labels by index
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        # Retrieves a data-label pair by index
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels
    
    
class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")


    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output

def predict_sentence(model, sentence):
    """
    Predicts labels for a given sentence using the specified model.

    Args:
        model (torch.nn.Module): The model used for prediction.
        sentence (str): The sentence to be predicted.

    Returns:
        list of int: The predicted labels for each token in the sentence.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    
    return predictions 


def remove_punct(text):
    """
    Removes punctuation from the given text.

    Args:
        text (str): The text from which punctuation will be removed.

    Returns:
        str: The text with punctuation removed.
    """
    return text.translate(str.maketrans('', '', '"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'))


def colorize_text(text, predictions, color_map):
    """
    Colorizes each word in the text based on its prediction.

    Args:
        text (str): The text to be colorized.
        predictions (list of int): The predictions for each word in the text.
        color_map (dict): A mapping of predictions to colors.

    Returns:
        HTML: The colorized text as HTML.
    """
    # Split the text into words and punctuation
    words = re.findall(r'\b\w+\b|[,.!?;]', text)
    color_map = {0: 'None', 1: 'Green', 2: 'Orange', 3: 'Yellow', 4: 'Blue'}

    # Start with an empty string and add each word with its corresponding color
    styled_text = ""
    for word, pred in zip(words, predictions):
        color = color_map.get(pred, 'None')  # Get the color, default to 'None'
        styled_text += f"<span style='background-color: {color}'>{word}</span> "

    # Return the styled text as HTML
    return HTML(styled_text)
