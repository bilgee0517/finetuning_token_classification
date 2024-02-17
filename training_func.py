import torch
from functions import align_word_ids, DataSequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD

LEARNING_RATE = 5e-3
EPOCHS = 10
BATCH_SIZE = 4

def train_loop(model, tokenizer, df_train, df_val):
    """
    Trains a model for token classification using the given data.

    Args:
        model (torch.nn.Module): The model to be trained.
        tokenizer: The tokenizer used to tokenize the input data.
        df_train (pandas.DataFrame): The training data.
        df_val (pandas.DataFrame): The validation data.

    Returns:
        None
    """

    train_dataset = DataSequence(df_train, tokenizer)
    val_dataset = DataSequence(df_val, tokenizer)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][train_label[i] != -100]
              label_clean = train_label[i][train_label[i] != -100]
 
              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_train += acc
              total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):

                logits_clean = logits[i][val_label[i] != -100]
                label_clean = val_label[i][val_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')
    
        if best_acc == 0 or val_accuracy > best_acc:
            torch.save(model.state_dict(), 'best_model.pth') # change best model name/location as needed
            print(f'Saved new best model with accuracy {val_accuracy: .3f}')
            best_acc = val_accuracy

    print('Training complete. Model saved as best_model.pth.')