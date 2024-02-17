import torch 
from torch.utils.data import DataLoader

def evaluate(model, df_test, tokenizer):
    """
    Evaluate the performance of a model on a test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        df_test (pandas.DataFrame): The test dataset.
        tokenizer: The tokenizer used to preprocess the data.

    Returns:
        float: The test accuracy.
    """
    from functions import DataSequence

    test_dataset = DataSequence(df_test, tokenizer)

    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=4)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0.0

    for test_data, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_data['attention_mask'].squeeze(1).to(device)

            input_id = test_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, test_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][test_label[i] != -100]
              label_clean = test_label[i][test_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_test += acc

    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {total_acc_test / len(df_test): .3f}')

