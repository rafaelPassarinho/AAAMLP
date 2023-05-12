import config_bert
import dataset_bert
import engine_bert
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model_bert import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def train():
    # this function trains the model

    # read the trainig file and fill NaN values with 'none'
    # you can also choose to drop NaN values in this case
    dfx = pd.read_csv(config_bert.TRAINING_FILE).fillna('none')

    # sentiment = 1 if its positive else 0
    dfx.sentiment = dfx.sentiment.apply(
        lambda x: 1 if x == 'positive' else 0
    )

    # we split the data into single training and validation folds
    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.sentiment.values
    )

    # reset index and drop old index
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # initialize BERTDataset from dataset_bert.py
    # for training dataset
    train_dataset = dataset_bert.BERTDataset(
        review=df_train.review.values,
        target=df_train.sentiment.values
    )

    # create a training data loader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config_bert.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    # initialize BERTDataset from dataset_bert.py
    # for validation dataset
    valid_dataset = dataset_bert.BERTDataset(
        review=df_valid.review.values,
        target=df_valid.sentiment.values
    )

    # create validation data loader
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config_bert.VALID_BATCH_SIZE,
        num_workers=1
    )

    # initialize the cuda device
    # use cpu if you dont have gpu
    device = torch.device('cuda')
    # load the model and send it to device
    model = BERTBaseUncased()
    model.to(device)

    # create parameters we want to optimize
    # we generally dont use any decay for bias and layer norm weights
    # as they dont generalize well
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # separate the parameters for applying weight decay
    # and not applying weight decay
    optimizer_parameters = [
        {
            'params': [
                p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay
                )
            ],
            'weight_decay': 0.001,
        },
        {
            'params': [
                p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay
                )
            ],
            'weight_decay': 0.0,
        },
    ]

    # calculate the number of training steps
    # this is used by scheduler
    num_train_steps = int(
        len(df_train) / config_bert.TRAIN_BATCH_SIZE * config_bert.EPOCHS
    )

    # AdamW optimizer
    # AdamW is the most widely used optimizer for transformer based networks
    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    # fetch the scheduler
    # you can also try using reduce lr on plateau
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     patience=3,
    #     threshold=0.0001,
    #     mode='max'
    # )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # if you have multiple GPUs then use DataParallel
    # model = nn.DataParallel(model)

    # start training the spochs
    best_accuracy = 0
    for epoch in range(config_bert.EPOCHS):
        engine_bert.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine_bert.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f'Accuracy Score = {accuracy}')
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config_bert.MODEL_PATH)
            best_accuracy = accuracy

if __name__ == '__main__':
    train()