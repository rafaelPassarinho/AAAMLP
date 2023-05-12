import torch
import torch.nn as nn

def loss_fn(outputs, targets):
    '''
    This function returns the loss.
    :param outputs: outputs from the model (real numbers)
    :param targets: input targets (binary)
    '''
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler):
    '''
    This is the training function which trains for one epoch.
    :param data_loader: it is the torch dataloader object
    :param model: torch model, bert in our case
    :param optimizer: adam, sgd, etc
    :param device: can be cpu or cuda
    :param scheduler: learning rate scheduler
    '''
    # put the model on training mode
    model.train()

    # loop over all batches
    for d in data_loader:
        # extract ids, token type ids and mask
        # from current batch
        # also extract targets
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        # move everything to specified device
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        # zero-grad the optimizer
        optimizer.zero_grad()
        # pass through the model
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        # calculate loss
        loss = loss_fn(outputs, targets)
        # backward step the loss
        loss.backward()
        # step optimizer
        optimizer.step()
        # step scheduler
        scheduler.step()

def eval_fn(data_loader, model, device):
    '''
    This is the validation function that generates predictions
    for validation data.
    :param data_loader: it is the torch dataloader object
    :param model: torch model, bert in our case
    :param device: can be cpu or cuda
    :return: output and target
    '''
    # put the model in eval mode
    model.eval()
    # initialize empty lists for targets and outputs
    fin_targets = []
    fin_outputs = []
    # use the no_grad scope
    # its very importante else you might run out of gpu memory
    with torch.no_grad():
        # this part is same as train function
        # except for the fact that there is no
        # zero_grad of optimizer and there is no loss
        # calculation or scheduler steps
        for d in data_loader:
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            targets = d['targets']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            # convert targets to cpu and extend the final list
            targets = targets.cpu().detach()
            fin_targets.extend(targets.numpy().tolist())
            # convert outputs to cpu and extend the final list
            outputs = torch.sigmoid(outputs).cpu().detach()
            fin_outputs.extend(outputs.numpy().tolist())
    # return final output and targets
    return fin_outputs, fin_targets
