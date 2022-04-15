import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_model_with_normalization(
    data_loader, optimizer, model, criterion, metric, configs, is_train=True
):
    cumulative_loss = 0
    dataset_size = len(data_loader.dataset)
    for batch, labels, obj_id in data_loader:
        if is_train == 0:
            optimizer.zero_grad(set_to_none=True)
        mean = torch.mean(batch, dim=1, keepdim=True, dtype=float).to(device)
        input = batch.float() - mean
        outputs = model(input)
        outputs[:, :-1, -1:] = outputs[:, :-1, -1:] + torch.transpose(mean, -2, -1)
        if is_train == 2:
            return outputs, None
        loss = criterion(outputs, labels)
        cumulative_loss += loss
        metric.batch_accum(batch, outputs, labels)
        if is_train == 0:
            loss.backward()
            optimizer.step()
    return cumulative_loss.item() / dataset_size, metric.epoch_result(dataset_size)


def run_model_without_normalization(
    data_loader, optimizer, model, criterion, metric, configs, is_train=True
):
    cumulative_loss = 0
    dataset_size = len(data_loader.dataset)
    for batch, labels, obj_id in data_loader:
        if is_train == 0:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        if is_train == 2:
            return outputs, None
        loss = criterion(outputs, labels)
        cumulative_loss += loss
        metric.batch_accum(batch, outputs, labels)
        if is_train == 0:
            loss.backward()
            optimizer.step()
    return cumulative_loss.item() / dataset_size, metric.epoch_result(dataset_size)
