import torch.optim as optim
from mmengine import track_iter_progress


device = 'cuda' # or 'cpu'
max_epochs = 10

optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(max_epochs):
    model.train()
    losses = []
    for data_batch in track_iter_progress(train_data_loader):
        data = model.data_preprocessor(data_batch, training=True)
        loss_dict = model(**data, mode='loss')
        loss = loss_dict['loss_cls']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f'Epoch[{epoch}]: loss ', sum(losses) / len(train_data_loader))

    with torch.no_grad():
        model.eval()
        for data_batch in track_iter_progress(val_data_loader):
            data = model.data_preprocessor(data_batch, training=False)
            predictions = model(**data, mode='predict')
            data_samples = [d.to_dict() for d in predictions]
            metric.process(data_batch, data_samples)

        acc = metric.acc = metric.compute_metrics(metric.results)
        for name, topk in acc.items():
            print(f'{name}: ', topk)