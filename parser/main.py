from MnistModel import MnistModel
from MnistDataset import MnistDataset
from transforms.Downscale import Downscale
from transforms.ToTensor import ToTensor
from quantize_model import quantize_model

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os
import sys

def train_one_epoch(epoch_index, tb_writer, model, loss_fn, training_loader, optimizer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input plus label pair
        _, labels, inputs = data.values()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def train():
    transformers = torchvision.transforms.Compose([Downscale(16), ToTensor()])

    training_set = MnistDataset('MNIST_CSV/mnist_train.csv', transform=transformers)
    validation_set = MnistDataset('MNIST_CSV/mnist_test.csv', transform=transformers)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=False)

    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    mnist_model = MnistModel()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(mnist_model.parameters(), lr=0.001, momentum=0.9)

    # Trenowanie
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('runs', exist_ok=True)
    writer = SummaryWriter('runs/trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 25

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on and do a pass over the data
        mnist_model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, mnist_model, loss_fn, training_loader, optimizer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        mnist_model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                _, vlabels, vinputs = vdata.values()
                voutputs = mnist_model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track the best performance and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            os.makedirs('models/model_{}'.format(timestamp), exist_ok=True)
            model_path = 'models/model_{}/epoch_{}'.format(timestamp, epoch_number)
            torch.save(mnist_model.state_dict(), model_path)

        epoch_number += 1

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] == 'quantize':
        quantize_model(sys.argv[2], sys.argv[3])
