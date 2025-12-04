import torch
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os


def train_one_epoch(epoch_index: int,
                    tb_writer: torch.utils.tensorboard.SummaryWriter,
                    model: torch.nn.Module,
                    loss_fn: torch.nn.modules.loss.Module,
                    training_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer) -> float:
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
            print(f'---- batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def train(training_set: torch.utils.data.Dataset,
          validation_set: torch.utils.data.Dataset,
          batch_size: int,
          model: torch.nn.Module,
          loss_function: torch.nn.modules.loss._Loss,
          learning_rate: float,
          momentum: float,
          epochs: int,
          ) -> None:
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    print(f'Training set has {len(training_set)} instances')
    print(f'Validation set has {len(validation_set)} instances')

    neural_network = model()
    loss_fn = loss_function() # Loss function doesn't require initialization yet
    optimizer = torch.optim.SGD(neural_network.parameters(), lr=learning_rate, momentum=momentum)

    # Trenowanie
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('runs', exist_ok=True)
    writer = SummaryWriter(f'runs/trainer_{timestamp}')
    epoch_number = 0

    best_vloss = 1_000_000.
    for epoch in range(epochs):
        print(f'EPOCH {epoch_number + 1}:')

        # Make sure gradient tracking is on and do a pass over the data
        neural_network.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, neural_network, loss_fn, training_loader, optimizer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        neural_network.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                _, vlabels, vinputs = vdata.values()
                voutputs = neural_network(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track the best performance and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            os.makedirs(f'models/model_{timestamp}', exist_ok=True)
            model_path = f'models/model_{timestamp}/epoch_{epoch_number}'
            torch.save(neural_network.state_dict(), model_path)

        epoch_number += 1
