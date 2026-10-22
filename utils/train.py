import torch
from tqdm import tqdm


def training_epoch(model, train_loader, optimizer, loss_function, device,
                   epoch, num_epochs, silent=False, ):
    model.train()

    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
                             leave=False, disable=silent, )

    for i, (x_i, x_j) in enumerate(train_loader_tqdm):
        x_i, x_j = x_i.to(device), x_j.to(device)
        z_i = model(x_i)
        z_j = model(x_j)

        loss = loss_function(z_i, z_j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if not silent:
            train_loader_tqdm.set_postfix(loss=loss.item(), refresh=False)

        # if ((i + 1) % 45 == 0) & (not silent):
        #     print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss
