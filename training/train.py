import torch
from tqdm import tqdm

from utils.lr_schedulers import cosine_lr_with_warmup
from utils.losses import nt_xent_loss
from utils.optimizers import disable_bn_running_stats, enable_bn_running_stats


def train_one_epoch(model, train_loader, optimizer, device, global_step,
    total_steps, warmup_steps, peak_lr, epoch, num_epochs, temperature=0.5, ):
    model.train()

    running_loss = 0.0
    total_samples = 0
    learning_rates = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}",)

    for x_i, x_j in progress_bar:
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)
        lr = cosine_lr_with_warmup(optimizer=optimizer, step=global_step, total_steps=total_steps,
            warmup_steps=warmup_steps, peak_lr=peak_lr, )

        optimizer.zero_grad(set_to_none=True)
        x = torch.cat([x_i, x_j], dim=0)
        z = model(x)
        z_i, z_j = z.chunk(2)

        loss = nt_xent_loss(z_i, z_j, temperature=temperature, )
        loss.backward()

        if optimizer.name == "SAM":
            optimizer.first_step(zero_grad=True)
            disable_bn_running_stats(model)
            z_perturbed = model(torch.cat([x_i, x_j], dim=0))
            z_i_perturbed, z_j_perturbed = z_perturbed.chunk(2)
            loss_perturbed = nt_xent_loss(z_i_perturbed, z_j_perturbed, temperature=temperature)
            loss_perturbed.backward()
            enable_bn_running_stats(model)
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        current_batch_size = x_i.size(0)
        running_loss += loss.item() * current_batch_size
        total_samples += current_batch_size
        learning_rates.append(lr)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}", )
        global_step += 1

    epoch_loss = running_loss / total_samples if total_samples else 0.0
    return epoch_loss, learning_rates, global_step
