import torch
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder

from data_pipeline.augmentations import simclr_train_transforms
from data_pipeline.datasets import SimCLRDataset
from utils.distributed import Runtime, setup_runtime, wrap_model
from utils.optimizers import LARS, SAM
from utils.plotting import plot_fig
from training.train import train_one_epoch
from methods.simclr import SimCLRModel


def run_pretraining(args, runtime: Runtime):
    data_path = args.data_path or f"{args.root_dir}/data/{args.data}/train"
    output_root = Path(args.output_dir or f"{args.root_dir}/outputs")
    checkpoint_dir = output_root / "checkpoints" / args.ssl_type
    plot_dir = output_root / "plots" / args.ssl_type

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = args.epochs if args.epochs is not None else 100
    batch_size = args.batch_size if args.batch_size is not None else 512
    weight_decay = (args.weight_decay if args.weight_decay is not None else 1e-6)
    momentum = args.momentum
    global_batch_size = batch_size * runtime.world_size
    base_lr = args.learning_rate if args.learning_rate is not None else 0.3 * global_batch_size / 256

    device = runtime.device
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    base_dataset = ImageFolder(root=data_path, transform=None, )
    dataset = SimCLRDataset(base_dataset, simclr_train_transforms(), )
    sampler = (DistributedSampler(dataset, num_replicas=runtime.world_size, rank=runtime.rank,
                                  shuffle=True, drop_last=True)
               if runtime.distributed else None)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
                        num_workers=args.num_workers, drop_last=True, pin_memory=(runtime.device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0))

    model = SimCLRModel(resnet_variant=args.model) #  .to(device)
    model = wrap_model(model, runtime)

    if args.optimizer is None:
        optimizer_name = "lars"
    else:
        optimizer_name = args.optimizer

    if optimizer_name == "lars":
        optimizer = LARS(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay,
                         eta=args.eta,weight_decay_filter=True, lars_adaptation_filter=True, )
    elif optimizer_name == "sam":
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=base_lr, momentum=momentum, weight_decay=weight_decay,
                        rho=args.sam_rho, )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay,)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay, )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise ValueError("The DataLoader has no batches. Reduce the batch size "
            "or check the dataset and drop_last setting.")

    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(int(num_epochs * args.warmup_epochs) * steps_per_epoch, total_steps - 1,)

    train_losses = []
    learning_rates = []
    global_step = 0

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss, epoch_lrs, global_step = train_one_epoch(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            device=runtime.device,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_lr=base_lr,
            epoch=epoch,
            num_epochs=num_epochs,
            temperature=args.temperature,
        )

        if runtime.distributed:
            loss_tensor = torch.tensor(epoch_loss, device=runtime.device, dtype=torch.float64)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            epoch_loss = (loss_tensor / runtime.world_size).item()
        if runtime.is_main:
            train_losses.append(epoch_loss)
            learning_rates.append(epoch_lrs)

            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} LR: {epoch_lrs[-1]:.6f}")

            checkpoint_path = (checkpoint_dir / f"simclr_{args.model}_{args.optimizer}_checkpoint.pth")
            encoder_path = (checkpoint_dir / f"simclr_{args.model}_{args.optimizer}_encoder.pth")
            model_to_save = model.module if runtime.distributed else model

            plot_fig(train_losses, "Epoch", "Mean NT-Xent Loss", "SimCLR Training Loss",
                     str(plot_dir / f"{args.model}_{args.optimizer}_loss.png"), )
            plot_fig(learning_rates, "Training step", "Learning rate",
                "LR: warm-up + cosine decay", str(plot_dir / f"{args.model}_{args.optimizer}_lr.png"), )

            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epochs": num_epochs,
                    "global_step": global_step,
                    "train_losses": train_losses,
                    "model": args.model,
                    "optimizer": optimizer_name,
                },
                checkpoint_path,
            )
            torch.save(model_to_save.encoder.state_dict(), encoder_path)

            print(f"Checkpoint saved to: {checkpoint_path}")
            print(f"Encoder saved to: {encoder_path}")
