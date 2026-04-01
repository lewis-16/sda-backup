import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from PIL import Image
from tqdm import tqdm

from train_fc6_recon import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FC6_DIM,
    RECON_SIZE,
    FC6Decoder,
    get_transforms,
    collect_image_paths,
    FC6ReconDataset,
)


class AlexNetFC6Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        return x


class FC6AutoEncoder(nn.Module):
    def __init__(self, recon_size=128):
        super().__init__()
        self.encoder = AlexNetFC6Encoder()
        self.decoder = FC6Decoder(fc6_dim=FC6_DIM, recon_size=recon_size)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/disk1/jinchentao/imagenet_256")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./ckpt_fc6_ae")
    parser.add_argument("--recon_size", type=int, default=RECON_SIZE)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0, help="freeze encoder for first N epochs, 0=train both from start")
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--clip_grad", type=float, default=1.0, help="gradient clipping max_norm, 0 to disable")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    image_paths = collect_image_paths(args.data_root)
    n_total = len(image_paths)
    n_train = int(n_total * args.train_ratio)
    n_val = n_total - n_train
    random.shuffle(image_paths)
    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:]

    transform_alex, transform_target = get_transforms(args.recon_size)
    train_ds = FC6ReconDataset(train_paths, transform_alex, transform_target)
    val_ds = FC6ReconDataset(val_paths, transform_alex, transform_target)
    nw = os.cpu_count() or 32
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FC6AutoEncoder(recon_size=args.recon_size).to(device)
    freeze_enc = args.freeze_encoder_epochs > 0
    for p in model.encoder.parameters():
        p.requires_grad = not freeze_enc
    params = list(model.decoder.parameters()) if freeze_enc else list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        if freeze_enc and epoch == args.freeze_encoder_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": list(model.encoder.parameters()), "lr": args.lr})
            print(f"Epoch {epoch+1}: unfreezing encoder, training both")
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for alex_in, target in pbar:
            alex_in = alex_in.to(device)
            target = target.to(device)
            recon = model(alex_in)
            loss = criterion(recon, target)
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        train_loss = epoch_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for alex_in, target in val_loader:
                alex_in = alex_in.to(device)
                target = target.to(device)
                recon = model(alex_in)
                val_loss += criterion(recon, target).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr_now:.2e}")
        torch.save({
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
        }, os.path.join(args.save_dir, "fc6_ae_latest.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
            }, os.path.join(args.save_dir, "fc6_ae_best.pth"))
            print(f"  -> best model saved (val_loss={val_loss:.6f})")
        if (epoch + 1) % 10 == 0:
            torch.save({
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
            }, os.path.join(args.save_dir, f"fc6_ae_ep{epoch+1}.pth"))
    print("Done. Best val_loss={:.6f}, checkpoints in {}".format(best_val_loss, args.save_dir))


if __name__ == "__main__":
    main()
