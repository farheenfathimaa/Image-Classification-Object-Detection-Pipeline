"""
Training script for fine-tuning ResNet-50 on the Oxford-IIIT Pet Dataset.
Implements backbone freezing, learning rate scheduling, early stopping,
and dual-mode dataset downloading (torchvision or HuggingFace fallback).
"""

import os
import json
import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


class TransformedSubset(Dataset):
    """
    Wraps a PyTorch Dataset Subset to apply image transformations on the fly.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class HFPetsSubset(Dataset):
    """
    Wraps a Hugging Face dataset split for PyTorch compatibility with custom transforms.
    """
    def __init__(self, hf_dataset, indices, transform=None):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.hf_dataset[real_idx]
        img = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_train_transforms():
    """
    Defines the training data augmentation pipeline.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    """
    Defines the validation data transformation pipeline (no augmentations).
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_datasets(data_dir):
    """
    Loads train and validation subsets of the Oxford-IIIT Pet Dataset.
    Tries torchvision.datasets.OxfordIIITPet first.
    Falls back to HuggingFace 'datasets' if torchvision download fails.
    
    Returns:
        train_dataset: PyTorch Dataset for training
        val_dataset: PyTorch Dataset for validation
        classes: List of 37 class names
    """
    train_dataset = None
    val_dataset = None
    classes = []

    # 1. Try torchvision datasets
    try:
        print("Attempting to load Oxford-IIIT Pet Dataset via torchvision.datasets...")
        # Download the full trainval split without transforms initially
        full_dataset = torchvision.datasets.OxfordIIITPet(
            root=data_dir,
            split="trainval",
            target_types="category",
            download=True,
            transform=None
        )
        classes = full_dataset.classes
        
        # 80/20 train/validation split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        
        train_dataset = TransformedSubset(train_subset, transform=get_train_transforms())
        val_dataset = TransformedSubset(val_subset, transform=get_val_transforms())
        print("Successfully loaded Oxford-IIIT Pet Dataset via torchvision.")
        
    except Exception as e:
        print(f"Warning: torchvision load failed: {e}")
        print("Attempting fallback download via Hugging Face datasets...")
        
        # 2. Try Hugging Face fallback
        try:
            from datasets import load_dataset
            # Download train split which has 3680 instances (same as torchvision trainval)
            hf_data = load_dataset("oxford_iiit_pet", cache_dir=data_dir)
            hf_trainval = hf_data["train"]
            classes = hf_trainval.features["label"].names

            # Manual random split index computation
            train_size = int(0.8 * len(hf_trainval))
            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(hf_trainval), generator=generator).tolist()
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_dataset = HFPetsSubset(hf_trainval, train_indices, transform=get_train_transforms())
            val_dataset = HFPetsSubset(hf_trainval, val_indices, transform=get_val_transforms())
            print("Successfully loaded Oxford-IIIT Pet Dataset via Hugging Face.")
            
        except Exception as hf_err:
            raise RuntimeError(
                f"Failed to load dataset using both torchvision and Hugging Face. Error: {hf_err}"
            )

    return train_dataset, val_dataset, classes


def freeze_backbone(model):
    """
    Freezes all layers of the model except the fully-connected classification head.
    """
    print("Freezing ResNet-50 backbone layers...")
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """
    Unfreezes all parameters of the model.
    """
    print("Unfreezing all ResNet-50 layers...")
    for param in model.parameters():
        param.requires_grad = True


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on validation data.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc


def main():
    """
    Main function to parse arguments, setup models, train and validate.
    """
    # Determine base directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    default_data_dir = os.path.join(project_root, "data")
    default_model_dir = os.path.join(project_root, "models")
    default_model_path = os.path.join(default_model_dir, "resnet50_pets_best.pth")
    default_log_path = os.path.join(default_model_dir, "training_log.csv")

    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 on Oxford-IIIT Pet Dataset.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Directory to save dataset files.")
    parser.add_argument("--model-path", type=str, default=default_model_path, help="Path to save best model.")
    parser.add_argument("--log-path", type=str, default=default_log_path, help="Path to save CSV training log.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset, val_dataset, classes = load_datasets(args.data_dir)
    print(f"Dataset summary: {len(train_dataset)} train, {len(val_dataset)} val, {len(classes)} classes")

    # Save classes JSON for inference consistency
    class_names_path = os.path.join(os.path.dirname(args.model_path), "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(classes, f, indent=4)
    print(f"Saved class names config to: {class_names_path}")

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # Initialize model
    print("Loading pretrained ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    model = model.to(device)

    # Freeze backbone initially
    freeze_backbone(model)

    # Setup loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    # Initialize optimizer with all parameters so it references them
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Track metrics
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0
    logs = []

    print("\nStarting Training Pipeline...\n")
    for epoch in range(args.epochs):
        # Unfreeze all layers at epoch 5
        if epoch == 5:
            print(f"\n--- Epoch {epoch}: Unfreezing backbone for full fine-tuning ---")
            unfreeze_all(model)
            # Recreate optimizer to update parameter groups properly
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # Link new optimizer to the scheduler and continue
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 5)

        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        epoch_time = time.time() - start_time

        # Print epoch metrics
        print(
            f"Epoch [{epoch+1:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # Log metrics
        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Early stopping and checkpoint saving on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.model_path)
            print(f"--> Saved best checkpoint to: {args.model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered! No validation loss improvement for {args.patience} epochs.")
                break

    # At the end print best val accuracy
    print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")

    # Save logs to CSV
    df = pd.DataFrame(logs)
    df.to_csv(args.log_path, index=False)
    print(f"Metrics logged to: {args.log_path}")


if __name__ == "__main__":
    main()
