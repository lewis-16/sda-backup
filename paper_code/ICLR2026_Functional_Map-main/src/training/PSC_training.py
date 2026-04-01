import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# ----------- Training Loop ------------
def train(model, dataloader, optimizer, criterion, device, writer, epoch, print_tensors=False):
    """
    Training loop for one epoch.

    Args:
        model (nn.Module): The Siamese model.
        dataloader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer for gradient updates.
        criterion (function): Loss function.
        device (torch.device): Device to train on.
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Current epoch number.
        print_tensors (bool): Whether to print sample predictions and labels.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (x1, x2, label) in enumerate(dataloader):
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(x1, x2)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        preds = (output > 0.25).float()
        correct += (preds == label).sum()
        total += label.size(0)
        if i == 0 and print_tensors:
            print("Train Preds:", preds[:10])
            print("Train Labels:", label[:10])
        writer.add_scalar('Train/Batch_Loss', loss.item(), epoch * len(dataloader) + i)
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    print(f"Epoch {epoch+1} Train Accuracy: {accuracy:.4f}")
    writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', accuracy, epoch)
    return avg_loss


def evaluate(model, dataloader, criterion, device, writer, epoch, print_tensors=False):
    """
    Evaluation loop for validation or test set.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Validation/test data loader.
        criterion (function): Loss function.
        device (torch.device): Device for evaluation.
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Current epoch number.
        print_tensors (bool): Whether to print sample predictions and labels.

    Returns:
        float: Average evaluation loss.
    """
    model.eval() # disable drop outs and does batch normalization in validation mode
    total_loss = 0.0
    correct = 0
    total = 0
    tcounter = 0
    with torch.no_grad():
        for x1, x2, label in dataloader:
            tcounter = tcounter + 1
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            output = model(x1, x2)
            loss = criterion(output, label)
            total_loss += loss.item()
            preds = (output > 0.25).float()  # Same for both train and test
            correct += (preds == label).sum().item()
            total += label.size(0)
            if tcounter == 1 and print_tensors:
                print("Test Preds:", preds[0:15])
                print("Test Labels:", label[0:15])
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Epoch {epoch+1} Test Accuracy: {accuracy:.4f}")
    writer.add_scalar('Test/Epoch_Loss', avg_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)
    return avg_loss

def training_validate_loop(model, train_loader, test_loader, device, run_dir, epochs, lr = 1e-3, patience=1000):
    """
    Full training and validation loop with early stopping and model checkpointing.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        device (torch.device): CUDA or CPU.
        run_dir (str): Directory for logs and saved models.
        epochs (int): Maximum number of training epochs.

    Returns:
        nn.Module: The best model (based on validation loss).
    """ 
    # TensorBoard setup
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5)

    criterion = contrastive_loss

    model_save_path = run_dir + "\\full_siamese_model_best.pt"
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, writer, epoch)
        validate_loss = evaluate(model, test_loader, criterion, device, writer, epoch)
        scheduler.step(validate_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.10f} - Train Loss: {train_loss:.4f} - Test Loss: {validate_loss:.4f}")
        writer.add_scalar('LearningRate', current_lr, epoch)

        # --- Early Stopping & Checkpoint ---
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            torch.save(model, model_save_path)
            print(f" New best model saved at epoch {epoch+1}")
            epochs_no_improve = 0

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f" Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    writer.close()
    
    return torch.load(model_save_path, weights_only=False)



def contrastive_loss(distances, labels, margin=0.5): 
        """
        Compute the contrastive loss for Siamese networks.
        
        Args:
            distances (Tensor): Predicted pairwise distances between embeddings.
            labels (Tensor): Binary labels (0 for similar, 1 for dissimilar).
            margin (float): Margin for dissimilar pairs.
        
        Returns:
            Tensor: Scalar loss.
        """
        labels = labels.float()
        loss_similar = (1 - labels) * distances.pow(2)
        loss_dissimilar = labels * F.relu(margin - distances).pow(2)
        return torch.mean(loss_similar + loss_dissimilar)