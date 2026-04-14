# ==========================================
# 2. Advanced Training Loop (with Scheduler & Early Stopping)
# ==========================================
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix

def train_model_advanced(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=20, early_stopping_patience=6):
    """
    Trains the model with Learning Rate Scheduling and Early Stopping.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    epochs_no_improve = 0 # Counter for Early Stopping

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Log for current Learning Rate (layer4)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')

            # --- VALIDATION PHASE LOGIC ---
            if phase == 'val':
                # 1. Step the scheduler based on validation loss
                scheduler.step(epoch_loss)
                
                # 2. Check for improvement (Early Stopping logic)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0 # Reset counter
                    print(f"🌟 New best model found! Loss: {best_loss:.4f}")
                    
                    # Store best predictions for the final report
                    best_val_labels = all_labels
                    best_val_preds = all_preds
                else:
                    epochs_no_improve += 1
                    print(f"Early stopping counter: {epochs_no_improve} out of {early_stopping_patience}")

        # Break out of the epochs loop if we haven't improved in a while
        if epochs_no_improve >= early_stopping_patience:
            print("\n🛑 Early stopping triggered! The model stopped improving.")
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Loss: {best_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_val_labels, best_val_preds




# ==========================================
# 2. Implementing Focal Loss
# ==========================================
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Class Classification.
    Formula: FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    
    - gamma: Focuses the model on hard-to-predict examples. (Default: 2.0)
    - alpha: Balances the importance of different classes.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha # Should be a tensor of weights for each class

    def forward(self, inputs, targets):
        # 1. Compute standard Cross Entropy Loss (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. Get the probability of the true class (p_t)
        # ce_loss = -log(p_t)  =>  p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 3. Compute the Focal Loss core: (1 - p_t)**gamma * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. Apply class weights (alpha) if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        # 5. Return mean or sum based on reduction parameter
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        



# ==========================================
# 3. Data Loader and Sampler Utilities
# ==========================================
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np

def create_balanced_train_loader(train_df, train_dataset, batch_size):
    """
    Creates a DataLoader with a WeightedRandomSampler to handle class imbalance.
    Returns the balanced DataLoader.
    """
    print("Setting up WeightedRandomSampler...")

    # 1. Map string labels to integers in our training dataframe
    label_map = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
    
    # We use .copy() to avoid SettingWithCopyWarning in pandas
    df_temp = train_df.copy()
    df_temp['label'] = df_temp['dx'].map(label_map)

    # 2. Calculate the count of each class
    class_counts = df_temp['label'].value_counts().sort_index().values

    # 3. Calculate weight for each class (Inverse frequency)
    class_weights = 1.0 / class_counts

    # 4. Assign a specific weight to EVERY SINGLE sample
    sample_weights = df_temp['label'].map(lambda x: class_weights[x]).values

    # 5. Convert to PyTorch tensor
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)

    # 6. Create the Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor, 
        num_samples=len(sample_weights_tensor), 
        replacement=True
    )

    # 7. Create the Training DataLoader
    # CRITICAL: 'shuffle' MUST be False when using a custom sampler.
    train_loader_balanced = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=False,
        num_workers=0
    )

    print("Balanced DataLoader ready.")
    return train_loader_balanced