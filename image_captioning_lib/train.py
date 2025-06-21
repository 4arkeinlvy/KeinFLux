from src.model import EncoderDecoder
from src.data_loader import train_loader, test_loader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
import os
from transformers import BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim

embed_size=300
attention_dim=256
encoder_dim=2048
decoder_dim=512
learning_rate = 3e-4

from torch import nn, optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Initialize the model
model = EncoderDecoder(
    embed_size=300,
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

# Initialize criterion and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Multiple GPUs support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    # Initialize the ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    print_every = 150
    checkpoint_base_dir = "/kaggle/working/"  # This is where Kaggle expects your output files
    checkpoint_dir = os.path.join(checkpoint_base_dir, 'checkpoints')
    
    # Create the checkpoint directory using the full path
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory created at: {checkpoint_dir}")
    
    # Track the best model
    best_test_loss = float('inf')
    best_checkpoint_path = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for idx, (image, captions) in enumerate(train_loader):
            image, captions = image.to(device), captions.to(device)
            
            optimizer.zero_grad()
            
            outputs, _ = model(image, captions)
            targets = captions[:, 1:]  # shifted target for teacher forcing
            loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculating accuracy, ignoring padding index
            _, predicted = outputs.max(2)
            mask = targets != tokenizer.pad_token_id
            correct_predictions += (predicted == targets).masked_select(mask).sum().item()
            total_predictions += mask.sum().item()
            
            if (idx + 1) % print_every == 0:
                avg_loss = running_loss / print_every
                accuracy = correct_predictions / total_predictions
                print(f"Epoch: {epoch}/{num_epochs}, Batch: {idx+1}/{len(train_loader)}, Loss: {avg_loss:.5f}, Accuracy: {accuracy:.5f}")
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
        
        # Evaluate the model on the test set
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.inference_mode():
            for image, captions in test_loader:
                image, captions = image.to(device), captions.to(device)
                outputs, _ = model(image, captions)
                targets = captions[:, 1:]
                loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.reshape(-1))
                test_loss += loss.item()
                
                _, predicted = outputs.max(2)
                mask = targets != tokenizer.pad_token_id
                test_correct += (predicted == targets).masked_select(mask).sum().item()
                test_total += mask.sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total
        print(f"Epoch: {epoch}/{num_epochs}, Test Loss: {avg_test_loss:.5f}, Test Accuracy: {test_accuracy:.5f}")
        
        # Step the scheduler
        scheduler.step(avg_test_loss)
        
        # Save model weights only if this is the best model so far
        try:
            # Check available disk space (in bytes)
            import shutil
            free_space = shutil.disk_usage(checkpoint_base_dir).free
            print(f"Available disk space: {free_space / (1024**3):.2f} GB")
            
            # Save checkpoint if we have sufficient space AND this is the best model
            if free_space > 1 * 1024**3:  # 1GB threshold
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    
                    # Remove the previous best checkpoint if it exists
                    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                        os.remove(best_checkpoint_path)
                        print(f"Removed previous best checkpoint")
                    
                    # Save the new best model
                    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pth")
                    
                    # Get model state
                    if isinstance(model, DataParallel):
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    
                    # Use a temporary file first to avoid corruption
                    temp_path = best_checkpoint_path + ".tmp"
                    torch.save(state_dict, temp_path)
                    
                    # If successful, rename to final path
                    os.rename(temp_path, best_checkpoint_path)
                    print(f"NEW BEST MODEL! Saved epoch {epoch} with test loss {avg_test_loss:.5f} at: {best_checkpoint_path}")
                else:
                    print(f"Test loss {avg_test_loss:.5f} not better than best {best_test_loss:.5f}. No checkpoint saved.")
                        
            else:
                print(f"Warning: Insufficient disk space ({free_space / (1024**3):.2f} GB). Skipping checkpoint save for epoch {epoch}")
                
        except Exception as e:
            print(f"Error saving checkpoint for epoch {epoch}: {str(e)}")
            print("Continuing training without saving checkpoint...")
        
    print(f"Training completed! Best model saved in: {checkpoint_dir}")
    print(f"Best test loss achieved: {best_test_loss:.5f}")
    return model

num_epochs = 100
train(model, train_loader, test_loader, criterion, optimizer, num_epochs, tokenizer)
torch.save(model.state_dict(), "model_weights.pth")