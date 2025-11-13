import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
from dataloader import create_dataloader
from inceptionResnetV1 import InceptionResnetV1, InceptionResnetV2

class ClassificationTrainer:
    def __init__(self, 
                 model,
                 train_loader,
                 test_loader,
                 learning_rate=0.0005,
                 weight_decay=1e-4,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer for classification tasks
        
        Args:
            model: Model instance
            train_loader: Training data loader
            test_loader: Test data loader
            learning_rate: Initial learning rate
            weight_decay: L2 regularization parameter
            device: Device to run the training on
        """
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), 
                                    lr=learning_rate, 
                                    weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                           mode='min',
                                           factor=0.5,
                                           patience=5,
                                           verbose=True)
        
        # Initialize metrics storage
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'accuracy': f'{total_correct/total_samples:.4f}'
            })

        epoch_accuracy = total_correct / total_samples
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss, epoch_accuracy
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model on test set"""
        self.model.eval()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for inputs, labels in self.test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            
            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        test_loss = running_loss / len(self.test_loader)
        return test_loss, accuracy
    
    def train(self, num_epochs, save_dir='checkpoints'):
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_test_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            train_loss, train_accuracy = self.train_epoch()
            test_loss, test_accuracy = self.evaluate()
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_accuracy)
            self.test_accuracies.append(test_accuracy)
            
            # Update learning rate
            self.scheduler.step(test_loss)
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Test Loss: {test_loss:.4f}')
            print(f'Train Accuracy: {train_accuracy:.4f}')
            print(f'Test Accuracy: {test_accuracy:.4f}')
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_test_loss': best_test_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch+1}.pth"))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(checkpoint, 
                           os.path.join(save_dir, 'best_model.pth'))
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot loss and accuracy curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        # Plot loss
        ax1.plot(self.train_losses, label='Train')
        ax1.plot(self.test_losses, label='Test')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Test Loss')
        # Plot accuracy
        ax2.plot(self.train_accuracies, label='Train')
        ax2.plot(self.test_accuracies, label='Test')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.set_title('Training and Test Accuracy')

        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train marmoset call classification model')
    
    # Data directories
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to the training data directory')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to the test data directory')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=60,
                        help='Number of training epochs (default: 60)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for L2 regularization (default: 1e-4)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='v2', choices=['v1', 'v2'],
                        help='Model version: v1 (InceptionResnetV1) or v2 (InceptionResnetV2) (default: v2)')
    
    # Save directory
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints (default: checkpoints)')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=43,
                        help='Random seed for reproducibility (default: 43)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Count number of classes from train directory
    num_classes = len([d for d in os.listdir(args.train_dir) 
                      if os.path.isdir(os.path.join(args.train_dir, d))])
    
    print(f"Detected {num_classes} classes in training directory")
    print(f"Training directory: {args.train_dir}")
    print(f"Test directory: {args.test_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Model: InceptionResnetV{args.model.upper()}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Create dataloaders
    train_loader, test_loader = create_dataloader(
        TRAIN_DATA_DIR=args.train_dir,
        TEST_DATA_DIR=args.test_dir,
        batch_size=args.batch_size,
    )
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'v1':
        model = InceptionResnetV1(device=device, classify=True, num_classes=num_classes)
    else:
        model = InceptionResnetV2(device=device, classify=True, num_classes=num_classes)
    
    print(f"Using device: {device}")
    print()
    
    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train model
    trainer.train(args.num_epochs, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
