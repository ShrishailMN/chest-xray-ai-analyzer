import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from medical_report_generator import MedicalReportGenerator

class MedicalReportDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, max_length=50):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.max_length = max_length
        self.create_vocab()
        
    def create_vocab(self):
        # Create vocabulary from reports
        words = set()
        for report in self.data['report']:
            words.update(report.lower().split())
            
        self.word2idx = {
            '<pad>': 0,
            '<start>': 1,
            '<end>': 2,
            '<unk>': 3
        }
        
        for word in sorted(words):
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
    def tokenize_report(self, report):
        # Convert report to token indices
        tokens = ['<start>'] + report.lower().split()[:self.max_length-2] + ['<end>']
        token_ids = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        
        # Pad sequence
        if len(token_ids) < self.max_length:
            token_ids += [self.word2idx['<pad>']] * (self.max_length - len(token_ids))
        return torch.tensor(token_ids)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['image_file'])
        image = Image.open(img_name).convert('RGB')
        report = self.data.iloc[idx]['report']
        
        if self.transform:
            image = self.transform(image)
            
        report_tensor = self.tokenize_report(report)
        return image, report_tensor

def verify_dataset(csv_file, img_dir):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    missing_images = []
    
    for img_file in df['image_file']:
        # Check for both .jpg and .jpeg versions of the file
        base_name = os.path.splitext(img_file)[0]
        jpg_path = os.path.join(img_dir, base_name + '.jpg')
        jpeg_path = os.path.join(img_dir, base_name + '.jpeg')
        
        if not (os.path.exists(jpg_path) or os.path.exists(jpeg_path)):
            missing_images.append(img_file)
    
    if missing_images:
        raise FileNotFoundError(
            f"Following images are missing in {img_dir}:\n" + 
            "\n".join(missing_images)
        )
    
    print(f"Found {len(df)} valid image-report pairs")

def train():
    # Add dataset verification
    csv_file = 'dataset/train_reports.csv'
    img_dir = 'dataset/train_images'
    verify_dataset(csv_file, img_dir)
    
    # Hyperparameters
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Create dataset
    dataset = MedicalReportDataset(
        csv_file='dataset/train_reports.csv',
        img_dir='dataset/train_images',
        transform=transform
    )
    
    # Create data loader with safer settings
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Change from 2 to 0 to avoid multiprocessing issues
        pin_memory=False,  # Disable pin_memory
        persistent_workers=False  # Disable persistent workers
    )
    
    # Initialize model
    model = MedicalReportGenerator(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=dataset.vocab_size,
        num_layers=NUM_LAYERS
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.word2idx['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        try:
            for batch_idx, (images, reports) in enumerate(progress_bar):
                try:
                    images = images.to(device)
                    reports = reports.to(device)
                    
                    # Forward pass
                    outputs = model(images, reports)
                    
                    # Calculate loss
                    loss = criterion(
                        outputs.view(-1, dataset.vocab_size),
                        reports[:, 1:].contiguous().view(-1)
                    )
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    avg_loss = total_loss / (batch_idx + 1)
                    
                    # Update progress bar
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    
                    # Clear memory
                    del outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab': dataset.word2idx,
                    'loss': avg_loss
                }
                torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pth')
                
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {str(e)}")
            # Save emergency checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': dataset.word2idx,
                'loss': avg_loss if 'avg_loss' in locals() else None
            }, f'checkpoints/emergency_checkpoint_epoch_{epoch+1}.pth')
            continue
    
    print("Training finished!")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': dataset.word2idx
    }, 'medical_report_generator.pth')

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('dataset/train_images', exist_ok=True)
    
    train() 