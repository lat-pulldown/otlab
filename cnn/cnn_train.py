import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import pandas as pd

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *** CONFIGURATION ***
NUM_CLASSES = 100 
INPUT_FILE = 'train01.csv' 

def generate(name, window_size):
    print(f"Loading {name}...")
    try:
        path = os.path.join('data', name)
        df = pd.read_csv(path, header=None)
    except FileNotFoundError:
        try:
             df = pd.read_csv(name, header=None)
        except:
             print(f"Error: File {name} not found.")
             return None

    seq = df[0].values.tolist()
    # Adjust for 0-based indexing if your EIDs start at 1
    seq = [n - 1 for n in seq]

    inputs = []
    outputs = []
    
    # Create Sliding Windows
    for i in range(len(seq) - window_size):
        inputs.append(seq[i:i + window_size])
        outputs.append(seq[i + window_size])
        
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.long), torch.tensor(outputs, dtype=torch.long))
    print(f"Total training samples (windows) generated: {len(inputs)}")
    return dataset

# --- MODEL SIZE PRINTER ---
def get_model_size(model):
    """Calculates and prints the model size in Parameters and MB."""
    param_count = sum(p.numel() for p in model.parameters())
    # 4 bytes per float32 parameter
    size_mb = param_count * 4 / (1024 * 1024)
    print('-' * 40)
    print(f"Model Structure Summary:")
    print(f"  Total Parameters: {param_count:,}")
    print(f"  Estimated Size:   {size_mb:.4f} MB")
    print('-' * 40)
    return size_mb

# --- 1D-CNN MODEL DEFINITION ---
class CNNModel(nn.Module):
    def __init__(self, num_classes, hidden_size, window_size):
        super(CNNModel, self).__init__()
        # Embedding: Converts integer Log Keys to dense vectors
        self.embedding = nn.Embedding(num_classes, hidden_size)
        
        # Conv1d: The "Local Encoder"
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU()
        )
        
        # Flatten: We keep the full temporal sequence info
        self.flatten_size = (hidden_size * 4) * window_size
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        # Input x: [Batch, Window_Size]
        x = self.embedding(x)          # -> [Batch, Window, Hidden]
        x = x.permute(0, 2, 1)         # -> [Batch, Hidden, Window] (Required for Conv1d)
        
        x = self.conv1(x)              # -> [Batch, Hidden*2, Window]
        x = self.conv2(x)              # -> [Batch, Hidden*4, Window]
        
        x = x.view(x.size(0), -1)      # Flatten
        out = self.fc(x)               # -> [Batch, Num_Classes]
        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_epochs', default=50, type=int)
    parser.add_argument('-batch_size', default=2048, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    args = parser.parse_args()

    model_dir = 'model'
    
    # Load Data
    seq_dataset = generate(INPUT_FILE, args.window_size)
    if seq_dataset is None:
        exit()
        
    dataloader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # Initialize Model
    model = CNNModel(NUM_CLASSES, args.hidden_size, args.window_size).to(device)
    
    # --- PRINT MODEL SIZE ---
    get_model_size(model)
    # ------------------------
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training (1D-CNN)... Devices: {device}")
    start_time = time.time()
    
    for epoch in range(args.num_epochs): 
        model.train()
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            seq = seq.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, train_loss / len(dataloader)))
    
    print('Total Training Time: {:.3f}s'.format(time.time() - start_time))
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), f'{model_dir}/cnn_model.pth')
    print('Model saved to model/cnn_model.pth')