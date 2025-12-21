#v2

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import pandas as pd

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *** CONFIGURATION ***
# Ensure this matches your max EID + 1 or a safe buffer
NUM_CLASSES = 100
INPUT_FILE = 'train.csv' 

def generate(name, window_size):
    print(f"Loading {name}...")
    try:
        path = os.path.join('../data', name)
        df = pd.read_csv(path, header=None)
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
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
        
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    print(f"Total training samples (windows) generated: {len(inputs)}")
    return dataset

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_epochs', default=50, type=int)
    parser.add_argument('-batch_size', default=2048, type=int)
    args = parser.parse_args()

    num_classes = NUM_CLASSES
    input_size = 1
    model_dir = 'model'
    log = 'adam_batch_size=' + str(args.batch_size) + ';epoch=' + str(args.num_epochs)

    # Load Data
    seq_dataset = generate(INPUT_FILE, args.window_size)
    if seq_dataset is None:
        exit()
        
    dataloader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # Initialize Model
    model = Model(input_size, args.hidden_size, args.num_layers, num_classes).to(device)
    
    # --- PRINT MODEL SIZE ---
    get_model_size(model)
    # ------------------------

    writer = SummaryWriter(log_dir='log/' + log)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    start_time = time.time()
    total_step = len(dataloader)
    
    print("Starting training...")
    for epoch in range(args.num_epochs): 
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            seq = seq.clone().detach().view(-1, args.window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        # Print every epoch or every few epochs
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, args.num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    
    print(f"Training Complete. Model saved to {model_dir}/{log}.pt")
    print(f"Total Logs Trained On (Windows): {len(seq_dataset)}")