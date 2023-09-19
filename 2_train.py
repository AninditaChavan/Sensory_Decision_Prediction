import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import h5py
from pathlib import Path
from lstm import LSTMModel

def read_chunks_from_h5(fp: Path):
    chunks = []
    with h5py.File(fp, 'r') as f:
        sorted_keys = sorted(f.keys(), key=int)
        for key in sorted_keys:
            chunk = f[key][:]
            chunks.append(chunk)
    return chunks

class TimeSeriesDataset(Dataset):
    ''' Xt, X and y dataset. '''
    def __init__(self, time_series_chunks, features_array, target_array):
        filtered_data = [
            (ts, feat, target) for ts, feat, target in zip(time_series_chunks, features_array, target_array) if len(ts) > 0
        ]
        self.time_series_chunks, self.features_array, self.target_array = zip(*filtered_data)
        self.features_array = np.array(self.features_array)
        self.target_array = np.array(self.target_array)

    def __len__(self):
        return len(self.time_series_chunks)

    def __getitem__(self, idx):
        ts_chunk = torch.tensor(self.time_series_chunks[idx].reshape(-1, 10), dtype=torch.float32)
        features = torch.tensor(self.features_array[idx], dtype=torch.float32)
        target = torch.tensor(self.target_array[idx], dtype=torch.long) # torch.tensor(, dtype=torch.float32)
        return ts_chunk, features, target

def custom_collate(batch):
    time_series, features, targets = zip(*batch)
    lengths = torch.tensor([len(ts) for ts in time_series])
    time_series_padded = pad_sequence(time_series, batch_first=True)
    features = torch.stack(features)
    targets = torch.stack(targets)
    return time_series_padded, features, targets, lengths


def load_data(csv_path, h5_path):
    ''' Load Xt, X and y from csv and h5 files.s'''
    # Load CSV data
    target_column = 'Direction'
    csv_data = pd.read_csv(csv_path)
    encoder = OneHotEncoder()
    trial_type_one_hot = encoder.fit_transform(csv_data[['TrialType']]).toarray()
    features_array = np.hstack([csv_data[['AMRate', 'Latency']].values, trial_type_one_hot])
    # encode y 
    label_unique = csv_data[target_column].unique()
    print("unique labels:",label_unique)
    label_mapping = {label: i for i, label in enumerate(label_unique)}
    print("label_mapping",label_mapping)
    target_array = csv_data[target_column].map(label_mapping).values
    # Load time series data
    time_series_chunks = read_chunks_from_h5(h5_path)
    time_series_chunks = [np.array(chunk) for chunk in time_series_chunks]
    return time_series_chunks, features_array, target_array

def create_datasets(time_series_chunks, features_array, target_array, test_size=0.3):
    ''' 
    Split data into train and validation sets. 
    TODO: 
        1. analyse data distribution?
        2. add stratification?
        3. implement k-fold cross validation?
        4. implement data augmentation?
        5. implement data normalization?
    '''
    # Split data
    X_time_series_train, X_time_series_val, X_features_train, X_features_val, y_train, y_val = train_test_split(
        time_series_chunks, features_array, target_array, test_size=test_size, random_state=42)
    # Create custom datasets
    train_dataset = TimeSeriesDataset(X_time_series_train, X_features_train, y_train)
    val_dataset = TimeSeriesDataset(X_time_series_val, X_features_val, y_val)
    return train_dataset, val_dataset

def train_model(model, train_loader, val_loader, epochs=30):
    optimizer = torch.optim.Adam(model.parameters()) # TODO: add weight decay? better optimizer?
    criterion = nn.CrossEntropyLoss() # TODO: better loss function? weighted loss for class imbalance?
    for epoch in range(epochs):
        ''' Train loop'''
        model.train()
        train_loss = 0.0
        total_train = 0
        correct_train = 0
        for batch_time_series, batch_features, batch_target, lengths in train_loader:
            # batch_time_series = nn.utils.rnn.pad_sequence(batch_time_series, batch_first=True)
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_time_series, batch_features, torch.tensor(lengths))
            loss = criterion(outputs, batch_target)

            #Calculate accuracy
            _, predicted = torch.max(outputs.data,1) # Get the index (class) with maximum score
            total_train += batch_target.size(0)
            correct_train += (predicted == batch_target).sum().item()

            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_acc = correct_train / total_train
        average_train_loss = train_loss / len(train_loader)
        ''' Val loop'''
        model.eval()
        val_loss = 0
        total_val = 0
        correct_val = 0
        with torch.no_grad():
            for batch_time_series, batch_features, batch_target, lengths in val_loader:
                # batch_time_series = nn.utils.rnn.pad_sequence(batch_time_series, batch_first=True)
                outputs = model(batch_time_series, batch_features, torch.tensor(lengths))
                # print("output:",outputs)
                # print("target:",batch_target)
                loss = criterion(outputs, batch_target)
                #print(outputs.data)
                _, predicted = torch.max(outputs.data,1)
                # print("PREDICTED:",predicted)
                # print("GOLD:",batch_target)
                total_val += batch_target.size(0)
                correct_val += (predicted == batch_target).sum().item()
                val_loss += loss.item()
        val_acc = correct_val / total_val
        average_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}, Training Acc: {train_acc:.2f}, Validation Acc: {val_acc:.2f}')



all_time_series_chunks = []
all_features_array = []
all_target_array = []

data_path = Path('./data')
output_path = Path('./output')
video_folders = [folder for folder in data_path.iterdir() if folder.is_dir()]

for video_folder in video_folders:
    # Grab the first (and only) csv file in the data folder
    csv_path = next(video_folder.glob("*.csv"), None)
    
    # Corresponding h5 file in the output folder
    h5_folder = output_path / video_folder.name
    h5_path = next(h5_folder.glob("*.h5"), None)

    # Ensure both files exist before proceeding
    if h5_path and csv_path:
        print(h5_path,csv_path)
        time_series_chunks, features_array, target_array = load_data(csv_path, h5_path)
        all_time_series_chunks.extend(time_series_chunks)
        all_features_array.append(features_array)
        all_target_array.append(target_array)
    else:
        print(f"Missing h5 or csv file in {video_folder.name} folder.")

#Convert the lists to numpy arrays for further processing
all_features_array = np.vstack(all_features_array)
all_target_array = np.hstack(all_target_array)


print(set(all_target_array))

train_dataset, val_dataset = create_datasets(all_time_series_chunks, all_features_array, all_target_array)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=16,shuffle=False, collate_fn=custom_collate)

feature_size = features_array.shape[1]
all_features_size = all_features_array.shape[1]
print(feature_size,all_features_size)
model = LSTMModel(all_features_size)
train_model(model, train_loader, val_loader)