import json
import os
import model
from src.model.LstmDataset import LstmDataset, EarlyPredLSTM # Importi le tue classi
import torch
from torch.utils.data import DataLoader
from src.data_analysis.data_parsing import parse_data
from src.utils import FEATURES_MAPPING_BENDING, MODEL_DIR, RESULTS_DIR, FEATURES_MAPPING_COMPRESSION
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal


# 1. Configurazione
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRIDE = 50

def rebuild_df(X, y, original_lst, label_col):
    df_list = []
    cols= [col for col in original_lst[0].columns if col not in ['static:Sample name', label_col]]
    for x_scaled, y_val, df_orig in zip(X, y, original_lst):
            new_df = df_orig.copy()
            new_df[cols] = x_scaled
            new_df[label_col] = y_val[0]
            df_list.append(new_df)
    return df_list

def prepare_training(dataframes, label_col):
    data_lst, val_lst = train_test_split(dataframes, test_size=0.2, random_state=42)
    train_lst, test_lst = train_test_split(data_lst, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
    scaler_x = StandardScaler()
    cols= [col for col in dataframes[0].columns if col not in ['static:Sample name', label_col]]
    X_train = [df[cols].values for df in train_lst]
    X_val = [df[cols].values for df in val_lst]
    X_test = [df[cols].values for df in test_lst]
    
    y_train = [df[label_col].iloc[0] for df in train_lst]
    y_val = [df[label_col].iloc[0] for df in val_lst]
    y_test = [df[label_col].iloc[0] for df in test_lst]
    
    scaler_x.fit(np.vstack([df[cols].values for df in train_lst])) #data_lst=train+val
    X_train = [scaler_x.transform(x) for x in X_train]
    X_val = [scaler_x.transform(x) for x in X_val]
    X_test = [scaler_x.transform(x) for x in X_test]
    
    scaler_y = StandardScaler()
    y_train = np.array(y_train).reshape(-1, 1)
    y_val = np.array(y_val).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    
    scaler_y.fit(y_train)
    y_train = scaler_y.transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)
    
    train_lst_scaled = rebuild_df(X_train, y_train, train_lst, label_col)
    val_lst_scaled = rebuild_df(X_val, y_val, val_lst, label_col)
    test_lst_scaled = rebuild_df(X_test, y_test, test_lst, label_col)
    
    train_ds = LstmDataset(train_lst_scaled, stride=STRIDE, label=label_col, mode='train')
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_ds = LstmDataset(val_lst_scaled, stride=STRIDE, label=label_col, mode='val')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_ds = LstmDataset(test_lst_scaled, stride=STRIDE, label=label_col, mode='test')
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader, scaler_y

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, max_patience, model_index, type: Literal['Bending', 'Compression']):
    
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (X, y, lengths) in enumerate(train_loader):
            X = X.squeeze(0).to(DEVICE) 
            y = y.squeeze(0).to(DEVICE) 
            lengths = lengths.squeeze(0).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X, lengths) 
            loss = criterion(outputs.view(-1), y.float())
            loss.backward()
    
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)  # Dividiamo per il numero effettivo di batch utilizzati
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Valutazione sul validation set (opzionale)
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, (X_val, y_val, lengths_val) in enumerate(val_loader):
                X_val = X_val.squeeze(0).to(DEVICE)
                y_val = y_val.squeeze(0).to(DEVICE)
                lengths_val = lengths_val.squeeze(0).to(DEVICE)
                outputs_val = model(X_val, lengths_val)
                val_loss += criterion(outputs_val.view(-1), y_val.float()).item()
            avg_val_loss = val_loss / len(val_loader)  # Dividiamo per il numero effettivo di batch utilizzati
            
            print(f"Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_model_cluster_{model_index}_{type}.pth"))
            print(f"model saved at epoch {epoch+1}")
        else:
            # DELUSIONE: La loss non è migliorata
            patience_counter += 1
            print(f"Patience: {patience_counter}/{max_patience}")

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}/{epochs}")
            break
    
def test_model(model, test_loader, criterion):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_test, y_test, lengths_test in test_loader:
                X_test = X_test.squeeze(0).to(DEVICE)
                y_test = y_test.squeeze(0).to(DEVICE)
                lengths_test = lengths_test.squeeze(0).to(DEVICE)
                outputs_test = model(X_test, lengths_test)
                test_loss += criterion(outputs_test.view(-1), y_test.float()).item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")
        


def find_early_stopping_point(errors, tau=0.9):
    y = np.array(errors)

    y0 = y[0]
    ymin = y.min()
    total_gain = y0 - ymin

    # Caso 1: nessun miglioramento reale
    if total_gain <= 1e-12:
        return 0

    gains = y0 - y
    ratio = gains / total_gain

    indices = np.where(ratio >= tau)[0]

    # Caso 2: non raggiunge mai tau
    if len(indices) == 0:
        return len(y) - 1

    return indices[0]

def predict_prefix(model, test_loader, scaler_y, device, curve_idx):
    """
    Usa direttamente il test_loader che ha già la logica randrange/padding/campionamento.
    """
    earliness_metrics = []
    model.eval()
    all_preds = []
    all_truths = []
    with torch.no_grad():
        for X_batch, y_batch, lengths in test_loader:
            X_batch = X_batch.squeeze(0).to(device)  # Togliamo la dimensione extra del DataLoader
            y_batch = y_batch.squeeze(0).to(device)
            lengths = lengths.squeeze(0).to(device)
            
            # Forward pass con pack_padded_sequence già integrato nel modello
            outputs_scaled = model(X_batch, lengths)
            
            # Invertiamo lo scaling per tornare ai MPa/Newton
            preds_real = scaler_y.inverse_transform(outputs_scaled.cpu().numpy())
            truths_real = scaler_y.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1))
            
            pred_error = ((abs(preds_real - truths_real) / truths_real) * 100).mean()
            earliness_metrics.append(pred_error)
        earliness_metrics = pd.Series(earliness_metrics).rolling(window=20, min_periods=1).mean()
        plt.plot(range(1, len(earliness_metrics) + 1), earliness_metrics, label='Earliness (smoothed)')
        plt.xlabel('Prefix Index')
        plt.ylabel('APE (%)')
        plt.title('Earliness Plot')
        point=find_early_stopping_point(earliness_metrics, tau=0.9)
        plt.axvline(x=point, color='r', linestyle='--', label=f'Early Stopping Point: {point}')
        plt.legend()
        dir = os.path.join(RESULTS_DIR, 'earliness_curves_LSTM')
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, f"earliness_curve_{curve_idx}.png"))
        plt.close()
        earliness=(point+1)/len(earliness_metrics) * 100
        print(f"Earliness: {earliness:.2f}%")
    
if __name__ == "__main__":
    print(DEVICE)
    dataframes_bending = parse_data('Bending', sep='\t', features_mapping=FEATURES_MAPPING_BENDING, label_col='static:Bending strength')
    dataframes_bending = [df[['static:Sample name', 'Deformation(mm)', 'Force applied (N)', 'static:Bending strength']] for df in dataframes_bending]
    clustering_results_bending = os.path.join(RESULTS_DIR, 'cluster_mapping_Bending.json')
    try:
        with open(clustering_results_bending, 'r') as f:
            cluster_mapping = json.load(f)
    except FileNotFoundError:
        print(f"Error: {clustering_results_bending} not found. Please run the clustering step first.")
        exit(1)
    dataset_0 = [df for df in dataframes_bending if df['static:Sample name'].iloc[0] in cluster_mapping['0']]
    dataset_1 = [df for df in dataframes_bending if df['static:Sample name'].iloc[0] in cluster_mapping['1']]
    dataset_2 = [df for df in dataframes_bending if df['static:Sample name'].iloc[0] in cluster_mapping['2']]
    dataset_3 = [df for df in dataframes_bending if df['static:Sample name'].iloc[0] in cluster_mapping['3']]
    
    train_loader, val_loader, test_loader, y_scaler = prepare_training(dataset_0, label_col='static:Bending strength')
    model_0 = EarlyPredLSTM(input_size=2, hidden_size=128).to(DEVICE)
    model_1 = EarlyPredLSTM(input_size=2, hidden_size=128).to(DEVICE)
    model_2 = EarlyPredLSTM(input_size=2, hidden_size=128).to(DEVICE)
    model_3 = EarlyPredLSTM(input_size=2, hidden_size=128).to(DEVICE)
   
    
    train_loader_1, val_loader_1, test_loader_1, y_scaler_1 = prepare_training(dataset_1, label_col='static:Bending strength')
    train_loader_2, val_loader_2, test_loader_2, y_scaler_2 = prepare_training(dataset_2, label_col='static:Bending strength')
    train_loader_3, val_loader_3, test_loader_3, y_scaler_3 = prepare_training(dataset_3, label_col='static:Bending strength')
    
    models_list = [model_0, model_1, model_2, model_3]
    loaders = [(train_loader, val_loader), (train_loader_1, val_loader_1), 
           (train_loader_2, val_loader_2), (train_loader_3, val_loader_3)]

    for i, model in enumerate(models_list):
        path = os.path.join(MODEL_DIR, f"best_model_cluster_{i}_Bending.pth")
    
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            print(f"Model {i} loaded successfully.")
        else:
            print(f"Model {i} not found. Training...")
            train_model(model,
                    loaders[i][0], # train_loader
                    loaders[i][1], # val_loader
                    torch.optim.Adam(model.parameters(), lr=1e-2),
                    nn.HuberLoss(delta=1.0),
                    epochs=200,
                    max_patience=15,
                    model_index=i,
                    type='Bending')
    
    test_model(model_0, test_loader, nn.HuberLoss(delta=1.0))
    test_model(model_1, test_loader_1, nn.HuberLoss(delta=1.0))
    test_model(model_2, test_loader_2, nn.HuberLoss(delta=1.0))
    test_model(model_3, test_loader_3, nn.HuberLoss(delta=1.0))
    
    predict_prefix(model_0, test_loader, y_scaler, DEVICE, 0)
    predict_prefix(model_1, test_loader_1, y_scaler_1, DEVICE, 1)
    predict_prefix(model_2, test_loader_2, y_scaler_2, DEVICE, 2)
    predict_prefix(model_3, test_loader_3, y_scaler_3, DEVICE, 3)
    
    dataframes_compression = parse_data('Compression', sep=';', features_mapping=FEATURES_MAPPING_COMPRESSION, label_col='static:Compressive stress at maximum strain')
    dataframes_compression = [df[['static:Sample name', 'Compression(%)', 'Deflection at standard load(%)', 'Force applied (N)', 'static:Compressive stress at maximum strain']] for df in dataframes_compression]
    clustering_results_compression = os.path.join(RESULTS_DIR, 'cluster_mapping_Compression.json')
    try:
        with open(clustering_results_compression, 'r') as f:
            cluster_mapping = json.load(f)
    except FileNotFoundError:
        print(f"Error: {clustering_results_compression} not found. Please run the clustering step first.")
        exit(1)
    
    dataset_0 = [df for df in dataframes_compression if df['static:Sample name'].iloc[0] in cluster_mapping['0']]
    dataset_1 = [df for df in dataframes_compression if df['static:Sample name'].iloc[0] in cluster_mapping['1']]
    dataset_2 = [df for df in dataframes_compression if df['static:Sample name'].iloc[0] in cluster_mapping['2']]
    
    model_0 = EarlyPredLSTM(input_size=3, hidden_size=128).to(DEVICE)
    model_1 = EarlyPredLSTM(input_size=3, hidden_size=128).to(DEVICE)
    model_2 = EarlyPredLSTM(input_size=3, hidden_size=128).to(DEVICE)
    
    train_loader, val_loader, test_loader, y_scaler = prepare_training(dataset_0, label_col='static:Compressive stress at maximum strain')
    train_loader_1, val_loader_1, test_loader_1, y_scaler_1 = prepare_training(dataset_1, label_col='static:Compressive stress at maximum strain')
    train_loader_2, val_loader_2, test_loader_2, y_scaler_2 = prepare_training(dataset_2, label_col='static:Compressive stress at maximum strain')
    models_list = [model_0, model_1, model_2]
    loaders = [(train_loader, val_loader), (train_loader_1, val_loader_1), 
           (train_loader_2, val_loader_2)]

    for i, model in enumerate(models_list):
        path = os.path.join(MODEL_DIR, f"best_model_cluster_{i}_Compression.pth")
    
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            print(f"Model {i} loaded successfully.")
        else:
            print(f"Model {i} not found. Training...")
            train_model(model,
                    loaders[i][0], # train_loader
                    loaders[i][1], # val_loader
                    torch.optim.Adam(model.parameters(), lr=1e-2),
                    nn.HuberLoss(delta=1.0),
                    epochs=200,
                    max_patience=15,
                    model_index=i,
                    type='Compression')
    
    test_model(model_0, test_loader, nn.HuberLoss(delta=1.0))
    test_model(model_1, test_loader_1, nn.HuberLoss(delta=1.0))
    test_model(model_2, test_loader_2, nn.HuberLoss(delta=1.0))
    
    predict_prefix(model_0, test_loader, y_scaler, DEVICE, 0)
    predict_prefix(model_1, test_loader_1, y_scaler_1, DEVICE, 1)
    predict_prefix(model_2, test_loader_2, y_scaler_2, DEVICE, 2)