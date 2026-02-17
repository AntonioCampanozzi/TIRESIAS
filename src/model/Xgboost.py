
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.utils import PROCESSED_DATA_DIR, FEATURES_MAPPING_BENDING, FEATURES_MAPPING_COMPRESSION, RESULTS_DIR
from src.data_analysis.data_parsing import parse_data
from matplotlib import pyplot as plt
from src.data_analysis.prefixDataset import prefixDataset
import numpy as np
from src.data_analysis.data_parsing import parse_data
from typing import Literal



def split(dataframes, test_size=0.2, random_state=42):
        data_lst, val_lst = train_test_split(dataframes, test_size=test_size, random_state=random_state)
        train_lst, test_lst = train_test_split(data_lst, test_size=0.25, random_state=random_state) # 0.25 x 0.8 = 0.2
        return train_lst, val_lst, test_lst
    
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

def predict_prefix(model, group, group_truth, idx, cluster_id, type: Literal['Bending', 'Compression']):
    earliness_metrics = []
    preds=model.predict(group)
    for p in range(len(preds)):
        #CALCOLO L'ABSOLUTE PERCENTAGE ERROR TRA LA PREDIZIONE E IL VALORE REALE
        pred_error = (abs(preds[p] - group_truth)/group_truth)*100
        earliness_metrics.append(pred_error)
    smooth_earliness_metrics = pd.Series(earliness_metrics).rolling(window=20,min_periods=1).mean()
    
    if idx < 5:
        plt.plot(range(1, len(group)+1), earliness_metrics, alpha=0.3, label='raw')
        plt.plot(range(1, len(group)+1), smooth_earliness_metrics, label='smoothed')
        plt.xlabel('prefix')
        plt.ylabel('APE (%)')
        plt.title('Earliness Plot')
        curve_dir=os.path.join(RESULTS_DIR, f'{type}_earliness_curves')
        os.makedirs(curve_dir, exist_ok=True)
        plt.savefig(os.path.join(curve_dir, f"{type}_earliness_curve_sample_c{cluster_id}_{idx}.png"))
        plt.close()
        
    early_stopping_point = find_early_stopping_point(smooth_earliness_metrics, tau=0.9)
    
    earliness=(early_stopping_point+1)/len(group) * 100
    return earliness

def train_and_avaluate(dataset, cluster_id, stride, label_col, type: Literal['Bending', 'Compression']):
    dir = os.path.join(PROCESSED_DATA_DIR, f"{type}")
    if (os.path.exists(os.path.join(dir, f"{type}_train_dataset_{cluster_id}.csv")) and 
            os.path.exists(os.path.join(dir, f"{type}_val_dataset_{cluster_id}.csv")) and 
            os.path.exists(os.path.join(dir, f"{type}_test_dataset_{cluster_id}.csv"))):
        print(f"Loading preprocessed datasets for cluster {cluster_id}...")
        dataset_train = pd.read_csv(os.path.join(dir, f"{type}_train_dataset_{cluster_id}.csv"))
        dataset_val = pd.read_csv(os.path.join(dir, f"{type}_val_dataset_{cluster_id}.csv"))
        dataset_test = pd.read_csv(os.path.join(dir, f"{type}_test_dataset_{cluster_id}.csv"))
    else:
        train, val, test = split(dataset)
        dataset_train = prefixDataset(train, label_col=label_col, stride=stride).dataset
        dataset_train.to_csv(os.path.join(dir, f"{type}_train_dataset_{cluster_id}.csv"), index=False)
        dataset_val = prefixDataset(val, label_col=label_col, stride=stride).dataset
        dataset_val.to_csv(os.path.join(dir, f"{type}_val_dataset_{cluster_id}.csv"), index=False)
        dataset_test = prefixDataset(test, label_col=label_col, stride=stride).dataset
        dataset_test.to_csv(os.path.join(dir, f"{type}_test_dataset_{cluster_id}.csv"), index=False)
    
    X_train = dataset_train.drop(columns=['ID', 'label'])
    y_train = dataset_train['label']
    X_val = dataset_val.drop(columns=['ID', 'label'])
    y_val = dataset_val['label']
    X_test = dataset_test.drop(columns=['label'])
    y_test = dataset_test[['ID','label']]
    
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False 
    )
    
    test_model(model, dataset_test)
    
    groups = X_test.groupby('ID')

    earliness_scores = []
    
    for i, (name, group) in enumerate(groups):
        group=group.drop(columns=['ID'])
        group_truth = y_test[y_test['ID'] == name]['label'].values[0]
        earliness=predict_prefix(model, group, group_truth, i, cluster_id,type)
        earliness_scores.append(earliness)
    avg_earliness = np.mean(earliness_scores).round(2)
    median_earliness = np.median(earliness_scores).round(2)
    std_earliness = np.std(earliness_scores).round(2)
    
    
    print(f"Average Earliness on cluster {cluster_id}: {avg_earliness:.2f}%")
    print(f"Median Earliness on cluster {cluster_id}: {median_earliness:.2f}%")
    print(f"Std Earliness on cluster {cluster_id}: {std_earliness:.2f}%")
    
    return avg_earliness

def test_model(model, test_set):
    X_test = test_set.drop(columns=['ID', 'label'])
    y_test = test_set['label']
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R^2: {r2:.4f}")

if __name__ == "__main__":
    
    dataframes_bending=parse_data('Bending', sep='\t', features_mapping=FEATURES_MAPPING_BENDING, label_col='static:Bending strength')
    dataframes_bending = [df[['static:Sample name', 'Deformation(mm)', 'Force applied (N)', 'static:Bending strength']] for df in dataframes_bending]

    with open(os.path.join(RESULTS_DIR, "cluster_mapping_Bending.json"), 'r') as f:
        cluster_mapping = json.load(f)
    
    dataframes_bending_c0 = [df for df in dataframes_bending if df['static:Sample name'].iloc[0] in cluster_mapping["0"]]
    print(f"Cluster 0 size: {len(dataframes_bending_c0)}")
    dataframes_bending_c1 = [df for df in dataframes_bending if df['static:Sample name'].iloc[0] in cluster_mapping["1"]]
    dataframes_bending_c2 = [df for df in dataframes_bending if df['static:Sample name'].iloc[0] in cluster_mapping["2"]]
    dataframes_bending_c3 = [df for df in dataframes_bending if df['static:Sample name'].iloc[0] in cluster_mapping["3"]]
    
    earliness_c0 = train_and_avaluate(dataframes_bending_c0, cluster_id=0, stride=10, label_col='static:Bending strength', type='Bending')
    earliness_c1 = train_and_avaluate(dataframes_bending_c1, cluster_id=1, stride=10, label_col='static:Bending strength', type='Bending')
    earliness_c2 = train_and_avaluate(dataframes_bending_c2, cluster_id=2, stride=10, label_col='static:Bending strength', type='Bending')
    earliness_c3 = train_and_avaluate(dataframes_bending_c3, cluster_id=3, stride=10, label_col='static:Bending strength', type='Bending')    
    earliness_total = train_and_avaluate(dataframes_bending, cluster_id=999, stride=10, label_col='static:Bending strength', type='Bending')
    
    dataframes_compression=parse_data('Compression', sep=';', features_mapping=FEATURES_MAPPING_COMPRESSION, label_col='static:Compressive stress at maximum strain')
    dataframes_compression = [df[['static:Sample name', 'Compression(%)', 'Deflection at standard load(%)', 'Force applied (N)', 'static:Compressive stress at maximum strain']] for df in dataframes_compression]
    
    with open(os.path.join(RESULTS_DIR, "cluster_mapping_Compression.json"), 'r') as f:
        cluster_mapping = json.load(f)
    
    dataframes_compression_c0 = [df for df in dataframes_compression if df['static:Sample name'].iloc[0] in cluster_mapping["0"]]
    dataframes_compression_c1 = [df for df in dataframes_compression if df['static:Sample name'].iloc[0] in cluster_mapping["1"]]
    dataframes_compression_c2 = [df for df in dataframes_compression if df['static:Sample name'].iloc[0] in cluster_mapping["2"]]
    
    earliness_c0 = train_and_avaluate(dataframes_compression_c0, cluster_id=0, stride=10, label_col='static:Compressive stress at maximum strain', type='Compression')
    earliness_c1 = train_and_avaluate(dataframes_compression_c1, cluster_id=1, stride=10, label_col='static:Compressive stress at maximum strain', type='Compression')
    earliness_c2 = train_and_avaluate(dataframes_compression_c2, cluster_id=2, stride=10, label_col='static:Compressive stress at maximum strain', type='Compression')
    earliness_total = train_and_avaluate(dataframes_compression, cluster_id=999, stride=10, label_col='static:Compressive stress at maximum strain', type='Compression')
    