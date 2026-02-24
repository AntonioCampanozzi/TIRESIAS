import json
from src.data_analysis import data_parsing
import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from tslearn.metrics import cdist_dtw
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
from src.utils import FEATURES_MAPPING_BENDING, FEATURES_MAPPING_COMPRESSION, RESULTS_DIR, import_distance_matrix
from kneed import KneeLocator
from typing import Literal
import pickle
from scipy.signal import find_peaks

# 1. Funzione per scalare i dati
def scale_data(X):
    """
    Normalizza le serie temporali: media 0 e varianza 1 per ogni feature.
    X: array 3D (n_serie, n_timestamp, n_feature)
    """
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# 2. Funzione per calcolare la matrice delle distanze con DTW
def compute_distance_matrix(X, type: Literal['Bending', 'Compression']):
    """
    Calcola la matrice di distanza DTW a coppie.
    Utilizza n_jobs=-1 per usare tutti i core della CPU.
    """
    print("Distance matrix calculation...")
    # cdist_dtw restituisce una matrice quadrata (n_serie, n_serie)
    dist_matrix = cdist_dtw(X, n_jobs=-1)
    np.save(os.path.join(RESULTS_DIR, f"dtw_distance_matrix_{type}.npy"), dist_matrix)
    print("Distance matrix calculated.")
    return dist_matrix

# 3. Funzione per trovare il valore ottimale di K (Metodo della Silhouette)
def find_optimal_k_elbow(dist_matrix, max_k=10):
    """
    Valuta il numero di cluster usando il Metodo dell'Elbow.
    Si cerca il punto in cui la curva della somma delle distanze 
    'flette' vistosamente (il gomito).
    """
    inertias = []
    k_range = range(1, max_k + 1) # L'elbow parte da 1
    
    for k in k_range:
        km = KMedoids(n_clusters=k, metric="precomputed", random_state=42, method='pam')
        km.fit(dist_matrix)
        # inertia_ in KMedoids è la somma delle distanze dei campioni dal loro medoide
        inertias.append(km.inertia_)
    
    return inertias

# 4. Funzione per raggruppare e conservare i risultati
def organize_cluster_results(X_original, labels, sample_names, type: Literal['Bending', 'Compression']):
    """
    Crea un dizionario dove ogni chiave è il numero del cluster 
    e il valore è una lista di nomi dei campioni o dati.
    """
    clusters_dict = {}
    for cluster_id in np.unique(labels):
        # Trova gli indici dei campioni appartenenti a questo cluster
        indices = np.where(labels == cluster_id)[0]
        clusters_dict[str(cluster_id)] = [sample_names[i] for i in indices]
        with open(os.path.join(RESULTS_DIR, f'cluster_mapping_{type}.json'), 'w') as f:
            json.dump(clusters_dict, f, indent=4)
    return clusters_dict

def clusterize(dist_matrix, max_k, type: Literal['Bending', 'Compression']):
    intertias = find_optimal_k_elbow(dist_matrix, max_k=max_k)
    
    best_k=KneeLocator(range(1, max_k + 1), intertias, curve='convex', direction='decreasing').elbow
    print(f"Optimal K found: {best_k}")
    
    model = KMedoids(n_clusters=best_k, metric="precomputed", random_state=42, method='pam')
    
    print("Clustering with KMedoids...")
    
    final_labels = model.fit_predict(dist_matrix)
    
    print("Clustering completed. Saiving medoids indices...")
    
    medoids_indices = model.medoid_indices_
    
    pickle.dump(medoids_indices, open(os.path.join(RESULTS_DIR, f'medoids_{type}.pkl'), 'wb'))
    
    print("Saving results...")
    
    results = organize_cluster_results(X_bending, final_labels, names_list, type=type)
    
    

def get_thresholds(medoid_indices, dataframes, type: Literal['Bending', 'Compression']):
    thresholds=[]
    for i in medoid_indices:
        df = dataframes[i]
        force_series = df['Force applied (N)'].values
            #the curve analysis on bending confirmed that the point of breakeage si most likely on the peak pof the curve
        if type == 'Compression':
            #For compression, we have two peaks, we consider the first one as the point of breakage
            peaks, _ = find_peaks(force_series, prominence=5000)
            
            if len(peaks) > 0:
                peak_force_point = peaks[0]
            else:
                peak_force_point = np.argmax(force_series)
        else:
            peak_force_point= np.argmax(force_series)
            
        threshold=peak_force_point/len(force_series)
        print(f"Medoid index: {i}, Threshold: {threshold:.2f}")    
        thresholds.append(threshold)
    print(f"Mean threshold: {np.mean(thresholds):.2f},\nStd threshold: {np.std(thresholds):.2f},\nMedian threshold: {np.median(thresholds):.2f}")

if __name__ == "__main__":
    
    # -------------------BENDING DATA--------------------
    dataframes_bending=data_parsing.parse_data('Bending', sep='\t', features_mapping=FEATURES_MAPPING_BENDING, label_col='static:Bending strength')
    
    dataframes_cleaned=[df[['Deformation(mm)', 'Force applied (N)']] for df in dataframes_bending]
    
    X_bending=data_parsing.get_tsdata(dataframes_cleaned, type='Bending')
    
    names_list = [df['static:Sample name'].iloc[0] for df in dataframes_bending]

    X_scaled = scale_data(X_bending)
    
    dist_matrix = import_distance_matrix('Bending')
    
    if dist_matrix is None:
        dist_matrix = compute_distance_matrix(X_scaled, type='Bending')
    
    clusterize(dist_matrix, max_k=10, type='Bending')
    
    get_thresholds(pickle.load(open(os.path.join(RESULTS_DIR, f'medoids_Bending.pkl'), 'rb')), dataframes_bending, type='Bending')
    
    # -------------------COMPRESSION DATA--------------------
    dataframes_compression=data_parsing.parse_data('Compression', sep=';', features_mapping=FEATURES_MAPPING_COMPRESSION, label_col='static:Compressive stress at maximum strain')
    
    dataframes_cleaned=[df[['Compression(%)', 'Deflection at standard load(%)', 'Force applied (N)']] for df in dataframes_compression]
    
    X_compression=data_parsing.get_tsdata(dataframes_cleaned, type='Compression')
    
    names_list = [df['static:Sample name'].iloc[0] for df in dataframes_compression]

    X_scaled = scale_data(X_compression)

    # 2. Computing DTW distance matrix
    
    dist_matrix = import_distance_matrix('Compression')
    
    if dist_matrix is None:
        resampler = TimeSeriesResampler(sz=4000) 
        X_resampled = resampler.fit_transform(X_scaled)
        print(X_resampled.shape)
        dist_matrix = compute_distance_matrix(X_resampled, type='Compression')

    clusterize(dist_matrix, max_k=10, type='Compression')
    
    get_thresholds(pickle.load(open(os.path.join(RESULTS_DIR, f'medoids_Compression.pkl'), 'rb')), dataframes_compression, type='Compression')