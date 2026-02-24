import json
import os
from typing_extensions import Literal
import pickle
import numpy as np

PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
EXPLORATIVE_ANALYSIS_DIR = os.path.join(DATA_DIR, 'explorative_analysis')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

FEATURES_MAPPING_BENDING={
    'Vorkraft': 'pre-load',
    'Geschwindigkeit Biegemodul': 'Bending module speed',
    'Prüfgeschwindigkeit': 'Test speed',
    'Probenbezeichnung': 'Sample name',
    'Biegemodul': 'Bending module', #tells us how much resistent the material is
    'Biegespannung bei Normdurchbiegung': 'Bending stress at maximum allowed deflection',
    'Biegefestigkeit': 'Bending strength', #resistance to deflection before breaking
    'Randfaserdehnung bei Biegefestigkeit': 'Deformation at bending strength', #deformation (strain) at the extreme fiber when the material reaches its maximum bending strength.
    'Zeit': 'Time(s)',
    'Verformung': 'Deformation(mm)',
    'Standardweg':'Deflection at standard load (mm)',
    'Traversenweg':'Crosshead displacement (mm)',
    'Standardkraft':'Force applied (N)'
}

FEATURES_MAPPING_COMPRESSION={
    'Probenbezeichnung': 'Sample name',
    '0.2% Stauchgrenze': '0.2% Yield Strength (Compressive)',
    'Druckfestigkeit': 'Maximum compressive strength',
    'Gesamtstauchung bei Druckfestigkeit': 'Total strain stress at maximum compressive strength',
    'Druckspannung bei maximaler Stauchung': 'Compressive stress at maximum strain',
    'Druckmodul per Hysterese': 'Compressive Modulus via Hysteresis',
    'Druckmodul': 'Compressive Modulus',
    'Zeit': 'Time(s)',
    'Prüfzeit': 'Test time',
    'Standardweg':'Deflection at standard load(%)',
    'Traversenweg absolut':'Crosshead displacement(mm)',
    'Stauchung': 'Compression(%)',
    'Nominelle Stauchung': 'Deformation(mm)',#compression of the sample with respect to its original length
    'Standardkraft':'Force applied (N)'
}

def open_clusters_mapping(type: Literal['Bending', 'Compression']):
    try:
        with open(os.path.join(RESULTS_DIR, f'cluster_mapping_{type}.json'), 'r') as f:
            clusters_dict = json.load(f)
            return clusters_dict
    except FileNotFoundError:
        print(f"Error: cluster_mapping_{type}.json not found in {RESULTS_DIR}")
        exit(1)
        
def get_medoids_from_mapping(type: Literal['Bending', 'Compression']):
    try:
        with open(os.path.join(RESULTS_DIR, f'medoids_{type}.pkl'), 'rb') as f:
            medoids = pickle.load(f)
            return medoids
    except FileNotFoundError:
        print(f"Error: medoids_{type}.pkl not found in {RESULTS_DIR}")
        exit(1)
        
def import_distance_matrix(type: Literal['Bending', 'Compression']):
    try:
        print('...loading distance matrix...')
        dist_matrix = np.load(os.path.join(RESULTS_DIR, f"dtw_distance_matrix_{type}.npy"))
        return dist_matrix
    except FileNotFoundError:
        print(f"Error: dtw_distance_matrix_{type}.npy not found in {RESULTS_DIR}")
        return None
            