from src.utils import RAW_DATA_DIR, INTERIM_DATA_DIR
from pathlib import Path
import os
from typing import Literal
from tqdm import tqdm
import pandas as pd
from tslearn.utils import to_time_series_dataset
from src.utils import FEATURES_MAPPING_BENDING, FEATURES_MAPPING_COMPRESSION

def parse_data(file_type: Literal['Bending', 'Compression'], features_mapping, sep: Literal['\t', ';'], label_col: str):
    data_path=RAW_DATA_DIR
    dataframes = []
    files_to_remove=[]
    for f in tqdm(Path(data_path).glob(f"*{file_type}*.txt"), desc=f"Parsing {file_type} data"):
        with open(f, 'r', encoding='latin-1') as file:
            material=file.name.split(f'{file_type}_')[1].split('_')[0]
            material_thickness=file.name.split(f'{file_type}_')[1].split('_')[1].strip('mm')
            content = file.readlines()
            start_idx = None
            static_features = {}
            for i, line in enumerate(content):
                if line.strip().startswith('\"Zeit\"'):
                    start_idx = i
                    break
                else:
                    for key in features_mapping.keys():
                        if key == line.split(sep)[0].strip().replace('"', ''):
                            key_name = features_mapping[key]
                            val_raw = line.split(sep)[1].strip().replace('"', '')

                            if key_name == 'Sample name':
                                static_features[f'static:{key_name}'] = val_raw
                            else:
                                try:
                                    # Prova a convertire in float (gestisce anche la virgola tedesca se necessario)
                                    static_features[f'static:{key_name}'] = float(val_raw)
                                except ValueError:
                                    static_features[f'static:{key_name}'] = None
                
            df=pd.read_csv(f, sep=sep, encoding='latin-1', skiprows=start_idx)
            for col in df.columns:
                df[col]=pd.to_numeric(df[col], errors='coerce')
           
            df=df.rename(columns=features_mapping)
            df=df.drop(0)
            for key, value in static_features.items():
                df[key] = value
            df['static:Material Thickness (mm)'] = int(material_thickness)
            df['static:Material'] = material
            dataframes.append(df)
    dataframes = [df for df in dataframes if df[label_col].notnull().any()]
    return dataframes
            
def get_tsdata(dataframes, type: Literal['Bending', 'Compression']):
    tsdata = []
    for df in dataframes:
        features = [c for c in df.columns if not str(c).startswith('static:') and 'Time' not in str(c)]
        tsdata.append(df[features].values)
    time_series_dataset=to_time_series_dataset(tsdata)
    print(f"Time series dataset shape for {type}: {time_series_dataset.shape}")
    return time_series_dataset    

if __name__ == "__main__":
    dataframes_bending=parse_data('Bending', sep='\t', features_mapping=FEATURES_MAPPING_BENDING, label_col='static:Bending strength')
    dataframes_compression=parse_data('Compression', sep=';', features_mapping=FEATURES_MAPPING_COMPRESSION, label_col='static:Compressive stress at maximum strain')
    
    null_cols=[]
    
    for df in dataframes_bending:
        for c in df.columns:
            if df[c].isnull().any(): 
                null_cols.append(c)
                
    from collections import Counter
    print(Counter(null_cols))
    
    for df in dataframes_compression:
        for c in df.columns:
            if df[c].isnull().any(): 
                null_cols.append(c)
    print(Counter(null_cols))