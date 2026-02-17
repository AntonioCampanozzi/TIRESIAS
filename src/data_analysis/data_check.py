from typing import Literal
from tqdm import tqdm
from src.data_analysis.data_parsing import parse_data
from src.utils import RAW_DATA_DIR, INTERIM_DATA_DIR, EXPLORATIVE_ANALYSIS_DIR, FEATURES_MAPPING_BENDING, FEATURES_MAPPING_COMPRESSION
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np

def check_missing_values(datasets, type: Literal['Bending', 'Compression']):
    null_summary=[]
    avg_null_percentages = {
        col: [] for col in datasets[0].columns if 'static:' not in col
    }
    for df in datasets:
        df = df[[col for col in df.columns if 'static:' not in col]]
        null_percentages = df.isna().mean() * 100
        for col, pct in null_percentages.items():
            if pct > 0:
                null_summary.append(col)
                avg_null_percentages[col].append(pct)
    avg_null_percentages = {col: np.mean(pcts) for col, pcts in avg_null_percentages.items() if pcts}
    if not null_summary:
        print("No missing values found in any dataset.")
    else:
        #making plot of the null summary and the average null percentages
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(avg_null_percentages.keys()), y=list(avg_null_percentages.values()), palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title("Average Null Percentages by Feature")
        plt.ylabel("Average Null Percentage")
        plt.xlabel("Feature")
        plt.tight_layout()
        dir = os.path.join(EXPLORATIVE_ANALYSIS_DIR, 'null_values')
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, f"average_null_percentages_{type.lower()}.png"))
        plt.close()
        plt.figure(figsize=(12, 8))
        sns.countplot(x=null_summary, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title("Count of Null Values by Feature")
        plt.ylabel("Number of datasets with Null Values")
        plt.xlabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"count_null_values_{type.lower()}.png"))
        plt.close()
        print(Counter(null_summary))
        print("Average Null Percentages:")
        for col, pct in avg_null_percentages.items():
            print(f"{col}: {pct:.2f}%")

def get_correlation_stats(datasets):
    """
    Calcola la matrice di correlazione Media, Minima e Massima su una lista di DataFrame.
    """
    all_corr_matrices = []

    for df in datasets:
        df = df[[col for col in datasets[0].columns if 'static:' not in col]]
        df = df.apply(pd.to_numeric, errors='coerce')
        corr = df.corr().abs()
        if corr.isnull().values.any():
            print(f"Attenzione: il dataset {df} contiene valori nulli nella correlazione!")
        all_corr_matrices.append(corr.to_numpy())
        

    corr_stack = np.array(all_corr_matrices)
    # Calcoliamo le statistiche lungo l'asse dei dataset (axis=0)
    mean_corr = pd.DataFrame(np.mean(all_corr_matrices, axis=0), index=df.columns, columns=df.columns)
    max_corr = pd.DataFrame(np.max(all_corr_matrices, axis=0), index=df.columns, columns=df.columns)
    min_corr = pd.DataFrame(np.min(all_corr_matrices, axis=0), index=df.columns, columns=df.columns)
    
    return mean_corr, max_corr, min_corr

def plot_stat_heatmap(matrix, type: Literal['Bending', 'Compression'], aggregate: Literal['Avg', 'Max', 'Min']):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"{aggregate} Feature Correlation")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    dir = os.path.join(EXPLORATIVE_ANALYSIS_DIR, f'feature_correlations_{type.lower()}')
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, f"{aggregate}_feature_correlation.png")) # Salva il file nella cartella corrente
    plt.close()

def distribution_by_material_and_material_thickness(datasets, type: Literal['Bending', 'Compression']):
    plt.figure(figsize=(12, 8))
    couples = []
    for df in tqdm(datasets, desc="processing experiments"):
        material = df['static:Material'].iloc[0]
        thickness = df['static:Material Thickness (mm)'].iloc[0]
        couples.append((material, thickness))
    c=Counter(couples)
    x=[f"{k[0]} ({k[1]}mm)" for k in c.keys()]
    y=[c[key] for key in c.keys()]
    sns.barplot(x=x, y=y, hue=x, palette='viridis', legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title("Experiment Distribution by Material and Thickness")
    dir = os.path.join(EXPLORATIVE_ANALYSIS_DIR, f'bar_plot_{type.lower()}')
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, f"distribution_material_thickness.png")) # Salva il file nella cartella corrente
    plt.close()

def distribution_by_material_thickness(datasets, type: Literal['Bending', 'Compression']):
    plt.figure(figsize=(12, 8))
    data = []
    for df in tqdm(datasets, desc="processing experiments"):
        thickness = df['static:Material Thickness (mm)'].iloc[0]
        data.append(thickness)
    c=Counter(data)
    x=[f"{k}mm" for k in c.keys()]
    y=[c[key] for key in c.keys()]
    sns.barplot(x=x, y=y, hue=x, palette='viridis', legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Experiment Distribution by Material Thickness")
    dir = os.path.join(EXPLORATIVE_ANALYSIS_DIR, f'bar_plot_{type.lower()}')
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, f"distribution_material_thickness.png")) # Salva il file nella cartella corrente
    plt.close()

def distribution_by_material(datasets, type: Literal['Bending', 'Compression']):
    plt.figure(figsize=(12, 8))
    data = []
    for df in tqdm(datasets, desc="processing experiments"):
        material = df['static:Material'].iloc[0]
        data.append(material)
    c=Counter(data)
    x=[f"{k}" for k in c.keys()]
    y=[c[key] for key in c.keys()]
    sns.barplot(x=x, y=y, hue=x, palette='viridis', legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Experiment Distribution by Material")
    dir = os.path.join(EXPLORATIVE_ANALYSIS_DIR, f'bar_plot_{type.lower()}')
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, f"distribution_material.png")) # Salva il file nella cartella corrente
    plt.close()

def force_deformation_curve(datasets, in_material, in_thickness, condition, type: Literal['Bending', 'Compression']):
    plt.figure(figsize=(12, 8))
    
    
    CONDITIONS = {
        0: [lambda d: d['static:Material Thickness (mm)'].iloc[0] == in_thickness,
            lambda d: f"{d['static:Material'].iloc[0]}",
            f"0_Force-Deformation Curve for {in_thickness}mm Thickness"],
        1: [lambda d: d['static:Material'].iloc[0] == in_material,
            lambda d: f"{d['static:Material Thickness (mm)'].iloc[0]}mm",
            f"1_Force-Deformation Curve for {in_material} Material"],
        2: [lambda d: (d['static:Material Thickness (mm)'].iloc[0] == in_thickness) and (d['static:Material'].iloc[0] == in_material),
            lambda d: f"{d['static:Material'].iloc[0]}-({d['static:Material Thickness (mm)'].iloc[0]}mm)",
            f"2_Force-Deformation Curve for {in_material} {in_thickness}mm"]
    }
    filtered_dfs = [df for df in datasets if CONDITIONS[condition][0](df)]
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    label_to_color = {}
    color_index = 0
    added_labels = set()
    
    for df in tqdm(filtered_dfs, desc="plotting force-deformation curves"):
        
        label_text = CONDITIONS[condition][1](df)
        if label_text not in label_to_color:
            label_to_color[label_text] = colors[color_index % len(colors)]
            color_index += 1
        
        current_color = label_to_color[label_text]
        plot_label = label_text if label_text not in added_labels else None
        added_labels.add(label_text)
        
        plt.plot(df['Deformation(mm)'],
                 df['Force applied (N)'],
                 label=plot_label, 
                 color=current_color,)
        
    plt.xlabel("Deformation (mm)")
    plt.ylabel("Force applied (N)")
    plt.title(CONDITIONS[condition][2])
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    dir = os.path.join(EXPLORATIVE_ANALYSIS_DIR, f'force_deformation_curves_{type.lower()}')
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, f"{CONDITIONS[condition][2]}.png")) # Salva il file nella cartella corrente
    plt.close()

if __name__ == "__main__":
    dataframes_bending=parse_data('Bending', FEATURES_MAPPING_BENDING, sep='\t', label_col='static:Bending strength')

    
    check_missing_values(dataframes_bending, type='Bending')
    
    mean_df, max_df, min_df=get_correlation_stats(dataframes_bending)
    plot_stat_heatmap(mean_df, type='Bending', aggregate='Avg')
    plot_stat_heatmap(max_df, type='Bending', aggregate='Max')
    plot_stat_heatmap(min_df, type='Bending', aggregate='Min')
    
    distribution_by_material_and_material_thickness(dataframes_bending, type='Bending')
    distribution_by_material_thickness(dataframes_bending, type='Bending')
    distribution_by_material(dataframes_bending, type='Bending')
    
    force_deformation_curve(dataframes_bending, in_material='PA', in_thickness=1, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PA', in_thickness=2, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PA', in_thickness=3, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PC', in_thickness=1, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PC', in_thickness=2, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PC', in_thickness=3, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PLA', in_thickness=1, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PLA', in_thickness=2, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PLA', in_thickness=3, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PETG', in_thickness=1, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PETG', in_thickness=2, condition=2, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PETG', in_thickness=3, condition=2, type='Bending')
    
    force_deformation_curve(dataframes_bending, in_material=None, in_thickness=1, condition=0, type='Bending')
    force_deformation_curve(dataframes_bending, in_material=None, in_thickness=2, condition=0, type='Bending')
    force_deformation_curve(dataframes_bending, in_material=None, in_thickness=3, condition=0, type='Bending')
    
    force_deformation_curve(dataframes_bending, in_material='PA', in_thickness=None, condition=1, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PC', in_thickness=None, condition=1, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PETG', in_thickness=None, condition=1, type='Bending')
    force_deformation_curve(dataframes_bending, in_material='PLA', in_thickness=None, condition=1, type='Bending')
    
    dataframes_compression=parse_data('Compression', FEATURES_MAPPING_COMPRESSION, sep=';', label_col='static:Compressive stress at maximum strain')
    
    check_missing_values(dataframes_compression, type='Compression')
    
    mean_df, max_df, min_df=get_correlation_stats(dataframes_compression)
    plot_stat_heatmap(mean_df, type='Compression', aggregate='Avg')
    plot_stat_heatmap(max_df, type='Compression', aggregate='Max')
    plot_stat_heatmap(min_df, type='Compression', aggregate='Min')
    
    distribution_by_material_and_material_thickness(dataframes_compression, type='Compression')
    distribution_by_material_thickness(dataframes_compression, type='Compression')
    distribution_by_material(dataframes_compression, type='Compression')
    
    force_deformation_curve(dataframes_compression, in_material='PA', in_thickness=1, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PA', in_thickness=2, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PA', in_thickness=3, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PC', in_thickness=1, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PC', in_thickness=2, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PC', in_thickness=3, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PLA', in_thickness=1, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PLA', in_thickness=2, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PLA', in_thickness=3, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PETG', in_thickness=1, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PETG', in_thickness=2, condition=2, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PETG', in_thickness=3, condition=2, type='Compression')
    
    force_deformation_curve(dataframes_compression, in_material=None, in_thickness=1, condition=0, type='Compression')
    force_deformation_curve(dataframes_compression, in_material=None, in_thickness=2, condition=0, type='Compression')
    force_deformation_curve(dataframes_compression, in_material=None, in_thickness=3, condition=0, type='Compression')
    
    force_deformation_curve(dataframes_compression, in_material='PA', in_thickness=None, condition=1, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PC', in_thickness=None, condition=1, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PETG', in_thickness=None, condition=1, type='Compression')
    force_deformation_curve(dataframes_compression, in_material='PLA', in_thickness=None, condition=1, type='Compression')

    
    