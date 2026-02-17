from tqdm import tqdm
import pandas as pd

class prefixDataset():
    def __init__(self, dataframes, label_col, stride):
        self.dataframes = dataframes
        self.series = [df for df in dataframes]
        self.label_col = label_col
        self.stride = stride
        self.max_len = max(len(s) for s in self.series)
        self.steps = list(range(stride, self.max_len + stride, stride))
        prefix_dataset = []
        for s in tqdm(self.series, desc="Processing time series"):
            for step in self.steps:
                prefix_dct=self.__extract_prefixes_and_collapse(s, step)
                prefix_dataset.append(prefix_dct)
        self.dataset = pd.DataFrame(prefix_dataset)
    
    
    def __extract_prefixes_and_collapse(self, time_serie, step):
        t=step
        prefix_dct = {}
        prefix=time_serie[:t]
        columns=[col for col in prefix.columns if not 'static:' in col]
        prefix_dct['ID'] = prefix['static:Sample name'].iloc[0]
        for col in columns:
            prefix_dct[f'{col}_mean'] = prefix[col].mean()
            prefix_dct[f'{col}_std'] = prefix[col].std()
            prefix_dct[f'{col}_skew'] = prefix[col].skew()
            prefix_dct[f'{col}_kurt'] = prefix[col].kurt()
            prefix_dct[f'{col}_last'] = prefix[col].iloc[-1]
            prefix_dct[f'{col}_diff'] = prefix[col].iloc[-1] - prefix[col].iloc[0]
        prefix_dct['label'] = time_serie[self.label_col].iloc[0]
        return prefix_dct