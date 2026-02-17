from random import randrange
from sklearn.model_selection import train_test_split
import torch
from src.data_analysis.data_parsing import parse_data
from src.utils import FEATURES_MAPPING_BENDING
import torch.nn as nn

class LstmDataset(torch.utils.data.Dataset):
    def __init__(self, df_list, stride, label, mode='train'):
        # Carichiamo i tensori originali (una sola volta)
        cols= [col for col in df_list[0].columns if col not in ['static:Sample name', label]]
        self.series = [torch.tensor(df[cols].values, dtype=torch.float32) for df in df_list]
        self.labels = [torch.tensor(df[label].iloc[0], dtype=torch.float32) for df in df_list]
        self.max_len = max(len(s) for s in self.series)
        self.mode=mode
        # Definiamo i punti di osservazione temporale
        self.stride = stride
        self.steps = list(range(stride, self.max_len + stride, stride))
        
    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx, test=False):
        batch_list = []
        label_list = []
        lengths = []
    
        fixed_length = (self.max_len + self.stride) // 10
    
        for s, l in zip(self.series, self.labels):
            if self.mode == 'test':
                t = self.steps[idx]
            else:
        # 1. Tira il dado
                t = randrange(self.stride, self.max_len + self.stride, self.stride)
        
        # 2. Taglia la serie (se s è più corta di t, prenderà quello che può)
            prefix = s[:t:10]
        
        # 3. GUARDA QUANTO È LUNGO IL PEZZO CHE HAI IN MANO
            actual_len = prefix.shape[0]
            
        
        # 4. IL PADDING SI FA SULLA LUNGHEZZA REALE PER ARRIVARE A FIXED_LENGTH
            pad_size = fixed_length - actual_len
        
            if pad_size > 0:
                padding = torch.zeros((pad_size, s.shape[1]))
            # Somma: actual_len + (fixed_length - actual_len) = fixed_length. SEMPRE.
                full_prefix = torch.cat([prefix, padding], dim=0)
            else:
                full_prefix = prefix[:fixed_length]
            
            batch_list.append(full_prefix)
            label_list.append(l)
            lengths.append(actual_len)
        
    # ORA sono tutti lunghi uguali e lo stack NON può fallire.
        return torch.stack(batch_list), torch.stack(label_list), torch.tensor(lengths)

class EarlyPredLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EarlyPredLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, lengths):
        # out: (batch, seq_len, hidden_size)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, (hn, cn) = self.lstm(x_packed)
        prediction = self.fc(hn[-1])
        return prediction

if __name__ == "__main__":
    dataframes_bending=parse_data('Bending', sep='\t', features_mapping=FEATURES_MAPPING_BENDING)
    data_lst, val_lst = train_test_split(dataframes_bending, test_size=0.2, random_state=42)
    train_lst, test_lst = train_test_split(data_lst, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
    print(f"Train set size: {len(train_lst)}, Validation set size: {len(val_lst)}")
    print(dataframes_bending[0].dtypes)
    dataset = LstmDataset(dataframes_bending, stride=50, label='static:Bending strength')
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[1]
    print(sample[0].shape)