
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_abc = 'abcdefghijklmnopqrstuvwxyz'
_non_special_chars = set('&$@#,;/:?!.' + _abc + _abc.upper())
def _ratio_chars_not_in_set(x, 
                            char_set = _non_special_chars, 
                            thresh = 0.85):
    not_in_set = 0
    for xx in x:
        if xx not in char_set:
            not_in_set += 1
    return not_in_set / len(x) > thresh



class TwitterDataClass:
    def __init__(self, dataset_folder = './data', csv_file_name = 'twitter_dataset_full.csv', load_cached_stage = None):
        self.dataset_folder = dataset_folder
        self.file_path = os.path.join(self.dataset_folder, csv_file_name)
        self.cached_stages = {
            1 : os.path.join(self.dataset_folder, 'twitter_dataset_sorted_token_length.pcl')
        }
        if load_cached_stage is not None and load_cached_stage in self.cached_stages:
            _cached_file = self.cached_stages[load_cached_stage]
            self.d = pd.read_pickle(_cached_file)
        else:
            self.d = pd.read_csv(self.file_path)
        
    def _preproc_st1(self, tokenizer = None, cache = True):
        self.d['is_noise'] = self.d['message'].apply(_ratio_chars_not_in_set)            
        def _get_batch(_dat, batch_size = 512):
            for bstart in range(0, _dat.shape[0], batch_size):
                yield _dat[bstart:bstart+batch_size]

        _b_num_tokens = []
        for b in tqdm(_get_batch(self.d['message'])):
            _b_num_tokens.extend([len(l) for l in tokenizer(list(b.values))['input_ids']])
        self.d['token_length'] = _b_num_tokens
        if cache:
            self.d.to_pickle(self.cached_stages[1])
    
    def train_test_split(self, train_ratio = .9, remove_noise = True):
        # stratify according to class and number of tokens:
        num_token_length_buckets = 10
        _d = self.d.copy()
        if remove_noise:
            _d = _d[~_d['is_noise']]
            _d = _d.drop_duplicates(subset = 'id', keep = 'first')
        _q = _d['token_length'].quantile(np.linspace(0, 1, num_token_length_buckets)[1:])
        _d['token_length_clusters'] = _d['token_length'].apply(lambda x : sum(x > _q.values))
        _strat = [(int(v1),int(v2)) for v1, v2 in _d[['token_length_clusters','is_positive']].values]
        train_inds, test_inds = train_test_split(_d.index, stratify = _strat)
        return _d.loc[train_inds], _d.loc[test_inds]

class TorchTwitterDataset(TorchDataset):
    def __init__(self, df_data : pd.DataFrame, tokenizer= None, max_token_length = 65):
        if any([m not in df_data.columns for m in ['message', 'is_positive']]):
            raise Exception("In order to create the dataset you need to provide a dataframe with `message` and `is_positive` columns!")
        if tokenizer is None:
            raise Exception("Please provide a tokenizer (with approp. interface)")
        self.df_data = df_data
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        
    def __len__(self):
        return self.df_data.shape[0]
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        x = list(self.df_data.iloc[idx]['message'].values)
        y = list(self.df_data.iloc[idx]['is_positive'].values)
        toks = self.tokenizer(list(x), max_length =self.max_token_length, padding = 'max_length',truncation=True, return_tensors = 'pt').to(DEVICE)
        
        # To remove the batch dimension added from the return_tensors = 'pt'
        toks = {key: val.squeeze() for key, val in toks.items()}
        toks['label'] = torch.Tensor([int(i) for i in y] ).to(dtype = torch.int32)
        return toks
