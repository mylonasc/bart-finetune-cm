## Train script for BART base

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import hashlib
import argparse

parser = argparse.ArgumentParser(description="Training script with command-line arguments")

# Define command-line arguments
parser.add_argument('--num_train_epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--skip_how_many', type=int, default=100, help="Number of samples to skip (if it is 1, then no sample is skipped)")
parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
parser.add_argument('--lrate', type=float, default=1e-6, help="Learning rate")
parser.add_argument('--seed', type=int, default=1337, help="Random seed")

from transformers import BartTokenizer, \
    BartForSequenceClassification, Trainer,\
    TrainingArguments, DataCollatorWithPadding

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

from src.data_utils import TwitterDataClass, TorchTwitterDataset

class Runner:
    """A thin wrapper around hugging face functionality and data manipulation 
    code (e.g., cleaning) for the twitter sentiment classification dataset 
    """
    def __init__(self, config : dict):
        self.config = config
        
    def run(self):
        """ Trains a BART model on the dataset and stores the """
        # numpy randomization affects skearn.model_selection.train_test_split.
        np.random.seed(self.config['seed'])
        config = self.config
        tw = TwitterDataClass(load_cached_stage = 1)
        d_train, d_test = tw.train_test_split()
        if 'skip_how_many' in config:
            SKIP_HOW_MANY = config['skip_how_many']
            d_train_small, d_test_small = d_train[::SKIP_HOW_MANY], d_test[::SKIP_HOW_MANY]
            
        tokenizer = BartTokenizer.from_pretrained(self.config['model_name'])
        # It should have worked with num_labels == 1 since it should be computing a logit only on the first column.
        # Cuda throws an error when attempting to run this with num_labels == 1, but the results seem to be correct.
        model = BartForSequenceClassification.from_pretrained(
                    self.config['model_name'], 
                    num_labels = 2, 
                    problem_type = 'single_label_classification'
                ).to(DEVICE)
        
        _config_hash = self.fname_hash_from_config()
        training_args = TrainingArguments(
            output_dir=f"./results_{_config_hash}",
            evaluation_strategy="epoch",
            learning_rate=config['lrate'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            num_train_epochs=config['num_train_epochs'],
            weight_decay=0.01,
            logging_dir="./logs"
        )
        print("--- loading pre-trained models ---")
        
        model_name = config['model_name'] 
        tokenizer = BartTokenizer.from_pretrained(model_name)
        # It should have worked with num_labels == 1 since it should be computing a logit only on the first column.
        # Cuda throws an error when attempting to run this with num_labels == 1, but the results seem to be correct.
        model = BartForSequenceClassification.from_pretrained(
            model_name, 
            num_labels = 2, 
            problem_type = 'single_label_classification'
        ).to(DEVICE)
        t_train = TorchTwitterDataset(d_train_small.sort_values('token_length'), tokenizer)
        t_test  = TorchTwitterDataset(d_test_small.sort_values('token_length'), tokenizer)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # Define the Trainer for sequence classification
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=t_train,
            eval_dataset=t_test, 
            data_collator = data_collator
        )
        print("--- starting training --- ")
        trainer.train()
        d_result = pd.DataFrame(trainer.state.log_history)
        for k, v in self.config.items():
            d_result[k] = v
        d_result.to_pickle(_config_hash + '_log_hist.pcl')
        print("--- done --- ")
    def fname_hash_from_config(self):
        _fname = self.fname_from_config()
        return hashlib.md5(_fname.encode()).hexdigest()
    def fname_from_config(self):
        # To make the hash (more or less...) unique I sort the keys.
        k_sorted = sorted(self.config.keys())
        return '-'.join([k + '_' + str(self.config[k]).replace('/','-') for k in k_sorted])
    
if __name__ == '__main__':
    args = parser.parse_args()
    config_small_run = {
        'skip_how_many' : args.skip_how_many,
        'lrate' : args.lrate,
        'num_train_epochs' : args.num_train_epochs,
        'batch_size' : args.batch_size,
        'model_name' : "facebook/bart-base",
        'seed' : args.seed
    }
    Runner(config_small_run).run()
