import json
import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ResultsReader:
    """A class to facilitate reading the results from the training run
    """
    def __init__(self, results_folder = './results'):
        self.results_folder = results_folder
        self.checkpoint_steps_dict = {int(f.split('-')[1]) : os.path.join(self.results_folder, f) for f in  os.listdir(results_folder)}
    
    def get_log_hist(self):
        """Gets a dataframe that contains the log history saved together with the last checkpoint.
        Earlier checkpoints also contain the log history, but the complete is with the last one. 
        """
        k_max = max(self.keys())
        cp_max = self.checkpoint_steps_dict[k_max]
        with open(os.path.join(cp_max, 'trainer_state.json'), 'r') as f:
            j = json.loads(f.read())['log_history']
        return pd.DataFrame(j)
    
    def get_best_checkpoint_id(self):
        """Returns the next checkpoint after the best recorded validation score.
        """
        log_hist = self.get_log_hist()
        best_val_step = int(log_hist.sort_values('eval_loss')['step'].iloc[0])
        checkpoint_closest = sorted(self.keys(), key = lambda x : abs(x - best_val_step))
        return max(checkpoint_closest[:2])
    
    def __len__(self):
            return len(self.checkpoint_steps_dict)
        
    def keys(self):
        return self.checkpoint_steps_dict.keys()
    
    def __getitem__(self, k):
        if k not in self.checkpoint_steps_dict:
            raise Exception("Unknown checkpoint key!")
        return self.checkpoint_steps_dict[k]
    
    def load_model(self, checkpoint_id): 
        """Given a valid checkpoint ID, it loads a model from the checkpoints.
        """
        fpath = self.checkpoint_steps_dict[checkpoint_id]
        # Passing the path directly for the AutoModelTokenizer does not work for some reason. 
        # I'm passing the model name, in case I include other models in the future besides BART 
        # for classification. 
        with open(os.path.join(fpath, 'config.json'), 'r') as f:
            j =  json.loads(f.read())
        model_name = j['_name_or_path']
        print(f"- loading model from {fpath}, model_type: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(fpath)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    def load_best_model(self):
        checkpoint_id = self.get_best_checkpoint_id()
        return self.load_model(checkpoint_id)