
from transformers import BartTokenizer, \
    BartForSequenceClassification, Trainer,\
    TrainingArguments, DataCollatorWithPadding
    

import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

###########################################################################

class BartZeroShot:
    """Adapted from the provided  - thin wrapper around facebook/bart-large-mnli
    """
    def __init__(self):
        self.nli_model = BartForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli"
        ).to(DEVICE)
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli")

    def predict(self, sentence, label):
        x = self.tokenizer.encode(
            sentence,
            f"This example is {label}",
            return_tensors="pt",
            truncation="only_first",
        )
        logits = self.nli_model(x.to(DEVICE))[0]
        
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(1)
        prob_label_is_true = probs[:, 1].item()
        return prob_label_is_true
    
    def predict_batched(self, sentences, label):
        x = self.tokenizer.batch_encode_plus(
            map(lambda x : x + f' This example is {label}', sentences), 
            return_tensors = 'pt',
            padding='longest'
        )
        
        logits = self.nli_model(x.to(DEVICE)['input_ids'])[0]
        l1 = logits[:,[0,2]]
        l2 = l1.softmax(1)
        l3 = l2[:,1]
        return l3.detach().to('cpu')