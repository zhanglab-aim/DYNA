#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import re
import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List
import torch
import torch.nn as nn
import transformers
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib as mpl
from torch.utils.data import Dataset
from scipy.special import softmax
from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    AdaLoraModel
)

import torch.nn.functional as F
import matplotlib.pyplot as plt
import gpn.model


# In[3]:


class OneHotTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocabulary = [line.strip() for line in f.readlines()]

        # Here, the tokens are individual characters (like A, C, G, T)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.token_length = 1  # Each nucleotide is considered separately in one-hot encoding

    def tokenize(self, sequence):
        # In one-hot encoding, each character/nucleotide is a token
        tokens = [sequence[i:i+self.token_length] for i in range(len(sequence))]
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.token_to_id.get('[UNK]')) for token in tokens]

    def __call__(self, sequence):
        tokens = self.tokenize(sequence)
        return self.convert_tokens_to_ids(tokens)


# In[4]:


class OverlappingTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocabulary = [line.strip() for line in f.readlines()]
        
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.token_length = 6  # Overlapping length

    def tokenize(self, sequence):
        tokens = [sequence[i:i+self.token_length] for i in range(0, len(sequence) - self.token_length + 1)]
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.token_to_id.get('[UNK]')) for token in tokens]

    def __call__(self, sequence):
        tokens = self.tokenize(sequence)
        return self.convert_tokens_to_ids(tokens)


# In[5]:


def create_alphabet_dict(file_path):
    alphabet = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            token = line.strip()  # Remove any trailing newline characters
            alphabet[token] = index
    return alphabet

# Usage
file_path = '/common/zhangz2lab/zhanh/dnabert-config/bert-config-6/vocab_nt.txt'  # Replace with your file path
alphabet_dict = create_alphabet_dict(file_path)
print(alphabet_dict)
alphabet_dict = {'[PAD]': 0, 'a': 3, 'g': 5, 'c': 4, '[MASK]': 1, '[UNK]': 2, 't': 6}
print(alphabet_dict)
alphabet_dict = {'[PAD]': 0, 'A': 3, 'G': 5, 'C': 4, '[MASK]': 1, '[UNK]': 2, 'T': 6}


# In[6]:


from transformers import TrainerCallback, TrainerControl
loss_name ="contrastive+BCE"

class TestSetCallback(TrainerCallback):
    def __init__(self, model, trainer, test_dataset, eval_steps, tokenizer):
        self.model = model
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.eval_steps = eval_steps
        self.tokenizer = tokenizer
        self.step_count = 0
        self.alphabet = alphabet_dict
        
    def plot_roc(self, labels, logits):
        fpr, tpr, _ = roc_curve(labels, logits)
        roc_curve_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_curve_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()
        
                
    def compute_pll_for_sequence(self, sequence, model):
        #tokens = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        tokens = self.tokenizer(sequence, return_tensors="pt", truncation=True, padding="max_length", max_length=training_args.model_max_length)
        model_device = next(model.parameters()).device
        for key in tokens.keys():
            tokens[key] = tokens[key].to(model_device)
            
        with torch.no_grad():
            outputs = model.base_model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        
        logits = torch.log_softmax(outputs.logits, dim=-1)
        #print('logits',logits)
        idx = [self.alphabet[t] for t in sequence]
        PLL = torch.sum(torch.diag(logits[0, 1:-1, :][:, idx]))
        return PLL.item(), logits
        

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        lambda_training_steps = self.model.lambda_training_steps
        self.model.lambda_weight = min(1, self.step_count / lambda_training_steps)
        if state.global_step % 10000 == 0:  # Adjust interval as needed
            #self.model.eval() 
            print(f"Step {state.global_step}, Lambda: {self.model.lambda_weight}")
            test_results = self.trainer.evaluate(self.test_dataset)
            #self.model.train()
            print(test_results)
        #if self.step_count == 2 or state.global_step % self.eval_steps == 0:
        if (self.step_count == 2 or (self.step_count > 2 and state.global_step % self.eval_steps == 0)) and loss_name =="PLLR":
            # Perform evaluation and plot ROC
            predictions = self.trainer.predict(self.test_dataset)
            logits = predictions.predictions
            labels = predictions.label_ids


            self.plot_roc(labels, logits)
            
            all_sequences = []
            df = pd.read_csv('/common/zhangz2lab/zhanh/Jupyter_Scripts/cm_spair/test_200.csv')
            all_sequences = df['seq_a'].tolist()

            all_plls_wt = []
            all_plls_wt_weighted = []
            
            for seq in all_sequences:
                wt_pll, wt_logits = self.compute_pll_for_sequence(seq, model)
                print(wt_pll)
                all_plls_wt.append(wt_pll)
                all_plls_wt_weighted.append(wt_pll / len(seq))
                
            
            all_sequences = []
            all_sequences = df['seq_b'].tolist()

            all_plls_mut = []
            all_plls_mut_weighted = []
            for seq in all_sequences:
                mut_pll, _ = self.compute_pll_for_sequence(seq, model)
                print(mut_pll)
                all_plls_mut.append(mut_pll)
                all_plls_mut_weighted.append(mut_pll / len(seq))

            all_plls_wt = np.array(all_plls_wt)
            all_plls_mut = np.array(all_plls_mut)
            
            all_plls_wt_weighted = np.array(all_plls_wt_weighted)
            all_plls_mut_weighted = np.array(all_plls_mut_weighted)
        
        # Compute the PLLR
            PLLR_callback = np.abs(all_plls_wt - all_plls_mut)
            PLLR_weighted_callback = np.abs(all_plls_wt_weighted - all_plls_mut_weighted)
        
        # Get true labels
            true_labels_callback = df['labels'].to_numpy()
            fpr, tpr, _ = roc_curve(true_labels_callback, PLLR_callback)
            roc_auc = auc(fpr, tpr)
            aupr = average_precision_score(true_labels_callback, PLLR_callback)

            # Compute metrics for PLLR_weighted_callback
            fpr_weighted, tpr_weighted, _ = roc_curve(true_labels_callback, PLLR_weighted_callback)
            roc_auc_weighted = auc(fpr_weighted, tpr_weighted)
            aupr_weighted = average_precision_score(true_labels_callback, PLLR_weighted_callback)

            # Plotting ROC for both PLLR_callback and PLLR_weighted_callback
            #plt.figure()
            plt.figure(figsize=(10, 7))
            mpl.rcParams['font.size'] = 18
            lw = 2  # line width
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='PLLR ROC curve (area = %0.2f)' % roc_auc)
            plt.plot(fpr_weighted, tpr_weighted, color='darkgreen', lw=lw, label='weighted PLLR ROC curve (area = %0.2f)' % roc_auc_weighted)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) for PLLR and weighted PLLR')
            plt.legend(loc="lower right")
            plt.show()


            
    

class CustomCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.step_count = 0
        self.alphabet = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

    def compute_pll_for_sequence(self, sequence, model):
        #tokens = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        tokens = self.tokenizer(sequence, return_tensors="pt", truncation=True, padding="max_length", max_length=training_args.model_max_length)
        model_device = next(model.parameters()).device
        for key in tokens.keys():
            tokens[key] = tokens[key].to(model_device)
            
        with torch.no_grad():
            outputs = model.base_model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        
        logits = torch.log_softmax(outputs.logits, dim=-1)
        #print('logits',logits)
        idx = [self.alphabet[t] for t in sequence]
        PLL = torch.sum(torch.diag(logits[0, 1:-1, :][:, idx]))
        return PLL.item(), logits
    

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.step_count += 1
        if self.step_count == 1 or self.step_count % 50 == 0:  # You can adjust the frequency as needed.
            all_sequences = []
            df = pd.read_csv("/common/zhangz2lab/zhanh/esm-variants/cropped/cm_test_data_1024.csv")
            all_sequences = df['wt_seq'].tolist()
            df_arm = pd.read_csv("/common/zhangz2lab/zhanh/esm-variants/cropped/arm_test_data_1024.csv")
            sequence_arm = df_arm['wt_seq'].iloc[0]

            all_plls_wt = []
            all_plls_wt_weighted = []
            
            for seq in all_sequences:
                wt_pll, wt_logits = self.compute_pll_for_sequence(seq, model)
                all_plls_wt.append(wt_pll)
                all_plls_wt_weighted.append(wt_pll / len(seq))


# In[7]:


# Dataset Definition
# class SiameseDataset(Dataset):
#     def __init__(self, tokenizer, filename):
#         data = pd.read_csv(filename)
#         self.tokenizer = tokenizer
#         self.seq_a = list(data['seq_a'])
#         self.seq_b = list(data['seq_b'])
#         self.labels = list(data['labels'])
#         self.num_examples = len(self.labels)
    
#     def __len__(self):
#         return self.num_examples
    
#     def __getitem__(self, idx):
#         inputs_a = self.tokenizer(self.seq_a[idx], return_tensors="pt", truncation=True, padding="longest", max_length=512)
#         inputs_b = self.tokenizer(self.seq_b[idx], return_tensors="pt", truncation=True, padding="longest", max_length=512)
#         return {
#             "input_ids1": inputs_a["input_ids"].squeeze(0), 
#             "attention_mask1": inputs_a["attention_mask"].squeeze(0),
#             "input_ids2": inputs_b["input_ids"].squeeze(0),
#             "attention_mask2": inputs_b["attention_mask"].squeeze(0),
#             "labels": torch.tensor(self.labels[idx], dtype=torch.long)
#         }
    
class SiameseDataset(Dataset):
    def __init__(self, tokenizer, tokenizer_type, filename):
        data = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.seq_a = list(data['seq_a'])
        self.seq_b = list(data['seq_b'])
        self.labels = list(data['labels'])
        self.num_examples = len(self.labels)
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        if self.tokenizer_type == "bpe":
            inputs_a = self.tokenizer(self.seq_a[idx], return_tensors="pt", truncation=True, padding="longest", max_length=512)
            inputs_b = self.tokenizer(self.seq_b[idx], return_tensors="pt", truncation=True, padding="longest", max_length=512)
            return {
                "input_ids1": inputs_a["input_ids"].squeeze(0), 
                "attention_mask1": inputs_a["attention_mask"].squeeze(0),
                "input_ids2": inputs_b["input_ids"].squeeze(0),
                "attention_mask2": inputs_b["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)
            }
        elif self.tokenizer_type == "6-mer" or self.tokenizer_type == "one-hot":  # Assume the other type is the custom overlapping tokenizer
            #print("Using 6-meer")
            input_ids_a = torch.tensor(self.tokenizer(self.seq_a[idx]))
            input_ids_b = torch.tensor(self.tokenizer(self.seq_b[idx]))
            
            attention_mask_a = torch.ones_like(input_ids_a)
            attention_mask_b = torch.ones_like(input_ids_b)
            max_length = 512
            if len(input_ids_a) < max_length:
                padding_size = max_length - len(input_ids_a)
                input_ids_a = F.pad(input_ids_a, pad=(0, padding_size), value=0)
                attention_mask_a = F.pad(attention_mask_a, pad=(0, padding_size), value=0)
            
            if len(input_ids_b) < max_length:
                padding_size = max_length - len(input_ids_b)
                input_ids_b = F.pad(input_ids_b, pad=(0, padding_size), value=0)
                attention_mask_b = F.pad(attention_mask_b, pad=(0, padding_size), value=0)

            return {
                "input_ids1": input_ids_a.squeeze(0),
                "attention_mask1": attention_mask_a.squeeze(0),
                "input_ids2": input_ids_b.squeeze(0),
                "attention_mask2": attention_mask_b.squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)
            }


# In[8]:


# class SiameseNetwork(nn.Module):
#     def __init__(self, model_name_or_path, num_labels, cache_dir=None):
#         super(SiameseNetwork, self).__init__()

#         # Load the base model
#         self.base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
#             model_name_or_path,
#             cache_dir=cache_dir,
#             num_labels=num_labels,
#             trust_remote_code=True,
#             output_hidden_states=True
#         )
        
        
#     def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels):
#         # Encoding sequences using the same base model
#         logits1 = self.base_model(input_ids=input_ids1, attention_mask=attention_mask1).logits
#         logits2 = self.base_model(input_ids=input_ids2, attention_mask=attention_mask2).logits
#         #pooler_output1 = self.base_model(input_ids=input_ids1, attention_mask=attention_mask1).pooler_output
#         #pooler_output2 = self.base_model(input_ids=input_ids2, attention_mask=attention_mask2).pooler_output
#         outputs1 = self.base_model(input_ids=input_ids1, attention_mask=attention_mask1)
#         outputs2 = self.base_model(input_ids=input_ids2, attention_mask=attention_mask2)
#         last_hidden_state1 = outputs1.hidden_states[-1][:, 0, :]
#         last_hidden_state2 = outputs2.hidden_states[-1][:, 0, :]
#         #output1 = logits1[:, 0]
#         #output2 = logits2[:, 0]
#         #cosine_sim = F.cosine_similarity(output1, output2, dim=-1)
#         cosine_sim = F.cosine_similarity(last_hidden_state1, last_hidden_state2, dim=-1)
#         mapped_sim = (cosine_sim + 1) / 2
#         loss = F.mse_loss(mapped_sim, labels.float())
        
#         return (loss, cosine_sim)

    

class SiameseNetwork(nn.Module):
    def __init__(self, model_name_or_path, num_labels, loss_type="contrastive+BCE", margin=2.0, lambda_weight=1.0, cache_dir=None):
        super(SiameseNetwork, self).__init__()

        self.loss_type = loss_type
        self.margin = margin
        self.lambda_weight = lambda_weight
        self.lambda_training_steps = 6000 

        # Load the base model
        self.base_model = transformers.AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            output_hidden_states=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
             nn.Linear(self.base_model.config.hidden_size * 2, 128),
             nn.ReLU(),
             nn.Linear(128, num_labels)
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels):
        # Encoding sequences using the same base model
        outputs1 = self.base_model(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.base_model(input_ids=input_ids2, attention_mask=attention_mask2)
        #print("Shape of logits1:", outputs1.logits.shape)

        #last_hidden_state1 = outputs1.hidden_states[-1][:, 0, :]
        #last_hidden_state2 = outputs2.hidden_states[-1][:, 0, :]
        last_hidden_state1 = outputs1.last_hidden_state[:, 0, :]
        last_hidden_state2 = outputs2.last_hidden_state[:, 0, :]
        #print("Shape of last hidden state 1:", last_hidden_state1.shape)
        if self.loss_type == "PLLR":
            logits1 = torch.log_softmax(outputs1.logits, dim=-1)
            logits2 = torch.log_softmax(outputs2.logits, dim=-1)
            #print('logits1',logits1)
            #print('logits2',logits2)
            batch_size = input_ids1.shape[0]
        
        
            PLLs1 = torch.zeros(batch_size, device=input_ids1.device)
            PLLs2 = torch.zeros(batch_size, device=input_ids2.device)

            for i in range(batch_size):
                idx1 = input_ids1[i, 1:-1]  # Excluding the special tokens <cls> and <eos>/<pad>
                PLLs1[i] = torch.sum(torch.diag(logits1[i, 1:-1, :][:, idx1]))
            for i in range(batch_size):
                idx2 = input_ids2[i, 1:-1]  # Excluding the special tokens <cls> and <eos>/<pad>
                PLLs2[i] = torch.sum(torch.diag(logits2[i, 1:-1, :][:, idx2]))
            PLLR = torch.abs(PLLs1 - PLLs2)
            print(PLLR)
            PLLR_t = PLLs1 - PLLs2
            print(PLLR_t)
            sigmoid_PLLR = torch.sigmoid(PLLR)
            pll_loss = F.binary_cross_entropy(2*sigmoid_PLLR-1, labels.float())

            #return (loss, cosine_sim)
            return (pll_loss, PLLR)

        elif self.loss_type == "cosine":
            cosine_sim = F.cosine_similarity(last_hidden_state1, last_hidden_state2, dim=-1)
            mapped_sim = (cosine_sim + 1) / 2  # map between 0 and 1
            # Invert mapped_sim when the label is 1 to encourage dissimilarity
            inverted_mapped_sim = 1 - mapped_sim
            # Use label to choose between mapped_sim and inverted_mapped_sim
            adjusted_sim = labels.float() * inverted_mapped_sim + (1 - labels.float()) * mapped_sim
            loss = F.mse_loss(adjusted_sim, labels.float())
            return (loss, cosine_sim)
        
        elif self.loss_type == "contrastive":
            euclidean_distance = F.pairwise_distance(last_hidden_state1, last_hidden_state2)
            # Contrastive loss
            loss = (1 - labels) * torch.pow(euclidean_distance, 2) +                    labels * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
            loss = torch.mean(loss)
            return (loss, euclidean_distance)
        
        elif self.loss_type == "contrastive+BCE":
            euclidean_distance = F.pairwise_distance(last_hidden_state1, last_hidden_state2)
            # Contrastive loss
            contrastive_loss = (1 - labels) * torch.pow(euclidean_distance, 2) +                    labels * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
            contrastive_loss = torch.mean(contrastive_loss)
            # Binary Cross-Entropy (BCE) Loss
            outputs = self.classifier(torch.cat([last_hidden_state1, last_hidden_state2], dim=1))
            bce_loss = F.cross_entropy(outputs, labels)
            #self.lambda_weight = min(1, self.forward_count /self.lambda_training_steps)
            # Combined loss
            total_loss = contrastive_loss + self.lambda_weight * bce_loss
            return (total_loss, outputs, euclidean_distance)

        else:
            raise ValueError("Invalid loss_type specified.")


# In[9]:


# training_args = TrainingArguments(
#      #optim = "adamw_torch",
#      run_name = "run",
#      gradient_accumulation_steps =1,
#      per_device_train_batch_size=8,
#      per_device_eval_batch_size=8,
#      fp16=False,
#      num_train_epochs=30,
#      evaluation_strategy="steps",
#      eval_steps = 100,
#      save_steps=100,
#      logging_dir='/common/zhangz2lab/zhanh/Jupyter_Scripts/output_0824/logs',
#      logging_steps=100,
#      load_best_model_at_end=True,  # this is defined twice in your class, consider removing one
#      metric_for_best_model="eval_loss",
#      greater_is_better=False,
#      #early_stopping_patience=3,  # Number of evaluations without improvement to wait before stopping training
#      #early_stopping_threshold=0.001,
#      logging_strategy="steps",
#      #warmup_ratio=0.1,
#      weight_decay=1e-4,
#      learning_rate=2e-5,
#      #lr_scheduler_type='linear',
#      do_train=True,
#      do_eval=True,
#      output_dir='/common/zhangz2lab/zhanh/Jupyter_Scripts/output_0825/results',
#      save_strategy='steps',
#      save_total_limit=5,
#      push_to_hub=False,
#      dataloader_pin_memory=False,
#      seed=42,
#      logging_first_step=True
#  )

# training_args = TrainingArguments(
#      per_device_train_batch_size=8,
#      per_device_eval_batch_size=8,
#      num_train_epochs=10,
#      evaluation_strategy="steps",
#      logging_dir='/common/zhangz2lab/zhanh/Jupyter_Scripts/output_0824/logs',
#      logging_steps=100,
#      do_train=True,
#      do_eval=True,
#      output_dir='/common/zhangz2lab/zhanh/Jupyter_Scripts/output_0824/results',
#      save_strategy='steps',
#      save_total_limit=2,
#      push_to_hub=False,
#  )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    num_train_epochs: int = field(default=5)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=500)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    evaluation_strategy: str = field(default="steps")
    load_best_model_at_end: bool = field(default=True)     # load the best model when finished training (default metric is loss)
    metric_for_best_model: str = field(default="eval_loss") # the metric to use to compare models
    greater_is_better: bool = field(default=False)           # whether the `metric_for_best_model` should be maximized or not
    logging_strategy: str = field(default="steps")  # Log every "steps"
    logging_steps: int = field(default=500)  # Log every 100 steps
    warmup_ratio: int = field(default=0.1)
    weight_decay: float = field(default=1e-4)
    learning_rate: float = field(default=2e-5)
    lr_scheduler_type: str = field(default='linear')
    save_total_limit: int = field(default=5)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="/common/zhangz2lab/zhanh/Jupyter_Scripts/output_0227/gpn_results")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    logging_first_step: bool = field(default=True)
    early_stopping_patience: int = field(default = 5)  # number of evaluations without improvement to wait
    early_stopping_threshold: float = field(default = 1e-3)  # threshold for an improvement
        
training_args = TrainingArguments()


# In[10]:


@dataclass
class ModelArguments:
    #model_name_or_path: Optional[str] = field(default="bert-base-uncased")
    #model_name_or_path: Optional[str] = field(default="InstaDeepAI/nucleotide-transformer-500m-human-ref")
    #model_name_or_path: Optional[str] = field(default="InstaDeepAI/nucleotide-transformer-2.5b-1000g")
    model_name_or_path: Optional[str] = field(default="songlab/gpn-brassicales")
    #model_name_or_path: Optional[str] = field(default="/common/zhangz2lab/zhanh/SpliceBERT/examples/models/SpliceBERT.1024nt")
    #model_name_or_path: Optional[str] = field(default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    #model_name_or_path: Optional[str] = field(default="facebook/esm1b_t33_650M_UR50S")
    #model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    #model_name_or_path: Optional[str] = field(default="decapoda-research/llama-7b-hf")
    #model_name_or_path: Optional[str] = field(default="microsoft/MiniLM-L12-H384-uncased")
    #model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    #lora_target_modules: str = field(default="k_proj,q_proj,v_proj,fc1,fc2,output_proj", metadata={"help": "where to perform LoRA"})
    #lora_target_modules: str = field(default="Wqkv,dense,mlp.wo", metadata={"help": "where to perform LoRA"})
    lora_target_modules: str = field(default="query,key,value", metadata={"help": "where to perform LoRA"})


# In[ ]:





# In[11]:


model_args = ModelArguments()


# In[12]:


#model_name_or_path = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
#data_path = "/common/zhangz2lab/zhanh/Jupyter_Scripts/cm_spair/"
data_path = "/common/zhangz2lab/zhanh/MFASS/processed_data/snv/"
test_data_path = "/common/zhangz2lab/zhanh/Clinvar/"
#model_name_or_path = "bert-base-uncased"
tokenizer_name = "bpe"
model = SiameseNetwork(model_args.model_name_or_path, num_labels=2)

#for name, param in model.named_parameters():
#    print(name, param.requires_grad)
    
if model_args.use_lora:
        lora_config = LoraConfig(
            r = model_args.lora_r,
            #init_r = 12,
            #target_r = 8,
            #target_modules=list(r"bert\.encoder\.layer\.\d+\.mlp\.wo"),
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            #target_modules = target[1:],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
            #peft_type="ADALORA",
        )
        print(list(model_args.lora_target_modules.split(",")))
        model = get_peft_model(model, lora_config)
        #model = AdaLoraModel(model, lora_config, "default")
        model.print_trainable_parameters()

#tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
###################################################6-mer NT###########################################################
##############################################################################################################
if tokenizer_name == "bpe":
    tokenizer = transformers.AutoTokenizer.from_pretrained(
             model_args.model_name_or_path,
             model_max_length=512,
             padding_side="right",
             use_fast=True,
             trust_remote_code=True,
         )

    tokenizer.eos_token = tokenizer.pad_token
    print("Using bpe tokenizer")
elif tokenizer_name == "6-mer":
    tokenizer = OverlappingTokenizer('/common/zhangz2lab/zhanh/dnabert-config/bert-config-6/vocab_nt.txt')
    print("Using 6-mer tokenizer")
elif tokenizer_name =="one-hot":
    tokenizer = OneHotTokenizer('/common/zhangz2lab/zhanh/dnabert-config/bert-config-6/one_hot.txt')
    print("Using one-hot tokenizer")
else:
    print("wrong value")
#tokenizer = OverlappingTokenizer('/common/zhangz2lab/zhanh/dnabert-config/bert-config-6/vocab.txt')
##############################################################################################################
####################################################6-mer NT##########################################################
###################################################6-mer NT###########################################################
##############################################################################################################    
train_dataset = SiameseDataset(tokenizer, tokenizer_name, os.path.join(data_path, 'train.csv'))
val_dataset = SiameseDataset(tokenizer, tokenizer_name, os.path.join(test_data_path, 'test_small.csv'))
test_dataset = SiameseDataset(tokenizer, tokenizer_name, os.path.join(test_data_path, 'test_small.csv'))
##############################################################################################################
####################################################6-mer NT##########################################################
sequence = "AAAATAAAAAGAAAAA"
token_ids = tokenizer(sequence)
print(token_ids)
sequence = "AAAATAAATAGAAAAA"
token_ids = tokenizer(sequence)
print(token_ids)


# In[13]:


# Define compute_metrics for evaluation
loss_name ="contrastive+BCE"
def compute_metrics(eval_pred):
    if loss_name =="PLLR":
        PLLR, labels = eval_pred
        auc = roc_auc_score(labels, PLLR)
        aupr = average_precision_score(labels, PLLR)
        return {
            'auc': auc,
            'aupr':aupr
        }
    elif loss_name =="cosine":
        logits, labels = eval_pred
#         predictions = (logits > 0.7).astype(np.int32)
#         probabilities = (logits + 1) / 2
        threshold = 0.97  # Example threshold, needs tuning

        # Assuming logits is your cosine_sim
        predictions = (logits < threshold).astype(np.int32)  # Dissimilar if below threshold
        probabilities = 1 - ((logits + 1) / 2)  # Invert the mapping

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        auc = roc_auc_score(labels, probabilities)


        # Plotting the ROC curve
        #plt.figure()
        #plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_curve_auc)
        #plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('Receiver Operating Characteristic (ROC)')
        #plt.legend(loc='lower right')
        #plt.show()

        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    elif loss_name == "contrastive":
        distances, labels = eval_pred
        distances = torch.tensor(distances)
        # Convert distances to probabilities. The larger the distance, the higher the probability of being dissimilar.
        probabilities = torch.sigmoid(distances).numpy()  # Larger distances have higher probabilities
        # Decide on a threshold for predictions
        threshold = 0.3
        predictions = (probabilities > threshold).astype(np.int32)

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        auc = roc_auc_score(labels, probabilities)

        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    else:
        logits_tuple, labels = eval_pred
        
        print(f"Type of logits: {type(logits_tuple)}")
        print(f"Logits content (first few entries): {logits_tuple[:5]}")  # Adjust as needed to avoid large outputs
        logits = logits_tuple[0]
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        # Use the probability of class 1 (dissimilar) for evaluation
        class_1_probabilities = probabilities[:, 1]

        # Predictions based on the class with the higher probability
        predictions = np.argmax(probabilities, axis=1)
        class_1_threshold = 0.3  # Set your own threshold here

        # Predictions based on custom threshold
        predictions = (class_1_probabilities > class_1_threshold).astype(int)

        # Compute evaluation metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        auc = roc_auc_score(labels, class_1_probabilities)
        aupr = average_precision_score(labels, class_1_probabilities)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'aupr':aupr
        }



# def custom_data_collator(data):
#     # Here, we ensure that each item in `data` has the necessary keys.
#     # If not, you can add handling or default values.
#     input_ids1 = torch.stack([item['input_ids1'] for item in data])
#     attention_mask1 = torch.stack([item['attention_mask1'] for item in data])
#     input_ids2 = torch.stack([item['input_ids2'] for item in data])
#     attention_mask2 = torch.stack([item['attention_mask2'] for item in data])

#     # Ensure labels exist or handle its absence
#     #labels = [item.get('labels', torch.tensor(-1)) for item in data]  # Using -1 as a default
#     #labels = torch.stack(labels)
#     labels = torch.stack([item['labels'] for item in data])

#     return {
#         'input_ids1': input_ids1,
#         'attention_mask1': attention_mask1,
#         'input_ids2': input_ids2,
#         'attention_mask2': attention_mask2,
#         'labels': labels
#     }


class GradientInspectionCallback(TrainerCallback):

    def __init__(self, output_dir):
        super(GradientInspectionCallback, self).__init__()
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)  # create directory if it doesn't exist

    def on_backward_end(self, args, state, control, **kwargs):
        print("Checking gradients...")
        model = kwargs["model"]
        
        gradients = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                gradients[name] = param.grad.tolist()  # converting tensor to list
        
        # Save the gradients to a file
        file_name = os.path.join(self.output_dir, f"gradients_step_{state.global_step}.txt")
        with open(file_name, 'w') as f:
            for name, grad in gradients.items():
                f.write(name + ":\n")
                f.write(str(grad) + "\n\n")



class CustomDataCollator(object):
    """
    Custom data collator to handle two input sequences and their respective attention masks.
    """
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, pad_token_id: int):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Check for necessary keys in the instances
        for key in ("input_ids1", "input_ids2", "attention_mask1", "attention_mask2", "labels"):
            if not all(key in instance for instance in instances):
                raise ValueError(f"One or more instances does not contain the key {key}")

        # Extract the respective fields
        input_ids1 = [item['input_ids1'] for item in instances]
        input_ids2 = [item['input_ids2'] for item in instances]
        
        
        
        # Pad the sequences
        input_ids1 = torch.nn.utils.rnn.pad_sequence(
            input_ids1, batch_first=True, padding_value=self.pad_token_id
        )
        input_ids2 = torch.nn.utils.rnn.pad_sequence(
            input_ids2, batch_first=True, padding_value=self.pad_token_id
        )
        
        #input_ids1 = torch.stack([item['input_ids1'] for item in instances])
        #input_ids2 = torch.stack([item['input_ids2'] for item in instances])
        attention_mask1 = torch.stack([item['attention_mask1'] for item in instances])
        attention_mask2 = torch.stack([item['attention_mask2'] for item in instances])
        

    
        labels = torch.stack([item['labels'] for item in instances])

        return {
            'input_ids1': input_ids1,
            'attention_mask1': attention_mask1,
            'input_ids2': input_ids2,
            'attention_mask2': attention_mask2,
            'labels': labels
        }
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    # Check for necessary keys in the instances
        for key in ("input_ids1", "input_ids2", "attention_mask1", "attention_mask2", "labels"):
            if not all(key in instance for instance in instances):
                raise ValueError(f"One or more instances does not contain the key {key}")

        # Convert lists to tensors
        input_ids1 = [torch.tensor(item['input_ids1'], dtype=torch.long) for item in instances]
        input_ids2 = [torch.tensor(item['input_ids2'], dtype=torch.long) for item in instances]
        attention_mask1 = [torch.tensor(item['attention_mask1'], dtype=torch.long) for item in instances]
        attention_mask2 = [torch.tensor(item['attention_mask2'], dtype=torch.long) for item in instances]
        labels = [torch.tensor(item['labels'], dtype=torch.long) for item in instances]

        # Pad the sequences and attention masks
        input_ids1 = torch.nn.utils.rnn.pad_sequence(input_ids1, batch_first=True, padding_value=self.pad_token_id)
        input_ids2 = torch.nn.utils.rnn.pad_sequence(input_ids2, batch_first=True, padding_value=self.pad_token_id)
        attention_mask1 = torch.nn.utils.rnn.pad_sequence(attention_mask1, batch_first=True, padding_value=0)
        attention_mask2 = torch.nn.utils.rnn.pad_sequence(attention_mask2, batch_first=True, padding_value=0)
        labels = torch.stack(labels)

        return {
            'input_ids1': input_ids1,
            'attention_mask1': attention_mask1,
            'input_ids2': input_ids2,
            'attention_mask2': attention_mask2,
            'labels': labels
        }


#custom_callback_instance = CustomDataCollator(tokenizer, pad_token_id=0)

  
# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=CustomDataCollator(tokenizer, pad_token_id=0)
)


test_set_callback = TestSetCallback(trainer=trainer, model=model, test_dataset=test_dataset, eval_steps=training_args.eval_steps, tokenizer=tokenizer)
trainer.add_callback(test_set_callback)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=4)  # Adjust the patience as needed
trainer.add_callback(early_stopping_callback)

#gradient_inspection_callback = GradientInspectionCallback(output_dir = training_args.output_dir)
#trainer.add_callback(gradient_inspection_callback)


# In[14]:


from torch.utils.data import DataLoader

with open(os.path.join(data_path, 'train.csv'), 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # skip the header
    first_row = next(reader)  # reads the first row after the header
    print(first_row[0]) 

sample_sequence = first_row[0]
tokenized_sample = tokenizer.tokenize(sample_sequence)
print(tokenized_sample)
print(len(tokenized_sample))
# Initialize datasets and collator
##############################################################################################################
####################################################6-mer NT##########################################################

train_dataset = SiameseDataset(tokenizer, tokenizer_name, os.path.join(data_path, 'train.csv'))
collator = CustomDataCollator(tokenizer, pad_token_id=0)

##############################################################################################################
####################################################6-mer NT##########################################################
# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collator)

# Fetch and visualize one batch
for batch in train_loader:
    print(batch["input_ids1"].shape)  # Should be [16, seq_length] 
    print(batch["input_ids2"].shape)  # Should be [16, seq_length]
    print(batch["attention_mask1"].shape)  # Should be [16, seq_length]
    print(batch["attention_mask2"].shape)  # Should be [16, seq_length]
    print(batch["labels"].shape)  # Should be [16]
    break


# In[15]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Given your previously defined Dataset and DataLoader:

##############################################################################################################
####################################################6-mer NT##########################################################
train_dataset = SiameseDataset(tokenizer, tokenizer_name, os.path.join(data_path, 'train.csv'))
##############################################################################################################
####################################################6-mer NT##########################################################
# Get the first training sample:
first_sample = train_dataset[0]

# Define your Siamese Network model:
#model = SiameseNetwork('bert-base-uncased', 2)

# Perform a forward pass using the first sample:
with torch.no_grad():
    loss, logits, cosine_sim = model(
        first_sample["input_ids1"].unsqueeze(0).to(device),  # Add batch dimension
        first_sample["attention_mask1"].unsqueeze(0).to(device),  # Add batch dimension
        first_sample["input_ids2"].unsqueeze(0).to(device),  # Add batch dimension
        first_sample["attention_mask2"].unsqueeze(0).to(device),  # Add batch dimension
        first_sample["labels"].unsqueeze(0).to(device)  # Add batch dimension
    )

print(loss.item(), cosine_sim.item())


# In[ ]:


# Training
import logging
logging.basicConfig(level=logging.INFO)

results = trainer.evaluate()
print(results)
trainer.train()

# Evaluate
results = trainer.evaluate()
print(results)


# In[ ]:


import torch
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.cuda.device_count())  # Print the number of GPUs available
print(torch.cuda.get_device_name(1))  # Print the name of the first CUDA device found


# In[ ]:


import torch

for i in range(torch.cuda.device_count()):
    device = torch.device(f'cuda:{i}')
    x = torch.tensor([1.0, 2.0]).to(device)


# In[ ]:


print(torch.__version__)


# In[ ]:


print(torch.cuda.device_count())


# In[ ]:


device = torch.device(f'cuda:1')
x = torch.tensor([1.0, 2.0]).to(device)


# In[ ]:


device = torch.device('cuda:0')
x = torch.tensor([1.0, 2.0]).to(device)
torch.cuda.current_device()


# In[ ]:


torch.cuda.current_device()


# In[ ]:


import torch
print(torch.version.cuda)
print(torch.backends.cudnn.version())


# In[ ]:


import torch
print(torch.__version__)
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")


# In[ ]:




