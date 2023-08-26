#!/usr/bin/env python
# coding: utf-8


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
from transformers import Trainer, TrainingArguments
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
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





from transformers import TrainerCallback, TrainerControl

class TestSetCallback(TrainerCallback):
    def __init__(self, trainer, test_dataset, eval_steps):
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.eval_steps = eval_steps
        
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

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            predictions = self.trainer.predict(self.test_dataset)
            logits = predictions.predictions
            labels = predictions.label_ids

            # Plot the ROC curve
            self.plot_roc(labels, logits)
            #results = self.trainer.evaluate(eval_dataset=self.test_dataset)
            self.trainer.model.train()
            
            #print(f"Results on test set at step {state.global_step}: {results}")




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
    def __init__(self, model_name_or_path, num_labels, loss_type="cosine", margin=2.0, cache_dir=None):
        super(SiameseNetwork, self).__init__()

        self.loss_type = loss_type
        self.margin = margin

        # Load the base model
        self.base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            output_hidden_states=True
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels):
        # Encoding sequences using the same base model
        outputs1 = self.base_model(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.base_model(input_ids=input_ids2, attention_mask=attention_mask2)

        last_hidden_state1 = outputs1.hidden_states[-1][:, 0, :]
        last_hidden_state2 = outputs2.hidden_states[-1][:, 0, :]

        if self.loss_type == "cosine":
            cosine_sim = F.cosine_similarity(last_hidden_state1, last_hidden_state2, dim=-1)
            mapped_sim = (cosine_sim + 1) / 2  # map between 0 and 1
            loss = F.mse_loss(mapped_sim, labels.float())
            return (loss, cosine_sim)
        
        elif self.loss_type == "contrastive":
            euclidean_distance = F.pairwise_distance(last_hidden_state1, last_hidden_state2)
            # Contrastive loss
            loss = (1 - labels) * torch.pow(euclidean_distance, 2) +                    labels * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
            loss = torch.mean(loss)
            return (loss, euclidean_distance)

        else:
            raise ValueError("Invalid loss_type specified.")




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

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    #model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    num_train_epochs: int = field(default=15)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    load_best_model_at_end: bool = field(default=True)     # load the best model when finished training (default metric is loss)
    metric_for_best_model: str = field(default="eval_loss") # the metric to use to compare models
    greater_is_better: bool = field(default=False)           # whether the `metric_for_best_model` should be maximized or not
    logging_strategy: str = field(default="steps")  # Log every "steps"
    logging_steps: int = field(default=100)  # Log every 100 steps
    warmup_ratio: int = field(default=0.1)
    weight_decay: float = field(default=1e-4)
    learning_rate: float = field(default=2e-5)
    lr_scheduler_type: str = field(default='linear')
    save_total_limit: int = field(default=5)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="/common/zhangz2lab/zhanh/Jupyter_Scripts/output_0825/results")
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




@dataclass
class ModelArguments:
    #model_name_or_path: Optional[str] = field(default="bert-base-uncased")
    model_name_or_path: Optional[str] = field(default="InstaDeepAI/nucleotide-transformer-500m-human-ref")
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










model_args = ModelArguments()




#model_name_or_path = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
data_path = "/common/zhangz2lab/zhanh/Jupyter_Scripts/cm_spair/"
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
    tokenizer = OverlappingTokenizer('/common/zhangz2lab/zhanh/dnabert-config/bert-config-6/vocab.txt')
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
train_dataset = SiameseDataset(tokenizer, tokenizer_name, os.path.join(data_path, 'train_400.csv'))
val_dataset = SiameseDataset(tokenizer, tokenizer_name, os.path.join(data_path, 'test_400.csv'))
test_dataset = SiameseDataset(tokenizer, tokenizer_name, os.path.join(data_path, 'test_400.csv'))
##############################################################################################################
####################################################6-mer NT##########################################################
sequence = "AAAATAAAAAGAAAAA"
token_ids = tokenizer(sequence)
print(token_ids)
sequence = "AAAATAAATAGAAAAA"
token_ids = tokenizer(sequence)
print(token_ids)




# Define compute_metrics for evaluation
loss_name ="cosine"
def compute_metrics(eval_pred):
    if loss_name =="cosine":
        logits, labels = eval_pred
        #predictions = np.argmax(logits, axis=-1)
        predictions = (logits > 0.7).astype(np.int32)
        probabilities = (logits + 1) / 2

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
    else:
        distances, labels = eval_pred
        distances = torch.tensor(distances)
        # Convert distances to probabilities. The smaller the distance, the higher the probability of being similar.
        probabilities = torch.sigmoid(-distances).numpy()  # Negating distance to ensure smaller distances have higher probabilities

        # Decide on a threshold for predictions
        threshold = 0.5
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


    
# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=CustomDataCollator(tokenizer, pad_token_id=0)
)


test_set_callback = TestSetCallback(trainer=trainer, test_dataset=test_dataset, eval_steps=training_args.eval_steps)

#trainer.callbacks.append(test_set_callback)
trainer.add_callback(test_set_callback)
#gradient_inspection_callback = GradientInspectionCallback(output_dir = training_args.output_dir)
#trainer.add_callback(gradient_inspection_callback)




from torch.utils.data import DataLoader

with open(os.path.join(data_path, 'train_400.csv'), 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # skip the header
    first_row = next(reader)  # reads the first row after the header
    print(first_row[0]) 

sample_sequence = first_row[0]
tokenized_sample = tokenizer.tokenize(sample_sequence)
print(tokenized_sample)

# Initialize datasets and collator
##############################################################################################################
####################################################6-mer NT##########################################################

train_dataset = SiameseDataset(tokenizer, tokenizer_name, os.path.join(data_path, 'train_400.csv'))
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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Given your previously defined Dataset and DataLoader:

##############################################################################################################
####################################################6-mer NT##########################################################
train_dataset = SiameseDataset(tokenizer, tokenizer_name, os.path.join(data_path, 'train_400.csv'))
##############################################################################################################
####################################################6-mer NT##########################################################
# Get the first training sample:
first_sample = train_dataset[0]

# Define your Siamese Network model:
#model = SiameseNetwork('bert-base-uncased', 2)

# Perform a forward pass using the first sample:
with torch.no_grad():
    loss, cosine_sim = model(
        first_sample["input_ids1"].unsqueeze(0).to(device),  # Add batch dimension
        first_sample["attention_mask1"].unsqueeze(0).to(device),  # Add batch dimension
        first_sample["input_ids2"].unsqueeze(0).to(device),  # Add batch dimension
        first_sample["attention_mask2"].unsqueeze(0).to(device),  # Add batch dimension
        first_sample["labels"].unsqueeze(0).to(device)  # Add batch dimension
    )

print(loss.item(), cosine_sim.item())




# Training
import logging
logging.basicConfig(level=logging.INFO)
trainer.train()

# Evaluate
#results = trainer.evaluate()
#print(results)





