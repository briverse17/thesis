import os
import json
import inspect
import warnings
 
import torch
import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from transformers import AutoModel, AutoTokenizer

__MODELS__ = [
    'vinai/phobert-base', 'vinai/phobert-large',
    'bert-base-cased', 'bert-base-uncased',
    'bert-base-multilingual-cased', 'bert-base-multilingual-uncased'
]

__LAYERS__ = list(range(1, 13))

__STRATEGIES__ = ['average', 'concat', 'sum']

class Data:
    def __init__(self,
                 input="test/input",
                 output="test/output",
                 input_batches=[],
                 tokenized_batches=[
                    {
                        "input_ids": None,
                        "attention_mask": None
                    }
                 ],
                 embedded_batches=[],
                 load=False):
        self.input = input
        self.output = output
        self.input_batches = input_batches
        self.tokenized_batches = tokenized_batches
        self.embedded_batches = embedded_batches
        if load:
            self.load()

    def from_config(self, config_path, load=False):
        with open(config_path) as f:
            config = json.load(f)["data"]
            self.input = config["input"]
            self.output = config["output"]
            if load:
                self.load()
        return self

    def load(self):
        return self.__load()
    
    def dump(self):
        return self.__dump()
    
    def __load(self):
        if os.path.isfile(self.input):
            try:
                with open(self.input, "r", encoding="utf-8") as f:
                    self.input_batches.append(f.read().split("\n"))
            except:
                raise Exception(f"Cannot open {self.input}")
        else: # os.path.isdir(self.input) == True
            dirs = os.listdir(self.input)
            for d in dirs:
                try:
                    with open(os.path.join(self.input, d), "r", encoding="utf-8") as f:
                        self.input_batches.append(f.read().split("\n"))
                except:
                    raise Exception(f"Cannot open {os.path.join(self.input, d)}")
        self.tokenized_batches = len(self.input_batches) * self.tokenized_batches

    def __dump(self):
        pass

class Tools:
    def __init__(self,
                 model_path="vinai/phobert-base",
                 tokenizer_path="vinai/phobert-base",
                 max_length=512,
                 padding="max_length",
                 truncation=True,
                 return_tensors="pt",
                 load=False):
        self.model_path = model_path
        if tokenizer_path != None:
            self.tokenizer_path = tokenizer_path
        else:
            self.tokenizer_path = self.model_path
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        print(self.model_path, self.tokenizer_path)
        self.validate()
        if load:
            self.load()
    
    def validate(self):
        if self.model_path not in __MODELS__:
            raise Exception(f"Expecting model in {__MODELS__}, got {self.model_path}")
        if self.tokenizer_path not in __MODELS__:
            raise Exception(f"Expecting tokenizer in {__MODELS__}, got {self.tokenizer_path}")
    
    def load(self):
        return self.__load()
    
    def __load(self):
        print(self.model_path, self.tokenizer_path)
        self.model = AutoModel.from_pretrained(self.model_path, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)

class Params:
    def __init__(self,
                 layers=(12,),
                 layers_ascending=False,
                 strategy="average"):
        self.layers = layers
        self.layers_ascending = layers_ascending
        self.strategy = strategy
        self.validate()
        self.sort_layers()

    def validate(self):
        def validate_layers(layers, layers_ascending):
            if not set(layers).issubset(__LAYERS__):
                raise Exception(f"Expecting layers in range {__LAYERS__}, got {layers}")
            if len(layers) > 4:
                warnings.warn("You are getting embeddings of more than 4 layers,\
                                which can be huge.")
            if layers_ascending == False:
                warnings.warn("'layers_ascending' == False means the script won't re-arrange the layers.")
                
        def validate_strategy(strategy):
            if strategy not in __STRATEGIES__:
                raise Exception(f"Expecting strategies in {__STRATEGIES__}, got {strategy}")
        
        validate_layers(self.layers, self.layers_ascending)
        validate_strategy(self.strategy)
                                
    def sort_layers(self):
        return self.__sort_layers()
        
    def __sort_layers(self):
        if self.layers_ascending and self.layers != "all":
            self.layers = sorted(self.layers)
            
class Embedder:
    def __init__(self, load_tools=False):
        self.tools = Tools(load=load_tools)
        self.params = Params()
    
    def from_config(self, config_path="config.json", load_tools=False):
        with open(config_path) as f:
            config = json.load(f)["embedder"]
            self.tools = Tools(config["tools"]["model_path"],
                               config["tools"]["tokenizer_path"],
                               config["tools"]["max_length"],
                               config["tools"]["padding"],
                               config["tools"]["truncation"],
                               config["tools"]["return_tensors"],
                               load=load_tools)
            
            if len(config["params"]["layers"]) != set(config["params"]["layers"]):
                warnings.warn("Your 'layers' contains duplicated layers. They will be removed.")
                config["params"]["layers"] = list(dict.fromkeys(config["params"]["layers"]).keys())
            
            self.params = Params(config["params"]["layers"],
                                 config["params"]["layers_ascending"],
                                 config["params"]["strategy"])
        return self

    def tokenize_batches(self, data):
        return self.__tokenize_batches(data)
    
    def __tokenize_batches(self, data):
        for i, batch in enumerate(data.input_batches):
            tokenizer_output = self.tools.tokenizer(batch,
                                              max_length=self.tools.max_length,
                                              padding=self.tools.padding,
                                              truncation=self.tools.truncation,
                                              return_tensors=self.tools.return_tensors)
            data.tokenized_batches[i]["input_ids"] = tokenizer_output["input_ids"]
            data.tokenized_batches[i]["attention_mask"] = tokenizer_output["attention_mask"]

    def bert_embed(self, data) -> torch.Tensor:
        def combine_layers(embeddings: torch.Tensor,
                           strategy: str = "average"):
            combined_embeddings = torch.Tensor()    
            if strategy == "average":
                combined_embeddings = torch.mean(hidden_states, dim=0)
            elif strategy == "concat":
                combined_embeddings = torch.cat(tuple(hidden_states), dim=2)
            else: # strategy == "sum"
                combined_embeddings = torch.sum(hidden_states, dim=0)

            return combined_embeddings
        
        with torch.no_grad():
            for i, batch in enumerate(data.tokenized_batches):
                # initially, batch_hidden_states is a tuple of torch.Tensor tensors
                batch_hidden_states = self.tools.model(batch["input_ids"],
                                                       batch["attention_mask"])[-1][1:]

                # here we stack them together to make one big tensor and call it "batch_embeddings"
                # batch_embeddings dimensions:
                # 0: BERT hidden layers (12)
                # 1: batch size (number of sentences in the batch)
                # 2: tokens of sentences (len = max_length over all sentences in the batch)
                # 3: hidden_sizes (base: 768, large: 1024)
                batch_embeddings = torch.stack(batch_hidden_states, dim=0)
                batch_combined_embeddings = torch.Tensor()
                if  self.params.layers == ["all"]:
                    # all layer embedding
                    batch_combined_embeddings = combine_layers_embeddings(batch_embeddings,
                                                                          self.params.strategy)
                elif len(self.params.layers) == 1:
                    # single layer embedding
                    # just return the embedding at that layer
                    # cuz we need not to do any combination strategy
                    batch_combined_embeddings = embeddings[self.params.layers[-1]]
                else:
                    # multi layer embedding
                    # first, we filter out the layers
                    multi_layer_embeddings = torch.stack([embeddings[i] for i in self.params.layers])
                    # do combining
                    batch_combined_embeddings = combine_layers_embeddings(multi_layer_embeddings,
                                                                          self.params.strategy)
            
                data.embedded_batches[i].append(batch_combined_embeddings)
