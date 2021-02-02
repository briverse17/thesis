import os
import json

import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union
from transformers import AutoModel, AutoTokenizer

__MODELS__ = [
    'vinai/phobert-base', 'vinai/phobert-large',
    'bert-base-cased', 'bert-base-uncased',
    'bert-base-multilingual-cased', 'bert-base-multilingual-uncased'
]

__LAYERS__ = list(range(1, 12))

__STRATEGIES__ = ['mean', 'concat', 'sum']

class Data():
    def __init__(
        self,
        input="test/input",
        output="test/output"
    ):
        self.input = input
        self.output = output
        self.raw_batches = []
        self.tokenized_batches = []
        self.embedded_batches = []
    
    def load(self):
        self.file_names = []
        inp_paths = []
        if os.path.isfile(self.input):
            self.file_names.append(os.path.basename(self.input))
            inp_paths.append(self.input)
        else:
            self.file_names = os.listdir(self.input)
            inp_paths = [os.path.join(self.input, fn) for fn in self.file_names]
        
        print("Loading data...")
        for inp in tqdm(inp_paths):
            with open(inp, "r", encoding="utf-8") as f:
                self.raw_batches.append(f.read().split("\n"))
        
        return self

    def dump(self):
        if not os.path.isdir(self.output):
            os.mkdir(self.output)

        for i, batch in tqdm(enumerate(self.embedded_batches)):
            if type(batch) != np.array:
                batch = batch.cpu().detach().numpy()
            npy_file = os.path.splitext(self.file_names[i])[0] + ".npy"
            out = os.path.join(self.output, npy_file)
            np.save(out, batch, allow_pickle=True)

class Embedder():
    def __init__(self, params_file : str = None):
        if params_file != None:
            self.read_params(params_file)
        else:
            self.model = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True)
            self.toker = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.toker_kwargs = {
                "max_length": None,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt"      
            }
            self.data = Data()
            
            self.layers = [11]
            self.strategy = "mean"
            self.dump = True
    
    def read_params(self, params_file):
        params = {}
        with open(params_file, "r") as pf:
            params = json.load(pf)
        
        self.model = AutoModel.from_pretrained(params["model"]["path"], output_hidden_states=True)
        if params["toker"]["path"]:
            self.toker = AutoTokenizer.from_pretrained(params["toker"]["path"])
        else:
            self.toker = AutoTokenizer.from_pretrained(params["model"]["path"])
        self.toker_kwargs = params["tokers"]["kwargs"]
        self.data = Data(
            params["data"]["input"],
            params["data"]["output"],
        )

        self.layers = list(dict.fromkeys(params["embedder"]["layers"]))
        if (params["embedder"]["layers_asc"]):
            self.layers = sorted(self.layers)
        self.strategy = params["embedder"]["strategy"]
        self.dump = params["embedder"]["dump"]

    def tokenize(self):
        print("Tokenizing...")
        for batch in tqdm(self.data.raw_batches):
            toker_output = self.toker(batch, **self.toker_kwargs)
            temp_dict = {
                "input_ids": toker_output["input_ids"],
                "attention_mask": toker_output["attention_mask"]
            }
            self.data.tokenized_batches.append(temp_dict)

        return self.data

    @staticmethod
    def combine_layers(embeddings: torch.Tensor, strategy: str = "mean"):
        combined_embeddings = torch.Tensor()    
        if strategy == "mean":
            combined_embeddings = torch.mean(embeddings, dim=0)
        elif strategy == "concat":
            combined_embeddings = torch.cat(tuple(embeddings), dim=2)
        else: # strategy == "sum"
            combined_embeddings = torch.sum(embeddings, dim=0)

        return combined_embeddings

    def bert_embed(self):
        print("Embedding...")
        with torch.no_grad():
            for batch in tqdm(self.data.tokenized_batches):
                batch_hidden_states = self.model(
                    batch["input_ids"],
                    batch["attention_mask"]
                )[-1][1:]
                # here we stack them together to make one big tensor and call it "batch_embeddings"
                # batch_embeddings dimensions:
                # 0: BERT hidden layers (12)
                # 1: batch size (number of sentences in the batch)
                # 2: tokens of sentences (len = max_length over all sentences in the batch)
                # 3: hidden_sizes (base: 768, large: 1024)
                batch_embeddings = torch.stack(batch_hidden_states, dim=0)
                batch_combined_embeddings = torch.Tensor()

                if len(self.layers) == 1:
                    # single layer embedding
                    # just return the embedding at that layer
                    # cuz we needn't do any combination
                    batch_combined_embeddings = batch_embeddings[self.layers[-1]]
                else:
                    # multi layer embedding
                    # first, we filter out the layers
                    multi_layer_embeddings = torch.stack([batch_embeddings[i] for i in self.layers])
                    # do combining
                    batch_combined_embeddings = self.combine_layers(multi_layer_embeddings, self.strategy)
            
                self.data.embedded_batches.append(batch_combined_embeddings)

        if self.dump:
            self.data.dump()
            print(f"Embedded vectors are dumped to {self.data.output}")

        return self.data
