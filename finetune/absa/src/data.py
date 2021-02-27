"""Data preprocessing and preparing for running ABSA"""

import os
import re
import json
import emoji
import pandas as pd
from vi_prep import ViPrep
from typing import Dict, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

__DATA__ = {
    "hotel": {
        "train": "./data/hotel_train_set.csv",
        "dev": "./data/hotel_train_set.csv",
        "test": "./data/hotel_test_set.csv",
        "labels": "./data/hotel_labels.json",
    },
    "restaurant": {
        "train": "./data/restaurant_train_set.csv",
        "dev": "./data/restaurant_train_set.csv",
        "test": "./data/restaurant_test_set.csv",
        "labels": "./data/restaurant_labels.json",
    }
}

def get_all_hotel_labels():
    """Make labels dictionary for the Hotel domain"""

    __ENTITIES__ = [
        "HOTEL",
        "ROOMS",
        "ROOM_AMENITIES",
        "FACILITIES",
        "SERVICE",
        "LOCATION",
        "FOOD&DRINKS"
    ]

    __ASPECTS__ = [
        "GENERAL",
        "PRICES",
        "DESIGN&FEATURES",
        "CLEANLINESS",
        "COMFORT",
        "QUALITY",
        "STYLE&OPTIONS",
        "MISCELLANEOUS"
    ]

    all_labels = []
    for e in __ENTITIES__:
        for a in __ASPECTS__:
            # if it comes to these cases, do not form the labels
            if (e in ["SERVICE", "LOCATION"] and a in __ASPECTS__[1:]) \
                or (e == "FOOD&DRINKS" and a in ["GENERAL", "DESIGN&FEATURES", "CLEANLINESS", "COMFORT"]) \
                or (e in __ENTITIES__[:-1] and a == "STYLE&OPTIONS"):
                continue
            else:
                all_labels += [
                    f"{{{e}#{a}, negative}}",
                    f"{{{e}#{a}, neutral}}",
                    f"{{{e}#{a}, positive}}"
                ]
    
    all_labels = sorted(all_labels)
    all_labels = {
        label: i for i, label in enumerate(all_labels)
    }

    with open("data/hotel_labels.json", "w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent="\t") 

    return all_labels

def get_label_dicts(domain):
    """Get the all labels dictionary for domain
    If the file does not exist, make new labels dictionary and save to that file
    """
    labels_file = __DATA__[domain]["labels"]
    all_labels_dict = None
    if os.path.isfile(labels_file):
        with open(labels_file, encoding="utf-8") as f:
            all_labels_dict = json.load(f)
    else:
        # only make labels for "Hotel" because "Restaurant" has been provided already
        all_labels_dict = get_all_hotel_labels()

    aspect_polarity_dict = {
        tuple(k.strip("{}").split(", ")): v \
            for k, v in all_labels_dict.items()
    }

    aspect_dict = {}
    count = 0
    for k in aspect_polarity_dict.keys():
        if k[0] not in aspect_dict:
            aspect_dict[k[0]] = count
            count += 1

    return aspect_dict, aspect_polarity_dict

class ABSA(Dataset):
    """The Aspect-Based Sentiment Analysis Dataset class"""
    def __init__(
        self,
        filepath: str,
        non_letter_to_keep: Dict[str, str],
        tokenizer_path: str,
        max_len: int,
        aspect_dict,
        aspect_polarity_dict
    ):
        self.vp = ViPrep(
            **non_letter_to_keep
        )
        self.df = self.get_df(filepath)[:8]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        self.aspect_dict = aspect_dict
        self.aspect_polarity_dict = aspect_polarity_dict

    @staticmethod
    def prep_doc(vp, doc):
        # check for emojis and convert them to text
        prep_doc = doc
        emojis = emoji.emoji_lis(doc)
        for emo in emojis:
            prep_doc.replace(
                emo["emoji"],
                emoji.demojize(string = emo["emoji"], delimiters=("", " "))
            )

        # preprocess pipeline
        # 1. remove characters that are hardly used in written Vietnamese
        # 2. standardize for abbreviation, honirifics, etc
        # 3. tokenize on whitespaces
        prep_doc = vp.pipeline_sentence(doc)

        return prep_doc

    @staticmethod
    def prep_abs(_abs):
        for i, item in enumerate(_abs):
            item = re.sub(" +", " ", item).strip(" ")
            item = tuple(item.strip("{}").split(", "))
            _abs[i] = item

        return _abs

    def get_df(self, filepath) -> pd.DataFrame:
        
        df = pd.read_csv(filepath).drop("ID", axis=1)

        # Preprocess text data
        df["Doc"] = df["Doc"].apply(lambda x: self.prep_doc(self.vp, x))

        # Preprocess labels
        df["Labels"] = df["Labels"].apply(lambda x: eval(x))
        df["Labels"] = df["Labels"].apply(lambda x: self.prep_abs((x)))

        return df

    def get_labels_ids(self, labels):
        aspect_ids = [0] * len(self.aspect_dict)
        aspect_polarity_ids = [0] * len(self.aspect_polarity_dict)

        for label in labels:
            aspect_ids[self.aspect_dict[label[0]]] = 1.0
            aspect_polarity_ids[self.aspect_polarity_dict[label]] = 1.0
        
        return aspect_ids, aspect_polarity_ids

    def get_encoded_doc(self, doc):
        encoding = self.tokenizer(
            doc,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors='pt',
        )

        return encoding
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        doc = self.df["Doc"][item]
        labels = self.df["Labels"][item]

        encoding = self.get_encoded_doc(doc)
        aspect_ids, aspect_polarity_ids = self.get_labels_ids(labels)

        return {
            "doc": doc,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "aspect_ids": torch.Tensor(aspect_ids),
            "aspect_polarity_ids": torch.Tensor(aspect_polarity_ids)
        }

def get_datasets(
    data_files: Dict[str, Union[str, None]],
    non_letter_to_keep: Dict[str, str],
    tokenizer_path: str,
    max_len: int,
    aspect_dict,
    aspect_polarity_dict
) -> Dict[str, Union[ABSA, None]]:

    datasets = {
        "train": ABSA(
            data_files["train"],
            non_letter_to_keep,
            tokenizer_path,
            max_len,
            aspect_dict,
            aspect_polarity_dict
        ) if data_files["train"] else None,
        "dev": ABSA(
            data_files["dev"],
            non_letter_to_keep,
            tokenizer_path,
            max_len,
            aspect_dict,
            aspect_polarity_dict
        ) if data_files["dev"] else None,
        "test": ABSA(
            data_files["test"],
            non_letter_to_keep,
            tokenizer_path,
            max_len,
            aspect_dict,
            aspect_polarity_dict
        ) if data_files["test"] else None,
    }

    return datasets
    
def get_data_loaders(params: Dict) -> Dict[str, DataLoader]:
    """Get data loaders"""

    domain = params["domain"]
    tokenizer = params["tokenizer"]
    max_len = params["max_length"]
    batch_size = params["batch_size"]
    non_letter_to_keep = params["non_letter_to_keep"]

    aspect_dict, aspect_polarity_dict = get_label_dicts(domain)

    data_files = {
        "train": __DATA__[domain]["train"] if params["do_train"] else None,
        "dev": __DATA__[domain]["dev"] if params["do_eval"] else None,
        "test": __DATA__[domain]["test"] if params["do_predict"] else None
    }

    datasets = get_datasets(
        data_files,
        non_letter_to_keep,
        tokenizer,
        max_len,
        aspect_dict,
        aspect_polarity_dict
    )

    data_loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            # num_workers=4,
            # collate_fn=lambda x: x
        ) if datasets["train"] else None,
        "dev": DataLoader(
            datasets["dev"],
            batch_size=batch_size,
            # num_workers=4,
            # collate_fn=lambda dict(x): x
        ) if datasets["dev"] else None,
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            # num_workers=4,
            # collate_fn=lambda dict(x): x
        ) if datasets["test"] else None
    }

    return data_loaders
