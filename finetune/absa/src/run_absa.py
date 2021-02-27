import os
import json
from sys import path
import warnings
import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

import data
from modeling import ABSAClassifier

__MODELS__ = {
    "briverse/vi-electra-small-cased": "small-cased",
    "briverse/vi-electra-small-uncased": "small-uncased",
    "briverse/vi-electra-base-cased": "base-cased",
    "briverse/vi-electra-base-uncased": "base-uncased",
    "briverse/vi-electra-large-cased": "large-cased",
    "briverse/vi-electra-large-uncased": "large-uncased",
    "briverse/vi-electra-large-cased-800": "large-cased-800",
    "briverse/vi-electra-large-uncased-800": "large-uncased-800",
    "vinai/phobert-base": "phobert-base",
    "vinai/phobert-large": "phobert-large",
    "FPTAI/velectra-base-discriminator-cased": "fpt-electra-base-cased"
}

__DEVICE__ = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")

def read_params(params_file: str) -> Dict:
    """Read the JSON formatted params"""

    params = {}
    with open(params_file, "r", encoding="utf-8") as f:
        params = json.load(f)
        
    # check if model and tokenizer are in supported HuggingFace model identifiers
    assert params["model"] in __MODELS__.keys(), \
        f"Model '{params['modeling']['model']}' not found in supported identifiers {list(__MODELS__.keys())}"

    if params["tokenizer"] == None:
        params["tokenizer"] = params["model"]
    elif params["tokenizer"] != params["model"]:
        warnings.warn("You are specifying an unassociated tokenizer with the model.")

        assert params["tokenizer"] in __MODELS__.keys(), \
            f"Tokenizer '{params['modeling']['tokenizer']}' not found in supported identifiers {list(__MODELS__.keys())}"
    
    return params

def logging(
    log_file, epoch, loss, f1_a, f1_b, eval_loss=None, eval_f1_a=None, eval_f1_b=None, is_best=False
):
    with open(log_file, "a+", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write("Epoch,Loss,F1_A,F1_B,Eval_loss,Eval_F1_A,Eval_F1_B,Best\n")

        print_content = f"Epoch {epoch}: loss = {loss}, f1_a = {f1_a}, f1_b = {f1_b}"
        save_content = f"{epoch},{loss},{f1_a},{f1_b}"
        if eval_loss:
            print_content += f", eval_loss = {eval_loss}, eval_f1_a = {eval_f1_a}, eval_f1_b = {eval_f1_b}"
            save_content += f",{eval_loss},{eval_f1_a},{eval_f1_b}"
            if is_best:
                print_content += " -- BEST"
                save_content += ",True\n"
            else:
                save_content += ",False\n"
        else:
            save_content += ",,,,False\n"

        print(print_content)
        f.write(save_content)

def resolve_absa_clf_output(aspect_polarity_ids, aspect_dict, aspect_polarity_dict):
    """
    Params:
        - aspect_polarity_ids: Rounded ABSA Classifier's output/prediction
        - aspect_dict: Aspect dictionary
        - aspect_polarity_dict: Aspect - Polarity dictionary

    Returns: 3-tuple
        - 0: Aspect - Polarity labels
        - 1: Aspect labels
        - 2: Aspect ids
    """
    pos = torch.where(aspect_polarity_ids==1.0)[0].tolist()
    aspect_polarity_labels = []
    aspect_labels = []
    aspect_ids = [0] *len(aspect_dict)

    for p in pos:
        aspect_polarity_labels.append(
            list(aspect_polarity_dict.keys())[
                list(aspect_polarity_dict.values()).index(p)
            ]
        )

    for a in aspect_polarity_labels:
        aspect_labels.append(a[0])
        aspect_ids[aspect_dict[a[0]]] = 1.0

    return aspect_polarity_labels, aspect_labels, torch.Tensor(aspect_ids).float()

def compute_f1(y_true_a, y_pred_a, y_true_b, y_pred_b):

    # print(y_true_a.dtype, y_pred_b.dtype, y_true_b.dtype, y_pred_b.dtype)
    # print(y_true_a, y_pred_a)
    # print(y_true_b, y_pred_b)

    # Phase A: Aspect (Entity-Attribute)
    f1_a = f1_score(
        y_true_a.cpu().detach().numpy(),
        y_pred_a.cpu().detach().numpy(),
        zero_division=0.0
    )

    # Phase B: Full (Aspect-Polarity)
    f1_b = f1_score(
        y_true_b.cpu().detach().numpy(),
        y_pred_b.cpu().detach().numpy(),
        zero_division=0.0
    )

    return f1_a, f1_b

def train_single_epoch(model, data_loader, criterion, optimizer, scheduler, aspect_dict, aspect_polarity_dict, device):
    model.train()

    epoch_loss = []
    epoch_f1_a, epoch_f1_b = [], []

    for batch_data in data_loader:
        # print(type(batch_data))
        batch_input_ids = Variable(batch_data["input_ids"].to(device))
        batch_attention_mask = Variable(batch_data["attention_mask"].to(device))
        batch_aspect_ids = Variable(batch_data["aspect_ids"].float().to(device))
        batch_aspect_polarity_ids = Variable(batch_data["aspect_polarity_ids"].float().to(device))

        optimizer.zero_grad()

        batch_output = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )

        batch_f1_a, batch_f1_b = [], []
        for i, single_aspect_polarity_ids in enumerate(batch_output):
            single_aspect_polarity_ids = torch.round(single_aspect_polarity_ids)
            _, _, single_aspect_ids = resolve_absa_clf_output(
                single_aspect_polarity_ids,
                aspect_dict,
                aspect_polarity_dict
            )
            single_f1_a, single_f1_b = compute_f1(
                batch_aspect_ids[i],
                single_aspect_ids,
                batch_aspect_polarity_ids[i],
                single_aspect_polarity_ids
            )
            batch_f1_a.append(single_f1_a)
            batch_f1_b.append(single_f1_b)
        
        batch_loss = criterion(batch_output, batch_aspect_polarity_ids)

        epoch_loss.append(batch_loss.item())
        epoch_f1_a.append(np.asarray(batch_f1_a).mean())
        epoch_f1_b.append(np.asarray(batch_f1_b).mean())

        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss": np.asarray(epoch_loss).mean(),
        "f1_a": np.asarray(epoch_f1_a).mean(),
        "f1_b": np.asarray(epoch_f1_b).mean()
    }

def eval(model, data_loader, criterion, aspect_dict, aspect_polarity_dict, device):
    model.eval()

    loss = []
    f1_a, f1_b = [], []

    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            batch_input_ids = Variable(batch_data["input_ids"].to(device))
            batch_attention_mask = Variable(batch_data["attention_mask"].to(device))
            batch_aspect_ids = Variable(batch_data["aspect_ids"].float().to(device))
            batch_aspect_polarity_ids = Variable(batch_data["aspect_polarity_ids"].float().to(device))

            batch_output = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )

            batch_f1_a, batch_f1_b = [], []
            for i, single_aspect_polarity_ids in enumerate(batch_output):
                single_aspect_polarity_ids = torch.round(single_aspect_polarity_ids)
                _, _, single_aspect_ids = resolve_absa_clf_output(
                    single_aspect_polarity_ids,
                    aspect_dict,
                    aspect_polarity_dict
                )
                single_f1_a, single_f1_b = compute_f1(
                    batch_aspect_ids[i],
                    single_aspect_ids,
                    batch_aspect_polarity_ids[i],
                    single_aspect_polarity_ids
                )
                batch_f1_a.append(single_f1_a)
                batch_f1_b.append(single_f1_b)
            
            batch_loss = criterion(batch_output, batch_aspect_polarity_ids)

            loss.append(batch_loss.item())
            f1_a.append(np.asarray(batch_f1_a).mean())
            f1_b.append(np.asarray(batch_f1_b).mean())

    return {
        "loss": np.asarray(loss).mean(),
        "f1_a": np.asarray(f1_a).mean(),
        "f1_b": np.asarray(f1_b).mean()
    }

def train(data_loaders, params, aspect_dict, aspect_polarity_dict, path_to_save, device):
    epochs = params["epochs"]
    eval_epochs = params["eval_epochs"]
    criterion = nn.BCELoss().to(device)

    log_file = path_to_save + "/log.csv"

    model = ABSAClassifier(
        params["model"],
        params["dropout"],
        102 if params["domain"] == "hotel" else 36
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"])
    total_steps = len(data_loaders["train"]) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    train_loss = []
    train_f1_a, train_f1_b = [], []
    eval_loss = []
    eval_f1_a, eval_f1_b = [], []
    best_loss, best_f1_a, best_f1_b = float("inf"), 0., 0.
    from_epoch = 0

    # Training has never been run
    if not os.path.isfile(log_file):
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save, exist_ok=True)
    # There are saved checkpoints
    else:
        df_log = pd.read_csv(log_file)
        from_epoch = df_log[df_log["Best"]==True].index.max() \
            if params["from_best"] else df_log.shape[0] - 1 
        checkpoint = torch.load(path_to_save + f"/ckpt-{from_epoch}.pt", map_location=device)

        model.load_state_dict(checkpoint["model"])
        model.to(device)
        optimizer = optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        scheduler.load_state_dict(checkpoint["scheduler"])
        train_loss = checkpoint["train"]["loss"]
        train_f1_a = checkpoint["train"]["f1_a"]
        train_f1_b = checkpoint["train"]["f1_b"]
        eval_loss = checkpoint["eval"]["loss"]
        eval_f1_a = checkpoint["eval"]["f1_a"]
        eval_f1_b = checkpoint["eval"]["f1_b"]
        best_loss, best_f1_a, best_f1_b = df_log["Eval_loss"][from_epoch], \
            df_log["Eval_F1_A"][from_epoch], \
            df_log["Eval_F1_B"][from_epoch]

    from_epoch = 0 if from_epoch == 0 else from_epoch + 1
    
    for epoch in range(epochs):
        train_epoch_out = train_single_epoch(
            model,
            data_loaders["train"],
            criterion,
            optimizer,
            scheduler,
            aspect_dict,
            aspect_polarity_dict,
            device
        )
        model = train_epoch_out["model"]
        criterion = train_epoch_out["criterion"]
        optimizer = train_epoch_out["optimizer"]
        scheduler = train_epoch_out["scheduler"]
        train_loss.append(train_epoch_out["loss"])
        train_f1_a.append(train_epoch_out["f1_a"])
        train_f1_b.append(train_epoch_out["f1_b"])
  
        if (data_loaders["dev"] is not None) and ((epoch + 1) % eval_epochs == 0):
            print("Eval")

            eval_out = eval(
                model,
                data_loaders["dev"],
                criterion,
                aspect_dict,
                aspect_polarity_dict,
                device
            )
            eval_loss.append(eval_out["loss"])
            eval_f1_a.append(eval_out["f1_a"])
            eval_f1_b.append(eval_out["f1_b"])
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train": {"loss": train_loss, "f1_a": train_f1_a, "f1_b": train_f1_b},
                "eval": {"loss": eval_loss, "f1_a": eval_f1_a, "f1_b": eval_f1_b}
            }

            torch.save(
                checkpoint,
                path_to_save + f"/ckpt-{epoch + from_epoch}.pt"
            )

            # The new best model is the model that is equal to or better than the current best model
            # on all the three conditions: eval_loss, eval_f1_a, eval_f1_b
            if eval_out["loss"] <= best_loss and \
                eval_out["f1_a"] >= best_f1_a and \
                eval_out["f1_b"] >= best_f1_b:
                logging(log_file, epoch + from_epoch,
                    train_epoch_out["loss"], train_epoch_out["f1_a"], train_epoch_out["f1_b"],
                    eval_out["loss"], eval_out["f1_a"], eval_out["f1_b"],
                    True
                )
            else:
                logging(log_file, epoch + from_epoch,
                    train_epoch_out["loss"], train_epoch_out["f1_a"], train_epoch_out["f1_b"],
                    eval_out["loss"], eval_out["f1_a"], eval_out["f1_b"],
                    False
                )
        else:
            logging(log_file, epoch + from_epoch,
                train_epoch_out["loss"], train_epoch_out["f1_a"], train_epoch_out["f1_b"]
            )

    return model

def predict(model, data_loader, aspect_dict, aspect_polarity_dict, path_to_save, device):
    model.eval()

    criterion = nn.BCELoss().to(device)
    loss = []
    f1_a, f1_b = [], []

    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            batch_input_ids = Variable(batch_data["input_ids"].to(device))
            batch_attention_mask = Variable(batch_data["attention_mask"].to(device))
            batch_aspect_ids = Variable(batch_data["aspect_ids"].float().to(device))
            batch_aspect_polarity_ids = Variable(batch_data["aspect_polarity_ids"].float().to(device))

            batch_output = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )

            batch_f1_a, batch_f1_b = [], []
            for i, single_aspect_polarity_ids in enumerate(batch_output):
                single_aspect_polarity_ids = torch.round(single_aspect_polarity_ids)
                _, _, single_aspect_ids = resolve_absa_clf_output(
                    single_aspect_polarity_ids,
                    aspect_dict,
                    aspect_polarity_dict
                )
                single_f1_a, single_f1_b = compute_f1(
                    batch_aspect_ids[i],
                    single_aspect_ids,
                    batch_aspect_polarity_ids[i],
                    single_aspect_polarity_ids
                )
                batch_f1_a.append(single_f1_a)
                batch_f1_b.append(single_f1_b)
            
            batch_loss = criterion(batch_output, batch_aspect_polarity_ids)

            loss.append(batch_loss.item())
            f1_a.append(np.asarray(batch_f1_a).mean())
            f1_b.append(np.asarray(batch_f1_b).mean())

    result_file = path_to_save + "/result.txt"
    with open(path_to_save, "a+", encoding="utf-8") as f:
        from datetime import datetime
        f.write(str(datetime.now()) + "\n")
        f.write(f"Loss = {loss}\nF1_A = {f1_a}\nF1_B = {f1_b}\n\n")

    return {
        "loss": np.asarray(loss).mean(),
        "f1_a": np.asarray(f1_a).mean(),
        "f1_b": np.asarray(f1_b).mean()
    }

def main(args):
    params = read_params(args.params)
    # print(params)

    data_loaders = data.get_data_loaders(params)
    # print(data_loaders)

    model_name = __MODELS__[params["model"]]
    path_to_save = params["output_dir"] + fr"/{model_name}"
    device = __DEVICE__
    aspect_dict, aspect_polarity_dict = data.get_label_dicts(params["domain"])

    if params["do_train"]:
        print("TRAINING")
        model = train(data_loaders, params, aspect_dict, aspect_polarity_dict, path_to_save, device)

    if data_loaders["test"] is not None:
        print("PREDICTING")
        predict_out = predict(model, data_loaders["test"], aspect_dict, aspect_polarity_dict, path_to_save, device)
        print(f"Loss = {predict_out['loss']}\nF1_A = {predict_out['f1_a']}\nF1_B = {predict_out['f1_b']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Aspect-Based Sentiment Analysis"
    )
    # parser.add_argument(
    #     "domain",
    #     type=str,
    #     help="The domain to be analysed"
    # )
    parser.add_argument(
        "--params",
        type=str,
        default="src/params.json",
        help="The JSON file containings parameters for data processing, modeling, training,..."
    )
    args = parser.parse_args()

    main(args)
