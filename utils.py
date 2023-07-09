import torch
import datasets
from math import floor, log

def get_device():
    """Get the device to use for training."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def human_readable(number):
    """Thanks to https://stackoverflow.com/a/45478574/13731609"""
    units = ['', 'K', 'M', 'B', 'T']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])


def load_dataset(tokenizer, training_datasets, max_length=None):

    wikitext = datasets.load_dataset("wikitext", "wikitext-2-v1")
    train_dataset = wikitext["train"]
    
    train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_special_tokens_mask=True), batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    train_dataset = train_dataset.map(lambda x: {
        "input_ids": x["input_ids"],
        "attention_mask": x["attention_mask"],
        "labels": x["input_ids"]
    })

    return train_dataset