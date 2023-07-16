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

    supported_datasets = {
        "wikipedia": ("wikipedia", "20220301.en"),
        "openwebtext": ("openwebtext", "plain_text"),
        "c4": ("c4", "en")
    }

    def encode(batch):
        return tokenizer(batch["text"], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

    _datasets = []
    for ds in training_datasets:
        if ds in supported_datasets:
            name, variant = supported_datasets[ds]
            d = datasets.load_dataset(name, variant, streaming=True)["train"]
            d = d.map(encode, batched=True)
            d = d.map(lambda x: {
                "input_ids": x["input_ids"],
                "attention_mask": x["attention_mask"],
                "labels": x["input_ids"]
            })
            _datasets.append(d)

    return datasets.interleave_datasets(_datasets)