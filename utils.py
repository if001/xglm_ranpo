import torch
import numpy as np
from dataloader import NextSentenceDataloader


## todo huggingfaceのサンプルによると,train dataはblock sizeごとに並び替えてたのでここで並び替え
def apply_group(ds, block_size = 256):
    concatenated = { k: [] for k in ds[0].keys() }
    for v in ds:
        for k in ds[0].keys():
            concatenated[k].extend(tensor2np(v[k]))
        
    total_length = len(concatenated[list(concatenated.keys())[0]])
    
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    

    results = []    
    for i in range(0, total_length, block_size):
        r = { k:  np2tensor(np.array(t[i: i + block_size])) for k,t in concatenated.items()}
        r["labels"] = r["input_ids"].clone()
        results.append(r)

    return results

def prepare_data_set(tokenizer, files, max_seq_length, max_dataset_length, per_file):
    ds = NextSentenceDataloader(
            tokenizer,
            files,            
            max_dataset_length,
            per_file
            )        
    ds = apply_group(ds, max_seq_length)
        
    train_size = int(len(ds) * 0.95)
    val_size = len(ds) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset=ds, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(42))

    print("data_set:", len(ds))
    print("train_data:", len(train_data))
    print("val_data:", len(val_data))
    return train_data, val_data


def get_device():
    if torch.cuda.is_available():
        print("use gpu...")
        return "cuda:0"
    else:
        return "cpu"

def np2tensor(x):
    return torch.from_numpy(x).clone()
    
def tensor2np(x):
    return x.to('cpu').detach().numpy().copy()

