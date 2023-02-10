import os
import random
import numpy as np
from torch.utils.data import Dataset

class NextSentenceDataloader(Dataset):
    def __init__(self, tokenizer, target_files, max_data_size=100, per_file = False):
        self.target_files = target_files
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.max_data_size = max_data_size
        self.__per_file = per_file

        self.__bos = tokenizer.special_tokens_map['bos_token']
        self.__eos = tokenizer.special_tokens_map['eos_token']
        self.__sep = tokenizer.special_tokens_map['sep_token']
        self._build()
        
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        # target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        # target_mask = self.targets[index]["attention_mask"].squeeze()

        # labels = np.copy(source_ids)
        return {"input_ids": source_ids, "attention_mask": source_mask, "labels": source_ids}        
                    
    def _build(self):
        if self.__per_file:
            print('per file dataset')
            self.__set_as_file()
        else:
            print('multi line dataset')
            self.__build_sentense()
        # random.shuffle(self.inputs)

    def __build_sentense(self):
        while True:
            if len(self.target_files) == 0:
                break
            if len(self.inputs) >= self.max_data_size:
                break

            idx = random.randint(0, len(self.target_files)-1)
            file_path = self.target_files.pop(idx)
            print('use file...', file_path)
            self.__set_as_line(file_path)
            # self.__set_as_paddinged_line(file_path)

    def __set_as_line(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            li =  f.readlines()
            
            for idx in range(1, len(li)-4):
                source = li[idx] + li[idx+1] + li[idx+2]  + li[idx+3]
                # source = source.strip()

                source = self.__bos + source + self.__eos

                tokenized_inputs = self.tokenizer(source, return_tensors="pt", add_special_tokens=False)
                self.inputs.append(tokenized_inputs)
                if len(self.inputs) >= self.max_data_size:
                    break

    ## ファイルの先頭から終わりまでを<s></s>でくくる. 
    def __set_as_file(self):
        for file_path in self.target_files:
            with open(file_path, "r", encoding="utf-8") as f:
                li =  f.readlines()
            source = ''.join(li)
            source = self.__bos + source + self.__eos
            tokenized_inputs = self.tokenizer(source, return_tensors="pt", add_special_tokens=False)
            self.inputs.append(tokenized_inputs)