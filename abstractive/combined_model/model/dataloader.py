from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
import json
import torch

class Dataset1(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length, target_lang, is_mt5):
        fp = open(data_path, 'r')
        self.df = [json.loads(line, strict=False) for line in fp.readlines()]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_mt5 = is_mt5
        self.languages_map = {
            'bn': 'bn_IN',
            'en': 'en_XX',
            'hi': 'hi_IN',
            'ml': 'ml_IN',
            'mr': 'mr_IN',
            'or': 'or_IN',
            'pa': 'pa_IN',
            'ta': 'ta_IN',
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_text = ' '.join(self.df[idx]['references'])
        input_text = str(self.df[idx]['page_title'] + ' ' + self.df[idx]['section_title'] + ' ' + input_text)
        target_text = self.df[idx]['content']
        src_lang_code = self.df[idx]['src_lang']
        tgt_lang_code = self.df[idx]['tgt_lang']
        if src_lang_code not in self.languages_map:
            src_lang_code='en'
        if tgt_lang_code not in self.languages_map:
            tgt_lang_code='en'
        src_lang = self.languages_map[src_lang_code]
        tgt_lang = self.languages_map[tgt_lang_code]

        input_encoding = self.tokenizer(src_lang + ' ' + input_text + ' </s>', return_tensors='pt', max_length=self.max_source_length ,padding='max_length', truncation=True)

        target_encoding = self.tokenizer(tgt_lang + ' ' + target_text + ' </s>', return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)

        input_ids, attention_mask = input_encoding['input_ids'], input_encoding['attention_mask']
        labels = target_encoding['input_ids']

        if self.is_mt5:
            labels[labels == self.tokenizer.pad_token_id] = -100    # for ignoring the cross-entropy loss at padding locations

        return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze(), 'src_lang': src_lang, 'tgt_lang': tgt_lang}    

class DataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
        
    def setup(self, stage=None):
        self.train = Dataset1(self.hparams.train_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.target_lang, self.hparams.is_mt5)
        self.val = Dataset1(self.hparams.val_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.target_lang, self.hparams.is_mt5)
        self.test = Dataset1(self.hparams.test_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.target_lang, self.hparams.is_mt5)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=1,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=1,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=1,shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()
