from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from indicnlp.transliterate import unicode_transliterate
from transformers import MBartForConditionalGeneration, MT5ForConditionalGeneration, AutoConfig, AutoModelForSeq2SeqLM, MBartTokenizer
import torch
import argparse
from rouge import Rouge
import sys
sys.setrecursionlimit(1024 * 1024 + 10)

class Dataset1(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length, is_mt5):
        fp = open(data_path, 'r')
        self.df = [json.loads(line, strict=False) for line in fp.readlines()]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_mt5 = is_mt5
        self.languages_map = {
            'bn': {0:'bn_IN'},
            'en': {0:'en_XX'},
            'hi': {0:'hi_IN'},
            'ml': {0:'ml_IN'},
            'mr': {0:'mr_IN'},
            'or': {0:'or_IN'},
            'pa': {0:'pa_IN'},
            'ta': {0:'ta_IN'},
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
        src_lang = self.languages_map[src_lang_code][0]
        tgt_lang = self.languages_map[tgt_lang_code][0]

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
        self.train = Dataset1(self.hparams.train_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)
        self.val = Dataset1(self.hparams.val_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)
        self.test = Dataset1(self.hparams.test_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=1,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=1,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=1,shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()
        

class Summarizer(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.rouge = Rouge()
        if self.hparams.is_mt5:
            self.model = MT5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)

        self.languages_map = {
            'bn': {0:'bn_IN'},
            'en': {0:'en_XX'},
            'hi': {0:'hi_IN'},
            'ml': {0:'ml_IN'},
            'mr': {0:'mr_IN'},
            'or': {0:'or_IN'},
            'pa': {0:'pa_IN'},
            'ta': {0:'ta_IN'},
        }

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def _step(self, batch):
        input_ids, attention_mask, labels, src_lang, tgt_lang = batch['input_ids'], batch['attention_mask'], batch['labels'], batch['src_lang'], batch['tgt_lang']
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs[0]
        return loss
    
    def _generative_step(self, batch):
        try:
            token_id = self.hparams.tokenizer.lang_code_to_id[batch['tgt_lang'][0]]
            self.hparams.tokenizer.tgt_lang = batch['tgt_lang'][0]
        except:
            token_id = 250010
            self.hparams.tokenizer.tgt_lang = 'hi_IN'
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            use_cache=True,
            num_beams=self.hparams.eval_beams,
            forced_bos_token_id=token_id,
            max_length=self.hparams.tgt_max_seq_len #understand above 3 arguments
            )


        input_text = self.hparams.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        pred_text = self.hparams.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if self.hparams.is_mt5:
            batch['labels'][batch['labels'] == -100] = self.hparams.tokenizer.pad_token_id
        ref_text = self.hparams.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        src_lang = batch['src_lang']
        tgt_lang = batch['tgt_lang']

        return input_text, pred_text, ref_text, src_lang, tgt_lang

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        # input_text, pred_text, ref_text = self._generative_step(batch)
        self.log("val_loss", loss, on_epoch=True)
        return 

    def validation_epoch_end(self, outputs):

        pred_text = []
        ref_text = []
        for x in outputs:
            pred = x['pred_text']
            if pred[0] == '':
                pred[0] = 'default text'
                pred_text.extend(pred)
            else:
                pred_text.extend(pred)

            ref = x['ref_text']
            if ref[0] == '':
                ref[0] = 'default text'
                ref_text.extend(ref)
            else:
                ref_text.extend(ref)

        rouge = self.rouge.get_scores(pred_text, ref_text, avg=True)

        self.log("val_rouge-1_prec", rouge['rouge-1']['p'])
        self.log("val_rouge-1_rec", rouge['rouge-1']['r'])
        self.log("val_rouge-1_f1", rouge['rouge-1']['f'])

        self.log("val_rouge-2_prec", rouge['rouge-2']['p'])
        self.log("val_rouge-2_rec", rouge['rouge-2']['r'])
        self.log("val_rouge-2_f1", rouge['rouge-2']['f'])

        self.log("val_rouge-l_prec", rouge['rouge-l']['p'])
        self.log("val_rouge-l_rec", rouge['rouge-l']['r'])
        self.log("val_rouge-l_f1", rouge['rouge-l']['f'])
        return


    def predict_step(self, batch, batch_idx):
        input_text, pred_text, ref_text, src_lang, tgt_lang = self._generative_step(batch)
        return {'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text}

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        input_text, pred_text, ref_text, src_lang, tgt_lang = self._generative_step(batch)
        return {'test_loss': loss, 'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text, 'src_lang': src_lang, 'tgt_lang': tgt_lang}

    def test_epoch_end(self, outputs):
        input_texts = []
        pred_texts = []
        ref_texts = []
        src_langs = []
        tgt_langs = []
        
        for x in outputs:
            if x['pred_text'][0] == '':
                x['pred_text'][0] = 'pred_text'
            if x['ref_text'][0] == '':
                x['ref_text'][0] = 'ref_text'
            input_texts.extend(x['input_text'])
            pred_texts.extend(x['pred_text'])
            ref_texts.extend(x['ref_text'])
            src_langs.extend(x['src_lang'])
            tgt_langs.extend(x['tgt_lang'])
        
        for key, value in self.languages_map.items():
            self.languages_map[key]['original_pred_text'] = [p for p, t in zip(pred_texts, tgt_langs) if value[0] == t]
            self.languages_map[key]['original_ref_text'] = [r for r, t in zip(ref_texts, tgt_langs) if value[0] == t]
            self.languages_map[key]['original_input_text'] = [i for i, t in zip(input_texts, tgt_langs) if value[0] == t]
            self.languages_map[key]['src_lang'] = [s for s, t in zip(src_langs, tgt_langs) if value[0] == t]
            self.languages_map[key]['tgt_lang'] = [t for t in tgt_langs if value[0] == t]

        overall_rouge = {'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}
        ln = 0
        for key in self.languages_map:
            try:
                self.languages_map[key]['rouge'] = self.rouge.get_scores(self.languages_map[key]['original_pred_text'], self.languages_map[key]['original_ref_text'], avg=True)
                logger.log_text(f"test_rouge_{key}", dataframe=pd.DataFrame(self.languages_map[key]['rouge']))
                for (k1, v1), (k2, v2) in zip(overall_rouge.items(), self.languages_map[key]['rouge'].items()):
                    overall_rouge[k1]['r'] += self.languages_map[key]['rouge'][k2]['r']
                    overall_rouge[k1]['p'] += self.languages_map[key]['rouge'][k2]['p']
                    overall_rouge[k1]['f'] += self.languages_map[key]['rouge'][k2]['f']
                ln+=1
            except:
                pass

        for k, v in overall_rouge.items():
            overall_rouge[k]['r']/=ln
            overall_rouge[k]['p']/= ln
            overall_rouge[k]['f']/=ln
        logger.log_text("test_rouge", dataframe=pd.DataFrame(overall_rouge))

        df_to_write = pd.DataFrame()
        for key in self.languages_map:
            l = len(self.languages_map[key]['original_pred_text'])
            self.languages_map[key]['rouges'] = [self.rouge.get_scores(self.languages_map[key]['original_pred_text'][i], self.languages_map[key]['original_ref_text'][i]) for i in range(len(self.languages_map[key]['original_pred_text']))]
            df_key = pd.DataFrame({
                'src_lang':[self.languages_map[key]['src_lang'][i] for i in range(l)],
                'input_text':[self.languages_map[key]['original_input_text'][i] for i in range(l)],
                'tgt_lang':[self.languages_map[key]['tgt_lang'][i] for i in range(l)],
                'ref_text':[self.languages_map[key]['original_ref_text'][i] for i in range(l)],
                'pred_text':[self.languages_map[key]['original_pred_text'][i] for i in range(l)],
                'rouge':[self.languages_map[key]['rouges'][i] for i in range(l)]
            })
            df_to_write = pd.concat([df_to_write, df_key])
            
        df_to_write.to_csv(method + '_' + domain + '_' + model_name.replace('google/', '').replace('facebook/', '') + '.csv', index=False)

       return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Bart Fine-tuning Parameters')
        parser.add_argument('--learning_rate', default=2e-5, type=float)
        parser.add_argument('--model_name_or_path', default='bart-base', type=str)
        parser.add_argument('--eval_beams', default=4, type=int)
        parser.add_argument('--tgt_max_seq_len', default=128, type=int)
        parser.add_argument('--tokenizer', default='bart-base', type=str)
        return parent_parser

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input parameters for extractive stage')
    parser.add_argument('--batch_size', default=1, type=int, help='test_batch_size')
    parser.add_argument('--train_path', default=None, help='path to input json file for a given domain in given language')
    parser.add_argument('--val_path', default=None, help='path to intermediate output json file for a given domain in given language')
    parser.add_argument('--test_path', default=None, help='path to output json file for a given domain in given language')
    parser.add_argument('--config', default=None, help='which config file to use')
    parser.add_argument('--tokenizer', default='facebook/mbart-large-50', help='which tokenizer to use')
    parser.add_argument('--model', default='facebook/mbart-large-50', help='which model to use')
    parser.add_argument('--target_lang', default='hi_IN', help='what is the target language')
    parser.add_argument('--ckpt_path', help='ckpt path')
    parser.add_argument('--is_mt5', type=int, help='is the model mt5')

    args = parser.parse_args()
    prediction_path = args.prediction_path

    ckpt_path = args.ckpt_path
    ckpt_path_1 = ckpt_path.split('/')[-1]
    method = ckpt_path_1.split('_')[0] + '_' + ckpt_path_1.split('_')[1]
    domain = ckpt_path_1.split('_')[2]
    model_name = ckpt_path_1.split('_')[3].split('-')[0]
    
    print('-----------------------------------------------------------------------------------------------------------')
    print(method, domain, model_name)
   
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    if 'mt5' in model_name:
        tokenizer = 'google/mt5-base'
        model_name = 'google/mt5-base'
        is_mt5 = 1
    else:
        tokenizer = 'facebook/mbart-large-50'
        model_name = 'facebook/mbart-large-50'
        is_mt5 = 0

    dm_hparams = dict(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            tokenizer_name_or_path=tokenizer,
            is_mt5=is_mt5,
            max_source_length=512,
            max_target_length=512,
            train_batch_size=1,
            val_batch_size=1,
            test_batch_size=args.batch_size
            )
    dm = DataModule(**dm_hparams)

    model_hparams = dict(
            learning_rate=2e-5,
            model_name_or_path=model_name,
            eval_beams=4,
            is_mt5=is_mt5,
            tgt_max_seq_len=512,
            tokenizer=dm.tokenizer,
        )

    model = Summarizer(**model_hparams)
    logger=WandbLogger(name='inference_' + method + '_' + domain +  '_' + model_name, save_dir='./', project='multilingual evaluation', log_model=False)
    trainer = pl.Trainer(gpus=1, logger=logger)

    model = model.load_from_checkpoint(ckpt_path)
    results = trainer.test(model=model, datamodule=dm, verbose=True)
    print('-----------------------------------------------------------------------------------------------------------')