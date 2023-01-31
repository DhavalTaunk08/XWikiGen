import sys
import json
import regex
import argparse
from tqdm import tqdm

from polyglot.text import Text
from transformers import AutoTokenizer, AutoModelForMaskedLM
from indicnlp.tokenize.sentence_tokenize import sentence_split
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

class AutoModelForMaskedLMwithLoss(AutoModelForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
        return outputs

class Extractive:
    def __init__(self, tokenizer, model):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForMaskedLMwithLoss.from_pretrained(model).cuda().eval()

    def convert_data(self, inp_file, int_out_file):
        fp = open(int_out_file, 'w')
        with open(inp_file, 'r') as f:
            lines = f.readlines()
            
            for line in tqdm(lines):
                data_to_write = {}
                line = json.loads(line)
                data_to_write['title'] = line['title']
                sections = line['sections']

                for section in sections:
                    if len(section['references'])>0:
                        refs = [t.replace('\"', "'") for t in section['references']]
                        data_to_write['section_title'] = section['title']
                        data_to_write['section_content'] = section['content']
                        data_to_write['section_refs'] = refs

                        fp.write(json.dumps(data_to_write, ensure_ascii=False))
                        fp.write('\n')
        fp.close()

    def get_scores(self, splitted_sents, title, section_title):
        scores = {}
        for sent in splitted_sents:
            st = self.tokenizer(section_title + ' ' + sent, return_tensors='pt', padding='max_length', truncation=True)
            input_ids, attention_mask = st['input_ids'].cuda(), st['attention_mask'].cuda()
            res = self.model(input_ids, attention_mask, labels=input_ids)
            scores[sent] = -res[0].detach().cpu().numpy()

        return scores

    def get_top_k_sentences(self, scores_dict, k):
        scores_dict = sorted(scores_dict.items(), key = lambda x:-x[1])[:k]
        return [tup[0] for tup in scores_dict]

    def remove_bad_chars(self, text):
        RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")
        return RE_BAD_CHARS.sub("", text)
    
    def perform_extractive_stage(self, inp_file, int_out_file, out_file, k):
        print('==========Extracting Top K sentences==========')
        fp = open(out_file, 'w')
        with open(inp_file, 'r') as f:
            lines = f.readlines()
            
            for line in tqdm(lines):
                line = json.loads(line)
                title = line['title']
                for section in line['sections']:
                    section_title = section['title']
                    section_content = section['content']
                    refs = section['references']

                    if len(refs)>0:
                        splitted_sents = []
                        for ref in refs:
                            if ref != '':
                                ref = self.remove_bad_chars(ref)
                                # lang = Text(ref).language.code
                                try:
                                    lang = detect(ref)
                                    splitted_sents.extend(sentence_split(ref, lang=lang))
                                except:
                                    pass

                        if len(splitted_sents)>0:
                            scores = self.get_scores(splitted_sents, title, section_title)
                            top_k_sentences = self.get_top_k_sentences(scores, k)

                            fp.write(json.dumps({
                                'page_title':title,
                                'section_title':section_title,
                                'content':section_content,
                                'references':top_k_sentences
                                }, ensure_ascii=False))
                            fp.write('\n')
        fp.close()
        print('==========Done==========')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input parameters for extractive stage')
    parser.add_argument('--inp_file', default=None, help='path to input json file for a given domain in given language')
    parser.add_argument('--int_out_file', default=None, help='path to intermediate output json file for a given domain in given language')
    parser.add_argument('--out_file', default=None, help='path to output json file for a given domain in given language')
    parser.add_argument('--tokenizer', default='xlm-roberta-base', help='which tokenizer to use')
    parser.add_argument('--model', default='xlm-roberta-base', help='which model to use')
    parser.add_argument('--top_k', default=100, type=int, help='how many sentences to pick')

    args = parser.parse_args()

    inp_file = args.inp_file
    int_out_file = args.int_out_file
    out_file = args.out_file
    k = args.top_k

    tokenizer = args.tokenizer
    model = args.model

    extractor = Extractive(tokenizer, model)

    extractor.perform_extractive_stage(inp_file, int_out_file, out_file, k)
    