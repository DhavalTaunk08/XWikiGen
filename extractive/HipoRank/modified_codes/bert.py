import numpy as np
import torch
from transformers import BertModel, BertConfig, BertTokenizer,AutoTokenizer
from transformers import AutoModel
from numpy import ndarray
from hipo_rank import Document, Embeddings, SectionEmbedding, SentenceEmbeddings
from typing import List


class BertEmbedder:
    def __init__(self, bert_model_path: str,
                 bert_tokenizer: str = "bert-base-multilingual-cased",
                 bert_pretrained: str = None,
                 max_seq_len: int = 60,
                 cuda: bool = True):
        self.max_seq_len = max_seq_len
        self.cuda = cuda
        if bert_pretrained:
            self.bert_model = AutoModel.from_pretrained(bert_pretrained)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_pretrained)
            if cuda:
                self.bert_model.cuda()
                self.bert_model.eval()

        else:
            self.bert_model = self._load_bert(bert_model_path)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer)

    def _load_bert(self, bert_model_path: str):
        model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model.to(device)
        return model

    def _get_sentences_embedding(self, sentences: List[str]) -> ndarray:
        # TODO: clean up batch approach
        input_ids = [self.bert_tokenizer.encode(s, add_special_tokens=True) for s in sentences]
        padded_len = min(max([len(x) for x in input_ids]), self.max_seq_len)
        num_inputs = len(input_ids)
        input_tensor = np.zeros((num_inputs, padded_len))
        for i, x in enumerate(input_ids):
            l = min(padded_len, len(x))
            input_tensor[i][:l] = x[:l]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cuda':
            input_tensor = torch.LongTensor(input_tensor).to(device)
        else:
            input_tensor = torch.LongTensor(input_tensor)
        batch_size = 20
        pooled_outputs = []
        for i in range(0, num_inputs, batch_size):
            input_batch = input_tensor[i:i + batch_size]
            # Original pacsum paper uses [CLS] next sentence prediction activations
            # this isn't optimal and should be changed for potentially better performance
            with torch.no_grad():
                pooled_output = self.bert_model(input_batch)[1]  # shape = (x, 768)
            if self.cuda:
                pooled_output = pooled_output.cpu()
            else:
                pooled_output = pooled_output
            pooled_outputs.append(pooled_output)
        pooled_outputs = torch.cat(pooled_outputs).numpy()
        return pooled_outputs

    def get_embeddings(self, doc: Document) -> Embeddings:
        sentence_embeddings = []
        for section in doc.sections:
            id = section.id
            sentences = section.sentences
            se = self._get_sentences_embedding(sentences)
            sentence_embeddings += [SentenceEmbeddings(id=id, embeddings=se)]
        section_embeddings = [SectionEmbedding(id=se.id, embedding=np.mean(se.embeddings, axis=0))
                              for se in sentence_embeddings]

        return Embeddings(sentence=sentence_embeddings, section=section_embeddings)




