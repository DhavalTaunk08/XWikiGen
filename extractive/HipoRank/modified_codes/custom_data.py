from collections import Counter
from pathlib import Path
import json

from typing import List, Iterator, Any, Optional
from dataclasses import dataclass
from hipo_rank import Document, Section


@dataclass
class CustomDoc:
    # dataclass wrapper for original pubmed dataset format
    # article_id: str
    article_text: List[str]
    abstract_text: List[str]
    # labels: Any
    section_names: List[str]
    sections: List[List[str]]
    page_title:str


class CustomDataset(object):
    def __init__(self,
                 file_path,
                 no_sections: bool = False,
                 min_words: Optional[int] = None,
                 max_words: Optional[int] = None,
                 min_sent_len: int = 1 # min num of alphabetical words
                 ):
        self._file_path = file_path
        self.no_sections = no_sections
        self.min_sent_len = min_sent_len
        docs = [CustomDoc(**json.loads(l)) for l in Path(self._file_path).read_text().split("\n") if l]
        docs = [d for d in docs if not all([s == [''] for s in d.sections])]
        if min_words or max_words:
            docs = self._filter_doc_len(docs, min_words, max_words)
        self.docs = docs

    def _filter_doc_len(self, docs: List[CustomDoc], min_words: int, max_words: int):
        def f(doc: CustomDoc):
            l = sum([len(s.split()) for s in doc.article_text])
            if min_words:
                if l < min_words:
                    return False
            if max_words:
                if l >= max_words:
                    return False
            return True
        return list(filter(f, docs))

    def _get_sections(self, doc: CustomDoc) -> List[Section]:
        if self.no_sections:
            sentences = sum([s for s in doc.sections if s != ['']], [])
            return [Section(id="no_sections", sentences=sentences)]
        # handles edge case where sections have the same name
        section_names = []
        sn_counter = Counter(doc.section_names)
        for sn in reversed(doc.section_names):
            c = sn_counter[sn]
            if c > 1:
                section_names = [f"{sn}_{c}"] + section_names
                sn_counter[sn] -= 1
            else:
                section_names = [sn] + section_names
        sentences = [[s for s in sents if len([w for w in s.split() if w.isalpha()]) >= self.min_sent_len] for sents in doc.sections]
        sections = [Section(id=n, sentences=s) for n, s in zip(section_names, sentences) if s]
        return sections

    def _get_reference(self, doc: CustomDoc) -> List[str]:
        # remove sentence tags in abstract which break rouge
        return [s.replace("<S>", "").replace("<S\>", "").replace("</S>", "") for s in doc.abstract_text]

    def __iter__(self) -> Iterator[Document]:
        for doc in self.docs:
            sections = self._get_sections(doc)
            reference = self._get_reference(doc)
            yield Document(sections=sections, reference=reference,page_title=doc.page_title)

    def __getitem__(self, i):
        if isinstance(i, int):
            doc = self.docs[i]
            sections = self._get_sections(doc)
            reference = self._get_reference(doc)
            return Document(sections=sections, reference=reference, page_title=doc.page_title)
        elif isinstance(i, slice):
            docs = self.docs[i]
            sections = [self._get_sections(doc) for doc in docs]
            references = [self._get_reference(doc) for doc in docs]
            return [Document(sections=s, reference=r,page_title=doc.page_title) for s,r in zip(sections, references)]
