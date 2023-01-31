from indicnlp.tokenize import sentence_tokenize
import ast
import sys
import re
from tqdm import tqdm
import json

lang='en'

def article_text(doc):
    doc_list = []

    for i in doc['sections']:
        for ref in i['references']:
            temp = sentence_tokenize.sentence_split(ref, lang=lang)
            for j in temp:
                doc_list.append(j)
    # print('doc_list:',doc_list)
    return doc_list


def section_text(doc):
    doc_list = []
    for i in doc['sections']:
        inside_list = []
        for ref in i['references']:
            temp_li = sentence_tokenize.sentence_split(ref, lang=lang)
            for j in temp_li:
                inside_list.append(j)

        doc_list.append(inside_list)

    return doc_list


def abstract_text(doc):
    abs = []
    for i in doc['sections']:
        temp = sentence_tokenize.sentence_split(i['content'], lang=lang)
        for j in temp:
            abs.append(j)
    return abs

def section_names(doc):
    section_names = [i['title'] for i in doc['sections']]
    section_names = [s.lower().strip() for s in section_names]

    return section_names

def main():
    output=open(str(sys.argv[2]),'w')

    with open(str(sys.argv[1]),'r') as f:
        docs=f.readlines()
        for j in range(len(docs)):
            docs[j]= ast.literal_eval(docs[j])

        for i in tqdm(range(len(docs)),desc = "Converting our data to the pubmed-like format:"):
            op_dict = {}
            op_dict['abstract_text']=abstract_text(docs[i])
            op_dict['article_text'] = article_text(docs[i])
            op_dict['section_names'] = section_names(docs[i])
            op_dict['sections'] = section_text(docs[i])

            try:
                output.write(json.dumps(op_dict, ensure_ascii=False))
                output.write('\n')

            except Exception as e:
                print(e)
                continue

    output.close()




if __name__=="__main__":
    main()
