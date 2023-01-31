from indicnlp.tokenize import sentence_tokenize
import ast
import sys
import re
from tqdm import tqdm
import json

lang='en'

from itertools import cycle, islice

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)  # .next on Python 2
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

def article_text(doc):
    doc_list=[]
    round_list = []
    for ref in doc['references']:
        temp = sentence_tokenize.sentence_split(ref, lang=lang)
        round_list.append(temp)

    op_list=list(roundrobin(*round_list))

    max_sentences=1700   #modify this value for extracting top n sentences across all refs

    for i in op_list:
        if len(doc_list)<max_sentences:
            doc_list.append(i)


        # for j in temp:
        #     if len(doc_list)<=1000:
        #         doc_list.append(j)
    # print('doc_list:',doc_list)


    return doc_list


def section_text(doc):
    doc_list = []
    inside_list = []
    for ref in doc['references']:
        temp_li = sentence_tokenize.sentence_split(ref, lang=lang)
        for j in temp_li:
            if len(inside_list)<1000:
                inside_list.append(j)

        doc_list.append(inside_list)

    return doc_list


def abstract_text(doc):
    abs = []
    temp = sentence_tokenize.sentence_split(doc['content'], lang=lang)
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

        for i in tqdm(range(len(docs)),desc = "Converting our data to the format format:"):
            op_dict = {}
            op_dict['page_title'] = docs[i]['title']

            for k in range(len(docs[i]['sections'])):

                # print('docs[i][sections]:', docs[i]['sections'])

                op_dict['abstract_text']=abstract_text(docs[i]['sections'][k])
                op_dict['article_text'] = article_text(docs[i]['sections'][k])
                op_dict['section_names'] = [docs[i]['sections'][k]['title']]

                op_dict['sections']=section_text(docs[i]['sections'][k])
                op_dict['content']=docs[i]['sections'][k]['content']

                if len(op_dict['article_text'])==0:
                    continue


                try:
                    output.write(json.dumps(op_dict, ensure_ascii=False))
                    output.write('\n')

                except Exception as e:
                    print(e)
                    continue

    output.close()




if __name__=="__main__":
    main()

