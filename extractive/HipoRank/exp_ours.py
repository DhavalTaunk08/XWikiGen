import sys
from hipo_rank.dataset_iterators.custom_data import CustomDataset
from hipo_rank.dataset_iterators.pubmed import PubmedDataset
import ast
from hipo_rank.embedders.w2v import W2VEmbedder
from hipo_rank.embedders.rand import RandEmbedder
from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.embedders.sent_transformers import SentTransformersEmbedder
from itertools import groupby
from operator import itemgetter
from hipo_rank.similarities.cos import CosSimilarity

from hipo_rank.directions.undirected import Undirected
from hipo_rank.directions.order import OrderBased
from hipo_rank.directions.edge import EdgeBased

from hipo_rank.scorers.add import AddScorer
from hipo_rank.scorers.multiply import MultiplyScorer

from hipo_rank.summarizers.default import DefaultSummarizer
# from hipo_rank.evaluators.rouge import evaluate_rouge

from pathlib import Path
import json
import time
from tqdm import tqdm

DEBUG = False

import gc

# PubMed hyperparameter gridsearch and ablation study

DATASETS = [
    ("data_val",CustomDataset, {"file_path": str(sys.argv[1])})
]


EMBEDDERS = [
    ("mbert", BertEmbedder,
     {"bert_model_path": "bert-base-multilingual-cased",
      "bert_pretrained":"bert-base-multilingual-cased",
      "bert_tokenizer": "bert-base-multilingual-cased"}
     )
]
SIMILARITIES = [
    ("cos", CosSimilarity, {}),
]
DIRECTIONS = [
    ("edge", EdgeBased, {}),
]

SCORERS = [
          	("add_f=0.0_b=1.0_s=0.5", AddScorer, {"section_weight": 0.5}),
          ]


Summarizer = DefaultSummarizer()

experiment_time = int(time.time())
# results_path = Path(f"results/{experiment_time}")
results_path = Path(f"results/exp1")

for embedder_id, embedder, embedder_args in EMBEDDERS:
    Embedder = embedder(**embedder_args)
    for dataset_id, dataset, dataset_args in DATASETS:
        DataSet = dataset(**dataset_args)
        docs = list(DataSet)
        if DEBUG:
            docs = docs[:5]
        print(f"embedding dataset {dataset_id} with {embedder_id}")
        embeds = [Embedder.get_embeddings(doc) for doc in tqdm(docs)]

        docs1=[]
        for similarity_id, similarity, similarity_args in SIMILARITIES:
            Similarity = similarity(**similarity_args)
            print(f"calculating similarities with {similarity_id}")
            sims =[]
            
            
            for i in range(len(embeds)):
                try:
                    temp=Similarity.get_similarities(embeds[i])
                    sims.append(temp)
                    docs1.append(docs[i])
                    gc.collect()
                except Exception as e:
                    print(e)
                    continue
            for direction_id, direction, direction_args in DIRECTIONS:
                print(f"updating directions with {direction_id}")
                Direction = direction(**direction_args)
                sims = [Direction.update_directions(s) for s in sims]
                for scorer_id, scorer, scorer_args in SCORERS:
                    Scorer = scorer(**scorer_args)
                    experiment = f"{dataset_id}-{embedder_id}-{similarity_id}-{direction_id}-{scorer_id}-{experiment_time}"
                    experiment_path = results_path / experiment
                    try:
                        experiment_path.mkdir(parents=True)

                        print("running experiment: ", experiment)
                        results = []
                        references = []
                        summaries = []
                        for sim, doc in zip(sims, docs1):
                            #print('doc:',docs)
                            scores = Scorer.get_scores(sim)
                            summary = Summarizer.get_summary(doc, scores)

                            stage_op = {}
                            stage_op['title'] =doc.page_title
                            stage_op['section_title']=doc.section_title
                            stage_op['references'] = [s[0] for s in summary]
                            stage_op['content']=doc.content

                            stage_op_path=str(experiment_path)+"/"+str(sys.argv[2])+"_summary_op.json"
                            with open(stage_op_path,'a') as f:
                                f.write(
                                json.dumps(stage_op, ensure_ascii=False)+'\n')

                            results.append({
                                "num_sects": len(doc.sections),
                                "num_sents": sum([len(s.sentences) for s in doc.sections]),
                                "summary": summary,

                            })
                            summaries.append([s[0] for s in summary])
                            references.append([doc.reference])
                        # rouge_result = evaluate_rouge(summaries, references)
                        # (experiment_path / "rouge_results.json").write_text(json.dumps(rouge_result, indent=4, ensure_ascii=False))
                        (experiment_path / "summaries.json").write_text(json.dumps(results, indent=4, ensure_ascii=False))
                        
                        save_name=str(sys.argv[2])
                        stage_op_path=str(experiment_path)+"/"+str(sys.argv[2])+"summary_op.json"
                        op=[]
                        with open(stage_op_path,'r') as f:
                            pagelist=f.readlines()
                            pagelist=[ast.literal_eval(x) for x in pagelist]
                            for key, value in tqdm(groupby(pagelist,key = itemgetter('title'))):
                                temp={}
                                temp['page_title']=key
                                temp['sections']=[]

                                for k in value:
                                    temp['sections'].append(k)
                                
                                op.append(temp)

                        with open(stage_op_path,'w') as fw:
                            for j in op:
                                fw.write(json.dumps(j,ensure_ascii=False)+'\n')
                            
                    except Exception as e:
                        print(e)
                        pass


