
from hipo_rank.dataset_iterators.custom_data import CustomDataset
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
from hipo_rank.evaluators.rouge import evaluate_rouge

from pathlib import Path
import json
import time
from tqdm import tqdm

DEBUG = False

# PubMed hyperparameter gridsearch and ablation study

DATASETS = [
    ("data_val", CustomDataset, {"file_path": "test_conv.txt"})
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
        for similarity_id, similarity, similarity_args in SIMILARITIES:
            Similarity = similarity(**similarity_args)
            print(f"calculating similarities with {similarity_id}")
            sims = [Similarity.get_similarities(e) for e in embeds]
            for direction_id, direction, direction_args in DIRECTIONS:
                print(f"updating directions with {direction_id}")
                Direction = direction(**direction_args)
                sims = [Direction.update_directions(s) for s in sims]
                for scorer_id, scorer, scorer_args in SCORERS:
                    Scorer = scorer(**scorer_args)
                    experiment = f"{dataset_id}-{embedder_id}-{similarity_id}-{direction_id}-{scorer_id}"
                    experiment_path = results_path / experiment
                    try:
                        experiment_path.mkdir(parents=True)

                        print("running experiment: ", experiment)
                        results = []
                        references = []
                        summaries = []
                        for sim, doc in zip(sims, docs):
                            #print('doc:',docs)
                            scores = Scorer.get_scores(sim)
                            summary = Summarizer.get_summary(doc, scores)

                            stage_op = {}
                            stage_op['title'] =doc.page_title
                            stage_op['section_title']=doc.section_title
                            stage_op['content']=[s[0] for s in summary]
                            stage_op_path=str(experiment_path)+"/summary_op.json"
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
						print('writing outputs in summary_op.json')
                        stage_op_path=str(experiment_path)+"/summary_op.json"
                        with open(stage_op_path,'r') as f:
                            pagelist=f.readlines()
                            pagelist=[ast.literal_eval(x) for x in pagelist]
                            for key, value in tqdm(groupby(pagelist,key = itemgetter('title'))):
                                temp={}
                                temp['page_title']=key
                                temp['sections']=[]
                                for k in value:
                                    temp['sections'].append(k)
                                with open(stage_op_path,'a') as fw:
                                    fw.write(json.dumps(temp,ensure_ascii=False))
                            
                    except Exception as e:
                        print(e)
                        pass

