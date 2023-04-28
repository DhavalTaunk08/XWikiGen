**XWikiGen**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/xwikigen-cross-lingual-summarization-for/cross-lingual-abstractive-summarization-on-4)](https://paperswithcode.com/sota/cross-lingual-abstractive-summarization-on-4?p=xwikigen-cross-lingual-summarization-for)

This repository contains code related to various experiments, which we performed on our dataset (XWikiRef).

Updated dataset link: [XWikiGen])(https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhaval_taunk_research_iiit_ac_in/EVzrMlk7-UFMr6iJZGYU7H0Bd8TzlsY0vIAGZTOKlLqRcA?e=POh165)

Overall it contains 3 directories:

	1. extractive:
		- Salience: Experiments containing salience extractive stage
		- HipoRank: Experiments containing hiporank extractive stage
	2. abstractive:
		- combined_model: Experiments containing combined model in abstractive stage
		- multidomain: Experiments containing multidomain model in abstractive stage
		- multilingual: Experiments containing multilingual model in abstractive stage
    3. evaluation:
        - evaluate_multidomain: Evaluation script for multidomain experiment
        - evaluate_multilingual: Evaluation script for multilingual experiment
        - evaluate_multilingual_multidomain: Evaluation script for multilingual multidomain experiment

The command to run the above experiments are mentioned in the bash file present in each of the directories mentioned above.

```
    conda create -n xwikigen python=3.8
    conda activate xikigen
    cd XWikiGen/
    pip install -r requirements.txt
```

To run the salience extractive stage:

```
    cd extractive/salience/
    bash run_extractive.sh
```

To run the hiporank extractive stage:

```
    cd extractive/hiporank/
    bash run_extractive.sh
```

To run the salience abstractive stage:

```
    cd abstractive/
    # Go to the desired expriment directory
    bash salience_run.sh
```

To run the hiporank abstractive stage:

```
    cd abstractive/
    # Go to the desired expriment directory
    bash hiporank_run.sh
```

***Note**: Make sure you make changes to the files path as per your machine.*

Below is the directory structure of this repo.

```
├── extractive
│   ├── HipoRank
│   │   ├── modified_codes
│   │   ├── exp8_run.py
│   │   ├── exp5_run.py
│   │   ├── exp10_run.py
│   │   ├── human_eval_data.jsonl
│   │   ├── ROUGE-1.5.5
│   │   ├── exp2_run.py
│   │   ├── run_extractive.sh
│   │   ├── readme.txt
│   │   ├── exp3_run.py
│   │   ├── .gitignore
│   │   ├── exp6_run.py
│   │   ├── exp4_run.py
│   │   ├── exp9_run.py
│   │   ├── human_eval_sample.ipynb
│   │   ├── plot_ablation.ipynb
│   │   ├── exp11_run.py
│   │   ├── convert_to_pubmed_like.py
│   │   ├── LICENSE
│   │   ├── human_eval_samples.jsonl
│   │   ├── plot_sentence_positions.ipynb
│   │   ├── hipo_rank
│   │   ├── human_eval_results.ipynb
│   │   ├── test.txt
│   │   ├── exp7_run.py
│   │   ├── op_indiv2.txt
│   │   ├── exp_ours.py
│   │   └── dataset_format_sentence_tokenization_individual_sectionwise.py
│   └── salience
│       ├── run_extractive.sh
│       └── extractive.py
├── evaluation
│   ├── evaludate_multidomain.py
│   ├── evaluate_multilingual.py
│   └── evaluate_multilingual_multidomain.py
├── requirements.txt
├── readme.md
└── abstractive
    ├── multilingual
    │   ├── hiporank_run.sh
    │   ├── readme.txt
    │   ├── model
    │   │   ├── dataloader.py
    │   │   └── model.py
    │   ├── saliency_run.sh
    │   ├── testing
    │   │   ├── testing.py
    │   │   └── test.sh
    │   └── train.py
    ├── multidomain
    │   ├── hiporank_run.sh
    │   ├── readme.txt
    │   ├── model
    │   │   ├── dataloader.py
    │   │   └── model.py
    │   ├── saliency_run.sh
    │   ├── testing
    │   │   ├── testing.py
    │   │   └── test.sh
    │   └── train.py
    └── combined_model
        ├── hiporank_run.sh
        ├── model
        │   ├── dataloader.py
        │   └── model.py
        ├── saliency_run.sh
        ├── testing
        │   ├── testing.py
        │   └── test.sh
        └── train.py
```
