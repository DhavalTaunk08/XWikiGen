#!/bin/bash

echo 'books'
python extractive.py --inp_file extractive/fr/fr_books.json --out_file extractive/output_data/fr/books/books_extractive.json --top_k 100 --tokenizer xlm-roberta-base --model xlm-roberta-base

echo 'films'
python extractive.py --inp_file extractive/fr/fr_films.json --out_file extractive/output_data/fr/films/films_extractive.json --top_k 100 --tokenizer xlm-roberta-base --model xlm-roberta-base

echo 'politicians'
python extractive.py --inp_file extractive/fr/fr_politicians.json --out_file extractive/output_data/fr/politicians/politicians_extractive.json --top_k 100 --tokenizer xlm-roberta-base --model xlm-roberta-base

echo 'sportsman'
python extractive.py --inp_file extractive/fr/fr_sportsman.json --out_file extractive/output_data/fr/sportsman/sportsman_extractive.json --top_k 100 --tokenizer xlm-roberta-base --model xlm-roberta-base

echo 'writers'
python extractive.py --inp_file extractive/fr/fr_writers.json --out_file extractive/output_data/fr/writers/writers_extractive.json --top_k 100 --tokenizer xlm-roberta-base --model xlm-roberta-base