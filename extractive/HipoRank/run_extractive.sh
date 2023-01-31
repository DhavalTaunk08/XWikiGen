#!/usr/bin/ bash

python dataset_format_sentence_tokenization_individual_sectionwise.py zen_scraped_data/bn/bn_books.json output_data/bn/books.txt
python exp_ours.py output_data/bn/books.txt books bn


python dataset_format_sentence_tokenization_individual_sectionwise.py zen_scraped_data/bn/bn_films.json output_data/bn/films.txt
python exp_ours.py output_data/bn/films.txt films bn


python dataset_format_sentence_tokenization_individual_sectionwise.py zen_scraped_data/bn/bn_politicians.json output_data/bn/politicians.txt
python exp_ours.py output_data/bn/politicians.txt politicians bn


python dataset_format_sentence_tokenization_individual_sectionwise.py zen_scraped_data/bn/bn_sportsman.json output_data/bn/sportsman.txt
python exp_ours.py output_data/bn/sportsman.txt sportsman bn


python dataset_format_sentence_tokenization_individual_sectionwise.py zen_scraped_data/bn/bn_writers.json output_data/bn/writers.txt
python exp_ours.py output_data/bn/writers.txt writers bn
