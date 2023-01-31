## Steps to run code:

Please Install the following libraries:
- transformers
- sentence_transformers
- summa
- gensim
- indic-nlp-library

To convert the wiki scraped data into the requisite format:<br>
python3 dataset_format_sentence_tokenization_individual_sectionwise.py <input_file> <output_file>  <br>

To run the experiment, run the following command:<br>

python3 exp_ours.py <inp_file.txt> <prefix_for_output>
