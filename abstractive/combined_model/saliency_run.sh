#!/bin/bash


# mbart
for domain in books  films  politicians  sportsman  writers; do rm -rf salience_${domain}_mbart_multilingual_multidomain.log; python train.py --train_path multilingual_multidomain_data/salience_output_data/${domain}/${domain}_train.json --val_path multilingual_multidomain_data/salience_output_data/${domain}/${domain}_val.json --test_path multilingual_multidomain_data/salience_output_data/${domain}/${domain}_test.json --tokenizer facebook/mbart-large-50 --model facebook/mbart-large-50 --is_mt5 0 --exp_name all_multilingual_multidomain_salience_${domain}_mbart-summ --save_dir ./ --num_epochs 20 --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 --max_source_length 512 --max_target_length 512 --n_gpus 4 --strategy ddp --sanity_run no  2>&1|tee -a salience_${domain}_mbart_multilingual_multidomain.log;done

# mt5
for domain in books  films  politicians  sportsman  writers; do rm -rf salience_${domain}_mt5_multilingual_multidomain.log; python train.py --train_path multilingual_multidomain_data/salience_output_data/${domain}/${domain}_train.json --val_path multilingual_multidomain_data/salience_output_data/${domain}/${domain}_val.json --test_path multilingual_multidomain_data/salience_output_data/${domain}/${domain}_test.json --tokenizer google/mt5-base --model google/mt5-base --is_mt5 1 --exp_name all_multilingual_multidomain_salience_${domain}_mt5-summ --save_dir ./ --num_epochs 20 --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 --max_source_length 512 --max_target_length 512 --n_gpus 4 --strategy ddp --sanity_run no  2>&1|tee -a salience_${domain}_mt5_multilingual_multidomain.log;done

