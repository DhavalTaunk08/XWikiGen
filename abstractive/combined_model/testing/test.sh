#!/bin/bash	

python testing_all/testing.py --ckpt_path salience_multilingual_multidomain_mbart-summ.ckpt --train_path multi_domain_lang_data/salience_output_data/train.json --val_path multi_domain_lang_data/salience_output_data/val.json --test_path multi_domain_lang_data/salience_output_data/test.json

python testing_all/testing.py --ckpt_path salience_multilingual_multidomain_mt5-summ.ckpt --train_path multi_domain_lang_data/salience_output_data/train.json --val_path multi_domain_lang_data/salience_output_data/val.json --test_path multi_domain_lang_data/salience_output_data/test.json

python testing_all/testing.py --ckpt_path hiporank_multilingual_multidomain_mbart-summ.ckpt --train_path multi_domain_lang_data/hiporank_output_data/train.json --val_path multi_domain_lang_data/hiporank_output_data/val.json --test_path multi_domain_lang_data/hiporank_output_data/test.json

python testing_all/testing.py --ckpt_path hiporank_multilingual_multidomain_mbart-summ.ckpt --train_path multi_domain_lang_data/hiporank_output_data/train.json --val_path multi_domain_lang_data/hiporank_output_data/val.json --test_path multi_domain_lang_data/hiporank_output_data/test.json