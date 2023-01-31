#!/bin/bash

#For hiporank mbart experiments
mkdir -p hiporank/mbart_predictions/
mkdir -p hiporank/mbart/ckpt
# Copy all the checkpoints related to hiporank mbart experiments to the below mentioned directory
# cp -r *_hiporank_mbart.ckpt hiporank/mbart/ckpt/
dir="hiporank/mbart/ckpt"
for f in "$dir"/*; do
   python testing.py --ckpt_path $f
done


# For salience mbart experiments
mkdir -p salience/mbart_predictions/
mkdir -p salience/mbart/ckpt
# Copy all the checkpoints related to salience mbart experiments to the below mentioned directory
# cp -r *_salience_mbart.ckpt salience/mbart/ckpt/
dir="salience/mbart/ckpt"
for f in "$dir"/*; do
   python testing.py --ckpt_path $f
done