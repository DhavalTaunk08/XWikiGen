#!/bin/bash


mkdir mbart_predictions
dir="ckpt"
for f in "$dir"/*; do
	echo $f
   python testing.py --ckpt_path $f
done