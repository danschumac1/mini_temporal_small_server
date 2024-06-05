#!/bin/bash
contexts=("no_context" "random_context" "wrong_date_context" "relevant_context")

for context in "${contexts[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python process_for_batch_jsonl_files.py \
    --context "${context}"
    echo "finished ${context}"
done
