#!/usr/bin/env bash

# set up log file
log_file="./tqe_eval_logging.txt"
echo "Generation started at $(date)" > ${log_file}

# Define the actual files for each dataset
actual_df="/home/dan/DeepLearning/mini_temporal/clean_for_fatemeh/new_data/test.jsonl"
actual_list="./eval_data1/tqe_actual.jsonl"

# Where do all the jsonl files live? 
pred_base_dir="/home/dan/DeepLearning/mini_temporal/generations/mixed"

# Empty the results.jsonl file
results_file="tqe_mixed_results.jsonl"
truncate -s 0 $results_file

# Find all prediction files in the directory structure and count them
file_count=$(find "$pred_base_dir" -type f -name "*.jsonl" | tee found_files.txt | wc -l)
echo "Found $file_count prediction files" >> ${log_file}

# Process each prediction file
while read -r pred_file; do
    
    echo "Processing: $pred_file" >> ${log_file}
    
    # Run the Python script and check if it produces output
    CUDA_VISIBLE_DEVICES=1 output=$(python eval.py --pred_file "$pred_file" --actual_list "$actual_list" --actual_df "$actual_df" 2>> ${log_file})

    # CHECK IF OUTPUT IS EMPTY OR NOT
    if [[ -n "$output" ]]; then
        # ECHO THE OUTPUT INTO THE RESULTS FILE
        echo "$output" >> $results_file
        echo "Successfully processed $pred_file" >> ${log_file}
    else
        # PRINT AN ERROR MESSAGE IF NO OUTPUT IS PRODUCED
        echo " !!!!! | !!!!! No output for $pred_file" >> ${log_file}
    fi
done < found_files.txt

# Clean up the temporary file
rm found_files.txt

echo "Generation script ended at $(date)" >> ${log_file}