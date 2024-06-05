#!/usr/bin/env bash

# Define the actual files for each dataset
declare -A actual_files
actual_files["AQA"]="/home/dan/DeepLearning/mini_temporal/data/AQA/final/test.jsonl"
actual_files["TQE"]="/home/dan/DeepLearning/mini_temporal/clean_for_fatemeh/new_data/test.jsonl"

# set up log file
log_file="./it_logging.txt"
echo "Generation started at $(date)" > ${log_file}

# Define the base directory for predictions
pred_base_dir="/home/dan/DeepLearning/mini_temporal/generations/TQE_IT"

# Empty the results.jsonl file
truncate -s 0 it_results.jsonl

# Find all prediction files in the directory structure and count them
file_count=$(find "$pred_base_dir" -type f -name "*.jsonl" | tee found_files.txt | wc -l)
echo "Found $file_count prediction files" >> ${log_file}

# Process each prediction file
while read -r pred_file; do
    # Determine the dataset based on the prediction file path
    if [[ "$pred_file" == *"AQA"* ]]; then
        dataset="AQA"
    elif [[ "$pred_file" == *"TQE"* ]]; then
        dataset="TQE"
    else
        echo "Unknown dataset for file: $pred_file" >> ${log_file}
        continue
    fi

    actual_file=${actual_files[$dataset]}
    
    echo "Processing Actual: $actual_file, Prediction: $pred_file" >> ${log_file}
    
    # Run the Python script and check if it produces output
    CUDA_VISIBLE_DEVICES=1 output=$(python eval_it.py --pred_file "$pred_file" --actual_file "$actual_file" 2>> ${log_file})

    # CHECK IF OUTPUT IS EMPTY OR NOT
    if [[ -n "$output" ]]; then
        # ECHO THE OUTPUT INTO THE RESULTS FILE
        echo "$output" >> it_results.jsonl
        echo "Successfully processed $pred_file" >> ${log_file}
    else
        # PRINT AN ERROR MESSAGE IF NO OUTPUT IS PRODUCED
        echo " !!!!! | !!!!! No output for $pred_file" >> ${log_file}
    fi
done < found_files.txt

# Clean up the temporary file
rm found_files.txt

echo "Generation script ended at $(date)" >> ${log_file}
