#!/usr/bin/env bash



CUDA_VISIBLE_DEVICES=1 python eval.py --pred_file /home/dan/DeepLearning/mini_temporal/generations/output/TQE/baseline/gemma-7b/relevant_context/TQE_baseline_gemma-7b_relevant_context_evaluated.jsonl --actual_file "/home/dan/DeepLearning/mini_temporal/clean_for_fatemeh/new_data/test.jsonl" > minify_results.jsonl


# Define the actual files for each dataset
declare -A actual_files
actual_files["AQA"]="/home/dan/DeepLearning/mini_temporal/data/AQA/final/test.jsonl"
actual_files["TQE"]="/home/dan/DeepLearning/mini_temporal/clean_for_fatemeh/new_data/test.jsonl"

# set up log file
log_file="./logging.txt"
echo "Generation started at $(date)" > ${log_file}

# Define the base directory for predictions
pred_base_dir="/home/dan/DeepLearning/mini_temporal/generations/TQE_NIT"

# Empty the results.jsonl file
truncate -s 0 nit_results.jsonl

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
    CUDA_VISIBLE_DEVICES=1 output=$(python eval_nit.py --pred_file "$pred_file" --actual_file "$actual_file" 2>> ${log_file})

    # CHECK IF OUTPUT IS EMPTY OR NOT
    if [[ -n "$output" ]]; then
        # ECHO THE OUTPUT INTO THE RESULTS FILE
        echo "$output" >> nit_results.jsonl
        echo "Successfully processed $pred_file" >> ${log_file}
    else
        # PRINT AN ERROR MESSAGE IF NO OUTPUT IS PRODUCED
        echo " !!!!! | !!!!! No output for $pred_file" >> ${log_file}
    fi
done < found_files.txt

# Clean up the temporary file
rm found_files.txt

echo "Generation script ended at $(date)" >> ${log_file}
