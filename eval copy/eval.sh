#!/usr/bin/env bash

# Define the actual files for each dataset
declare -A actual_files
actual_files["AQA"]="/home/dan/DeepLearning/mini_temporal/data/AQA/final/test.jsonl"
actual_files["TQE"]="/home/dan/DeepLearning/mini_temporal/clean_for_fatemeh/new_data/test.jsonl"

# Define the base directory for predictions
pred_base_dir="/home/dan/DeepLearning/mini_temporal/generations/TQE"

# Empty the results.jsonl file
truncate -s 0 results.jsonl

# Find all prediction files in the directory structure and count them
# echo "Finding prediction files in $pred_base_dir"
find "$pred_base_dir" -type f -name "*.jsonl" | tee found_files.txt
file_count=$(wc -l < found_files.txt)
echo "Found $file_count prediction files"

# Process each prediction file
while read -r pred_file; do
    # Determine the dataset based on the prediction file path
    if [[ "$pred_file" == *"/AQA/"* ]]; then
        dataset="AQA"
    elif [[ "$pred_file" == *"/TQE/"* ]]; then
        dataset="TQE"
    else
        echo "Unknown dataset for file: $pred_file"
        continue
    fi

    actual_file=${actual_files[$dataset]}
    
    echo "Processing Actual: $actual_file, Prediction: $pred_file"
    
    # Run the Python script and check if it produces output
    output=$(python eval.py --pred_file "$pred_file" --actual_file "$actual_file" 2>&1)
    if [[ -n "$output" ]]; then
        echo "$output" >> results.jsonl
        echo "Successfully processed $pred_file"
    else
        echo "No output for $pred_file"
    fi
done < found_files.txt

# Clean up
rm found_files.txt



# #!/usr/bin/env bash

# # Define the actual files for each dataset
# declare -A actual_files
# actual_files["AQA"]="/home/dan/DeepLearning/mini_temporal/data/AQA/final/test.jsonl"
# actual_files["TQE"]="/home/dan/DeepLearning/mini_temporal/clean_for_fatemeh/new_data/test.jsonl"

# # Define the base directory for predictions
# pred_base_dir="/home/dan/DeepLearning/mini_temporal/generations/TQE"

# # Empty the results.jsonl file
# truncate -s 0 results.jsonl

# # Find all prediction files in the directory structure
# find "$pred_base_dir" -type f -name "*.jsonl" | while read -r pred_file; do
#     # Determine the dataset based on the prediction file path
#     if [[ "$pred_file" == *"/AQA/"* ]]; then
#         dataset="AQA"
#     elif [[ "$pred_file" == *"/TQE/"* ]]; then
#         dataset="TQE"
#     else
#         echo "Unknown dataset for file: $pred_file"
#         continue
#     fi

#     actual_file=${actual_files[$dataset]}
    
#     echo "Processing Actual: $actual_file, Prediction: $pred_file"
    
#     # Run the Python script and redirect standard output to results.jsonl, suppressing stderr
#     python eval.py --pred_file "$pred_file" --actual_file "$actual_file" >> results.jsonl 2>/dev/null
# done
