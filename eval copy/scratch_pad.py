import json

file_paths = [
    "/home/dan/DeepLearning/mini_temporal/generations/TQE/gemma-1_1-7b-it/no_context_trained/random_context_evaluated/TQE_gemma-1_1-7b-it_no_context_trained_random_context_evaluated.jsonl",
    "/home/dan/DeepLearning/mini_temporal/generations/TQE/gemma-1_1-7b-it/no_context_trained/relevant_context_evaluated/TQE_gemma-1_1-7b-it_no_context_trained_relevant_context_evaluated.jsonl",
    "/home/dan/DeepLearning/mini_temporal/generations/TQE/gemma-1_1-7b-it/no_context_trained/wrong_date_context_evaluated/TQE_gemma-1_1-7b-it_no_context_trained_wrong_date_context_evaluated.jsonl",
    "/home/dan/DeepLearning/mini_temporal/generations/TQE/gemma-1_1-7b-it/no_context_trained/no_context_evaluated/TQE_gemma-1_1-7b-it_no_context_trained_no_context_evaluated.jsonl"
]

for file_path in file_paths:
    try:
        with open(file_path, 'r') as f:
            print(f"Processing file: {file_path}")
            content = f.read()
            if content.strip():
                data = json.loads(content)
                # Assuming 'data' should be a list of dictionaries or similar structure
                print(f"Loaded {len(data)} records from {file_path}")
                # Process data here
                # Add more debug statements as needed to trace processing steps
            else:
                print(f"File {file_path} is empty.")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
