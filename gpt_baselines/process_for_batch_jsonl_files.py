"""
Created on 06/04/2024

@author: Dan Schumacher
"""
#endregion
#region # IMPORTS AND SET-UP
# =============================================================================
# IMPORTS AND SET-UP
# =============================================================================
import json
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="BEM Score Evaluation Script")
    parser.add_argument('--context', type=str, required=True, choices=['no_context','relevant_context','wrong_date_context','random_context'], help='What type of context will you be evaluating the model on?')
    return parser.parse_args()

#endregion
#region # FUNCTIONS
# =============================================================================
# FUNCTIONS
# =============================================================================
def format_instruction_test(df, context):

    tokenized_instructions = []
 
    if context != 'no_context':
        # print('TOP')
        for question, context in zip(df['question'], df[context]):

            message = f'{question} Here is the context: {context} The answer is: '

            # apply eos to end of the string
            tokenized_instructions.append(message)

    else:
        # print('BOTTOM')
        for question in df['question']:
            
            message = f'{question} The answer is: '

            # apply eos to end of the string
            tokenized_instructions.append(message)

    return tokenized_instructions

def convert_to_jsonl(df, output_file, system_content):
    """
    Convert a DataFrame to JSONL format for batch processing with OpenAI.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the user content.
    output_file (str): The output file path.
    system_content (str): The system content for all requests.
    """
    with open(output_file, 'w') as f:
        for i, row in df.iterrows():
            json_object = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo-0125",
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": row['user_content']}
                    ],
                    "max_tokens": 1000
                }
            }

            f.write(json.dumps(json_object) + '\n')

# Example DataFrame


#endregion
#region # MAIN
# =============================================================================
# MAIN
# =============================================================================

def main(): # @$@
    args = parse_args() # @$@
    df = pd.read_json('/home/dan/DeepLearning/mini_temporal/data/AQA/test.jsonl', lines=True, orient='records')

    context = args.context # @$@ COMMANDLINE ARG
    # context = 'relevant_context' # @$@

    questions = pd.DataFrame({
    'user_content':format_instruction_test(df, context)
    })

    # System content
    system_content = "You are a helpful assistant. Answer the question to the best of your ability."

    # Output file
    output_file = f'prompts/{args.context}_prompt.jsonl' # @$@ COMMANDLINE ARG

    # Convert DataFrame to JSONL
    convert_to_jsonl(questions, output_file, system_content)

    print(f"Data has been successfully written to {output_file}")

if __name__ == "__main__": # @$@
    main()
