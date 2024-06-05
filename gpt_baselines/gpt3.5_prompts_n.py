"""
Created on 06/03/2024

@author: Dan Schumacher
"""

import openai
from openai import OpenAI
from dotenv import load_dotenv
import json
import argparse
import os
import pandas as pd
os.chdir('/home/dan/DeepLearning/mini_temporal')

# =============================================================================
# ARGPARSE
# =============================================================================
# def parse_args():
#     parser = argparse.ArgumentParser(description="OpenAI API interaction script")
#     parser.add_argument('--context', type=str, required=True, choices=['mixed_context', 'no_context', 'random_context', 'relevant_context', 'wrong_date_context'], help='Context to be used in the prompt')
#     parser.add_argument('--dataset', type=str, required = True, choices=['AQA','TQE'])
#     return parser.parse_args()

# args = parse_args()

# =============================================================================
# LOAD DATA
# =============================================================================
# pd.read_json(f'/home/dan/DeepLearning/mini_temporal/data/{args.dataset}/test.jsonl', lines=True, orient='records') # @$@

df = pd.read_json('/home/dan/DeepLearning/mini_temporal/data/AQA/test.jsonl', lines=True, orient='records') # @$@
df = df.iloc[0:10]
df.columns

# =============================================================================
# API CONFIG
# =============================================================================
# Load environment variables from the .env file
load_dotenv('./gpt_baselines/.env')

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Get the PROJECT key from the environment variable
proj_key = os.getenv('PROJECT_KEY')

# Get the ORGANIZATION key from the enviorment variables
org = os.getenv('ORGANIZATION')


# Check if the API key is available
if api_key is None:
    raise ValueError("API key is missing. Make sure to set OPENAI_API_KEY in your environment.")

# Check if the API key is available
if proj_key is None:
    raise ValueError("PROJECT key is missing. Make sure to set PROJECT_KEY in your environment.")

if org is None:
    raise ValueError("ORGANIZATION key is missing. Make sure to set ORGANIZATION in your environment.")

# Set the API key for the OpenAI client
openai.api_key = api_key

client = OpenAI(
#   organization=org,
  project=proj_key,
)

# =============================================================================
# PROMPTING
# =============================================================================
output_list = []
context = args.context # @$@ 

# context = 'no_context'

def create_prompt(row, context):
    if context == 'no_context':
        return f"{row['question']} The Answer is "
    else:
        return f"{row['question']} Here is the context: {row[context]} The Answer is "

df['prompt'] = df.apply(lambda row: create_prompt(row, context), axis=1)

for i, row in df.iterrows():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "system",
                    "content": "Act as a helpful assistant and answer these questions to the best of your ability."
                },
                {"role": "user", "content": row['prompt']}
            ],
            temperature=0.7,
            max_tokens=100
        )

        output = response.choices[0].message.content.strip()
        
        # Print the output as JSON
        print(json.dumps({'index': i, 'output': output}))

    except Exception as e:
        print(f"Error processing question {i}: {e}")
