"""
Created on 05/13/2024

@author: Dan Schumacher

Functions include:
    - Price approximator

"""
#endregion
#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================
import os
os.chdir('/home/dan/DeepLearning/mini_temporal/data')

# IMPORTS
import json
import pandas as pd
from dotenv import load_dotenv
import openai
from openai import OpenAI


# import json

# Load environment variables from the .env file
load_dotenv('.env')

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if api_key is None:
    raise ValueError("API key is missing. Make sure to set OPENAI_API_KEY in your environment.")

# Set the API key for the OpenAI client
openai.api_key = api_key
client = OpenAI(api_key=api_key)

#endregion
#region # PRICE APPROXIMATOR
# =============================================================================
# PRICE APPROXIMATOR
# =============================================================================

def price_approximator(df, col, price_per_1000_tokens_input):
    """
    Calculates the estimated cost of using a model based on the number of tokens generated from
    a dataframe column for inputs.

    Args:
    df (pd.DataFrame): The dataframe containing the text data.
    col (str): The name of the column in the dataframe that contains the text data.
    price_per_1000_tokens_input (float): The price per 1000 tokens for the input text.

    Returns:
    float: The total estimated cost based on the input text tokens.

    This function first calculates the total number of words in the specified dataframe column and converts
    this to an estimated number of tokens assuming 1 word is approximately 1.33 tokens. It then calculates
    the cost for these input tokens. The total price for input usage is returned.
    """

    total_number_of_words = 0
    for i in range(len(df)):
        total_number_of_words += len(df[col].iloc[i].split())
    approx_total_tokens = total_number_of_words * (4/3)  # Converting words to tokens
    input_price = (approx_total_tokens / 1000) * price_per_1000_tokens_input
    # print('cost:', f'${input_price:.2f}')
    return input_price

#endregion
#region # GET CONTEXT
# =============================================================================
# GET CONTEXT
# =============================================================================
def get_context(question, answer, system_prompt):
    """
    Retrieves generated context based on a provided question and answer using OpenAI's GPT-3.5-turbo model.
    
    This function formats a user query consisting of a question and its answer into a prompt,
    sends it to OpenAI's API, and returns the generated response in JSON format.
    The response is expected to be a single paragraph that provides context relevant to the question,
    formatted similarly to a Wikipedia article entry.

    Args:
        question (str): The question related to the context generation.
        answer (str): The answer to the question, which the model uses to generate context.

    Returns:
        str: The generated context in JSON format as returned by the model.
    
    Raises:
        OpenAIError: If an error occurs in the API call.
    """
    user_prompt = f'question: "{question}"\nanswer: "{answer}"\noutput:\n'

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature=0.1,
        response_format={ 
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
    )

    return response.choices[0].message.content

#endregion
#region # TRIAL PRINT
# =============================================================================
# TRIAL PRINT
# =============================================================================
def trial_print(df, system_prompt):
    """
    Prints generated contexts for the first five rows of a DataFrame based on the 'question' and 'answer' columns.
    
    This function iterates through the first five entries of the provided DataFrame, extracts the question and
    answer for each entry, and uses these to generate context via the `get_context` function. The result for each
    query is printed, along with the original question for clarity.

    Args:
        df (pd.DataFrame): A DataFrame containing at least two columns: 'question' and 'answer'.
                           The DataFrame should have the questions and answers for which context is generated.

    Returns:
        None: This function prints the results and does not return any value.
    """
        
    for _, row in df[:5].iterrows():
        question = row['question']
        answer = row['answer']
        result = get_context(question, answer, system_prompt)
        print(f"QUESTION: {question}\n\nRESULT: {result}")
        print("\n\n------------------------------------------------------------------------------------\n\n")

#endregion
#region # FORMAT FOR BATCHING
# =============================================================================
# FORMAT FOR BATCHING
# =============================================================================
def format_for_batching(df, system_prompt):

    example = '''
    question: "What company was formed in 1986 by the merger of Burroughs and Sperry?"
    answer: "Unisys"
    output:
    {
        "generated_context": "The Burroughs Corporation was a major American manufacturer of business equipment. The company was founded in 1886 as the American Arithmometer Company by William Seward Burroughs. In 1986, it merged with Sperry UNIVAC to form Unisys. The company's history paralleled many of the major developments in computing. At its start, it produced mechanical adding machines, and later moved into programmable ledgers and then computers. It was one of the largest producers of mainframe computers in the world, also producing related equipment including typewriters and printers."
    }'''


    tasks = []

    for index, row in df.iterrows():
        
        question = row['question']
        answer = row['answer']
        user_prompt = f'{example}\nquestion: "{question}"\nanswer: "{answer}"\noutput:\n'
        task = {
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                # This is what you would have in your Chat Completions API call
                "model": "gpt-3.5-turbo-1106",
                "temperature": 0.1,
                "response_format": { 
                    "type": "json_object"
                },
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
            }
        }
        
        tasks.append(task)
    return tasks


#endregion
#region # RETRIEVE AND SAVE BATCH RESULTS
# =============================================================================
# RETRIEVE AND SAVE BATCH RESULTS       
# =============================================================================
# generation_functions.py


def retrieve_and_save_batch_results(batch_id, save_location, dataset):
    """
    Retrieve the results of a batch job from the OpenAI API, save them to a file, 
    and print the first few results mapped back to the original dataset.
    
    Parameters:
    batch_id (str): The ID of the batch job to retrieve.
    save_location (str): The file path to save the retrieved results.
    dataset (pandas.DataFrame): The dataset to map the results back to the input questions.
    
    Returns:
    None
    """
    # Retrieve batch job details
    batch_job = openai.batches.retrieve(batch_id)
    
    # Retrieve the output file ID
    result_file_id = batch_job.output_file_id
    
    # Retrieve results from the output file
    result = openai.files.content(result_file_id).content
    
    # Save results to file
    with open(save_location, 'wb') as file:
        file.write(result)
    
    # Load data from saved file
    results = []
    with open(save_location, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)
    
    # Read only the first results
    for res in results[:5]:
        task_id = res['custom_id']
        index = task_id.split('-')[-1]
        result_content = res['response']['body']['choices'][0]['message']['content']
        question = dataset.iloc[int(index)]['question']
        print(f"QUESTION: {question}\n\nRESULT: {result_content}")
        print("\n\n----------------------------\n\n")

#endregion
#region # CREATE BATCH JOB
# =============================================================================
# CREATE BATCH JOB
# =============================================================================
def create_batch_job(input_file, description):
    batch_job = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": description
        }
    )
    return batch_job.id