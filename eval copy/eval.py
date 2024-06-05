
"""
Created on 05/22/2024

@author: Dan Schumacher

# THIS SCRIPT TAKES IN 2 JSONL FILES
ONE IS A BUNCH OF PREDICTIONS
THE OTHER IS THE ACTUAL QA DATA
PURPOSE OF SCRIPT IS COMPARE THE PREDS TO ACTUAL AND MAKE F1, CONTAINS ACC, AND BEM SCORE
"""

#endregion
#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import os
import json
import string
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from transformers import BertTokenizer
import numpy as np
from scipy.special import softmax
from bs4 import BeautifulSoup
import argparse

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')

#endregion
#region # ARGPARSE
# =============================================================================
# ARGPARSE
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="BEM Score Evaluation Script")
    parser.add_argument('--pred_file', type=str, required=True, help='Path to the prediction JSONL file')
    parser.add_argument('--actual_file', type=str, required=True, help='Path to the actual data JSONL file')
    return parser.parse_args()

#endregion
#region # FUNCTIONS
# =============================================================================
# FUNCTIONS
# =============================================================================
def load_funky_json(file_path):
    """
    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries loaded from the JSONL file.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                pass
                # print(f"Error decoding JSON: {e} - Line skipped")
        return data

def remove_html_tags(text):
    """
    Removes HTML tags from a string.

    Args:
        text (str): The input string containing HTML tags.

    Returns:
        str: The cleaned string without HTML tags.
    """
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text()

def fix_index_discontinuities(data):
    """
    Fixes discontinuities in the 'INDEX' field of a list of dictionaries.

    Args:
        data (list): A list of dictionaries, each containing an 'INDEX' key.

    Returns:
        list: A list of corrected indices if any corrections were made.
    """
    if not data or 'INDEX' not in data[0]:
        return []
    fixed_indices = []
    previous_index = data[0]['INDEX'] - 1
    for i, entry in enumerate(data):
        if entry['INDEX'] != previous_index + 1:
            corrected_index = previous_index + 1
            entry['INDEX'] = corrected_index
            fixed_indices.append(corrected_index)
        previous_index = entry['INDEX']
    return fixed_indices

def exact_match_f1(pred, answer_list):
    """
    Computes the best F1 score for a prediction against a list of possible answers.

    The F1 score is calculated based on the overlap of words between the prediction and each answer.
    The best F1 score is returned.

    Args:
        pred (str): The predicted answer as a string.
        answer_list (list of str): A list of possible correct answers.

    Returns:
        float: The highest F1 score among the predictions compared to the answers.
    """
    pred_words = set(pred.lower().split())  # Convert the prediction into a set of words for faster operations
    best_f1 = 0  # Initialize best F1 score

    for answer in answer_list:
        answer_words = set(answer.lower().split())  # Convert answer into a set of words
        TP = len(answer_words.intersection(pred_words))
        FP = len(pred_words.difference(answer_words))
        FN = len(answer_words.difference(pred_words))
        
        if TP == 0:
            f1 = 0
        else:
            prec = TP / (TP + FP) if TP + FP > 0 else 0
            rec = TP / (TP + FN) if TP + FN > 0 else 0
            if (prec + rec) > 0:
                f1 = 2 * ((prec * rec) / (prec + rec))
            else:
                f1 = 0

        if f1 > best_f1:
            best_f1 = f1

    return best_f1

def contains_metric(pred, answer_list):
    """
    Checks if any answer in the list is contained within the prediction after removing punctuation
    and converting to lowercase.

    Parameters:
    - pred (str): The prediction string to be evaluated.
    - answer_list (list of str): A list of answer strings against which the prediction is evaluated.

    Returns:
    - bool: True if any answer is contained within the prediction, False otherwise.
    """
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    normalized_pred = pred.lower().translate(translator)

    for answer in answer_list:
        # Normalize each answer
        normalized_answer = answer.lower().translate(translator)
        # Check if the normalized answer is contained within the normalized prediction
        if normalized_answer in normalized_pred:
            return 1

    return 0

def bertify_example(example, tokenizer, cls_id, sep_id):
    """
    Converts a single example into BERT input format.

    Args:
        example (dict): A dictionary containing 'question', 'reference', and 'candidate' keys.
        tokenizer (BertTokenizer): The BERT tokenizer.
        cls_id (tf.Tensor): The CLS token ID.
        sep_id (tf.Tensor): The SEP token ID.

    Returns:
        dict: A dictionary with 'input_ids' and 'segment_ids' keys if the shapes are consistent, otherwise None.
    """
    question = tokenizer.tokenize(example['question']).merge_dims(1, 2)
    reference = tokenizer.tokenize(example['reference']).merge_dims(1, 2)
    candidate = tokenizer.tokenize(example['candidate']).merge_dims(1, 2)
    
    # print(f"Shapes - Question: {question.shape}, Reference: {reference.shape}, Candidate: {candidate.shape}")
    
    if question.shape[0] != reference.shape[0] or reference.shape[0] != candidate.shape[0]:
        # print("Inconsistent shapes detected, skipping this example.")
        # print(f"Example content: {example}")
        return None
    
    input_ids, segment_ids = text.combine_segments(
        (candidate, reference, question), cls_id, sep_id)

    thingy = {'input_ids': input_ids.numpy(), 'segment_ids': segment_ids.numpy()}
    return thingy

def pad(a, length=512):
    """
    Converts a list of examples into BERT input format.

    Args:
        examples (list): A list of dictionaries, each containing 'question', 'reference', and 'candidate' keys.
        tokenizer (BertTokenizer): The BERT tokenizer.
        cls_id (tf.Tensor): The CLS token ID.
        sep_id (tf.Tensor): The SEP token ID.

    Returns:
        dict: A dictionary with 'input_ids' and 'segment_ids' keys.
    """
    thingy = np.append(a, np.zeros(length - a.shape[-1], np.int32))
    return thingy

def bertify_examples(examples, tokenizer, cls_id, sep_id):
    input_ids = []
    segment_ids = []
    for example in examples:
        example_inputs = bertify_example(example, tokenizer, cls_id, sep_id)
        if example_inputs is not None:
            input_ids.append(pad(example_inputs['input_ids']))
            segment_ids.append(pad(example_inputs['segment_ids']))
    thingy = {'input_ids': np.stack(input_ids), 'segment_ids': np.stack(segment_ids)}
    return thingy

#endregion
# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    args = parse_args()
    #region # DATA SET UP
    # =============================================================================
    # DATA SET UP
    # =============================================================================
    
    # Load predictions
    pred_list = load_funky_json(args.pred_file)


    # Fix indexes
    fix_index_discontinuities(pred_list)


    # Remove HTML tokens
    clean_list_of_dicts = [
        {
            'INDEX': d['INDEX'],
            'QUESTION': d['QUESTION'],
            'PREDICTION': remove_html_tags(d['PREDICTION'])
        }
        for d in pred_list
    ]

    # Convert to pandas DataFrame
    pred_df = pd.DataFrame(clean_list_of_dicts)
    # print(f'PRED INDEX: : {pred_df['i']}')


    # Load actual data and merge
    actual_df = pd.read_json(args.actual_file, lines=True)
    print(f'Actual_df i: {actual_df["i"]}')
    print(f'Actual_df idx: {actual_df["idx"]}')

    print('\n')
    actual_df['answer'] = actual_df['answer'].apply(lambda ans: ans[0].split('__or__'))
    pred_df.rename(columns={'QUESTION': 'question', 'PREDICTION': 'prediction'}, inplace=True)
    merged_df = pd.merge(actual_df, pred_df, on='question', how='left')
    print(f'merged_df column names: {merged_df.columns}')
    print('merged df created')
    #endregion
    #region # F1 AND CONTAINS
    # =============================================================================
    # F1 AND CONTAINS
    # =============================================================================
    f1s = []
    contains = []
    for pred, ans_list in zip(merged_df['prediction'], merged_df['answer']):
        f1s.append(exact_match_f1(pred, ans_list))
        contains.append(contains_metric(pred, ans_list))
    print(f1s)
    print(contains)
    avg_f1 = sum(f1s) / len(f1s)
    avg_contains = sum(contains) / len(contains)

    #endregion
    #region # BEM
    # =============================================================================
    # BEM
    # =============================================================================
    # Correct the format of dict_for_bem to handle multiple answers
    dict_for_bem = []
    for index, row in merged_df.iterrows():
        for answer in row['answer']:
            dict_for_bem.append({
                'question': row['question'],
                'reference': answer,
                'candidate': row['prediction']
            })

    # Define BERT tokenizer and helper functions
    vocab_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename='gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt',
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER
        ),
        num_oov_buckets=1
    )
    cls_id, sep_id = vocab_table.lookup(tf.convert_to_tensor(['[CLS]', '[SEP]']))
    tokenizer = text.BertTokenizer(
        vocab_lookup_table=vocab_table,
        token_out_type=tf.int64,
        preserve_unused_token=True,
        lower_case=True
    )

    # Load BEM model
    bem = hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')

    # Evaluate BEM scores
    total_scores = []

    # Process each prediction against all possible answers and keep the highest score
    for example in dict_for_bem:
        input_data = bertify_examples([example], tokenizer, cls_id, sep_id)
        if input_data:
            raw_outputs = bem(input_data)
            score = float(softmax(np.squeeze(raw_outputs))[1])
            total_scores.append(score)
            # print(f'Example: {example}')
            # print(f'Score: {score}')

    if total_scores:
        average_bem_score = np.mean(total_scores)
    print(json.dumps({'NAME':args.pred_file, 'BEM': average_bem_score, 'F1': avg_f1, 'ACC':avg_contains}))
    
    #endregion
if __name__ == "__main__":
    main()
