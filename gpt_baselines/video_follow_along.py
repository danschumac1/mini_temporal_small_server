from openai import OpenAI
import time
from datetime import datetime
import json
import os

os.chdir('/home/dan/DeepLearning/mini_temporal/gpt_baselines')
os.path.exists('batch_trial.jsonl')

class OpenAIBatchProcessor:
    def __init__(self, api_key):
        client = OpenAI(api_key = api_key)
        self.client = client
      
    def process_batch(self, input_file_path, endpoint, completion_window, model):
        
        headers = {
                'OpenAI-Project-ID': self.project_id
            }

        # UPLOAD THE INPUT FILE
        with open(input_file_path, 'rb') as file:
            uploaded_file = self.client.files.create(
                file = file,
                purpose = 'batch',
                headers=headers
            )
    
        # CREATE THE BATCH JOB
        batch_job = self.client.batches.create(
            input_file_id = uploaded_file.id,
            endpoint=endpoint,
            model=model,
            completion_window = completion_window
        )

        # MONITOR AND SAVE THE RESULTS
        while batch_job.status not in ['completed','failed','cancelled']:
            batch_job = self.client.batches.retrieve(batch_job.id)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f'{current_time} batch job status: {batch_job.status}... trying again in 60 seconds ...')
            time.sleep(60)
        
        if batch_job.status == 'completed':
            result_file_id = batch_job.output_file_id
            result = self.client.files.retrieve(result_file_id).decode('utf-8')

            result_file_name = 'batch_job_results.jsonl'
            with open(result_file_name, 'w') as file:
                file.write(result)
            
            # LOAD DATA FROM SAVED FILE
            results = []
            with open(result_file_name, 'r') as file:
                for line in file:
                    json_object = json.loads(line.strip())
                    results.append(json_object)

            return results
        else:
            print(f'Batch job failed with status: {batch_job.status}')
            return None


# Initialize the OpenAIBatch Processor
api_key = os.getenv('OPEN_API_KEY')
processor = OpenAIBatchProcessor(api_key)

input_file_path = 'batch_trial.jsonl'
endpoint = '/v1/chat/completions'
completion_window = '24h'
model = 'gpt-3.5-turbo-0125'


results = processor.process_batch(input_file_path, endpoint, completion_window, model)

print(results)