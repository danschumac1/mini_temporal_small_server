import os
os.chdir('./gpt_baselines')
os.path.exists('batch_trial.jsonl')

from openai import OpenAI
client = OpenAI()

batch_input_file = client.files.create(
    file=open('batch_trial.jsonl', 'rb'),
    purpose='batch'
)

batch_input_file_id = batch_input_file.id

client.batches.create( 
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    }
)
# batch_ukQXfiF5JktQaipQIdWtEtCV
# file-5PHjIEsPIHkVGGoJGpWeidKq

# CHECKING STATUS OF BATCH
client.batches.retrieve("batch_ukQXfiF5JktQaipQIdWtEtCV").status

# RETRIEVING THE RESULTS
content = client.files.content("file-5PHjIEsPIHkVGGoJGpWeidKq")

content.response