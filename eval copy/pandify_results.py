import os

os.chdir('/home/dan/DeepLearning/mini_temporal')

import pandas as pd

results = pd.read_json('./eval/results.jsonl', lines = True, orient='records')

results['NAME'] = results['NAME'].split('/')[-1]

results['NAME'] = results['NAME'].apply(lambda x: x.split('/')[-1][:-6].replace('_context','-context').replace('1_1','1.1').replace('_',' '))

results['DATASET'] = results['NAME'].apply(lambda x: x.split()[0])
results['BASE_MODEL'] = results['NAME'].apply(lambda x: x.split()[1])
results['TRAINED_ON'] = results['NAME'].apply(lambda x: x.split()[2])
results['TESTED_ON'] = results['NAME'].apply(lambda x: x.split()[4])

results.drop('NAME',axis = 1, inplace = True)

results