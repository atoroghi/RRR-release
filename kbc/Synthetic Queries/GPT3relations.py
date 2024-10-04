#%%
import openai
import os, pickle, sys, time
from tqdm import tqdm
import random
import json
import re
import requests
from requests.exceptions import ConnectionError

#%%
dataset = 'Recipe-MPR'
from pathlib import Path
import pickle

path = os.path.join(os.getcwd() ,'..', 'data', dataset)
if not os.path.exists(path):
    
    path = os.path.join(os.getcwd() , 'data', dataset)


root = Path(path)
os.chdir(root)
print(root)
print(os.listdir(root))
#%%
def create_prompt(triple):
    prompt = ""
    prompt_context = f'''
    TASK: Convert the knowledge graph triple’s relation to natural language while preserving the meaning of the relation using the original relation’s keywords.\n
    TRIPLE: ('Dominican Republic', '/location/country/form_of_government', 'republic')\n 
    ANSWER: ('Dominican Republic', 'has the form of government', 'republic')\n\n
    TASK: Convert the knowledge graph triple’s relation to natural language while preserving the meaning of the relation using the original relation’s keywords.\n
    TRIPLE: ('The Bahamas', '/organization/organization_member/member_of./organization/organization_membership/organization', 'World Bank')\n
    ANSWER: ('The Bahamas', 'is a member of the organization', 'World Bank')\n\n
    TASK: Convert the knowledge graph triple’s relation to natural language while preserving the meaning of the relation using the original relation’s keywords.\n
    TRIPLE: ({triple})
    ANSWER:\n'''
    prompt += prompt_context
    return prompt

#%%
def get_response(prompt):
    tries = 5
    for i in range(tries):
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = [{"role": "user", "content": prompt}],
                temperature = 0,
                max_tokens = 60)
            break
        except ConnectionError as err:
            if i == tries - 1:
                raise err
            else:
                time.sleep(5)

    return response['choices'][0]['message']['content']
#%%
with open('readable_queries_sofar.txt', 'r') as f:
    triples = f.readlines()

with open('GPT readable relations.txt', 'w') as f1:
    for i, triple in enumerate(triples[:50]):
        triple = triple.strip("()")
        prompt = create_prompt(triple)
        response = get_response(prompt)
        f1.write(response + '\n')
        print(f'relations converted: {i}')
f1.close()
# %%
import copy
with open('queries.json', 'r') as json_file:
    queries = json.load(json_file)

queries_readable_rels = {}

with open('GPT readable relations.txt', 'r') as f:
    modified_triples = f.readlines()
f.close()

for i, q in enumerate(queries):
    if i == len(modified_triples):
        break

    query = queries[q]
    modified_readable_triple = modified_triples[i].rstrip('\n').strip('()""').split(',')
    
    modified_rel = modified_triples[i].strip('()').split(',')[1]

    queries_readable_rels[q] = copy.deepcopy(query)
    queries_readable_rels[q]['relations'] = modified_rel
    queries_readable_rels[q]['triples readable'] = modified_readable_triple
with open('queries_readable_rels.json', 'w') as json_file:
    json.dump(queries_readable_rels, json_file)


# %%
