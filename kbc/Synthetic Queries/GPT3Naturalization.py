#%%
import openai
import os, pickle, sys, time
from tqdm import tqdm
import random
import json
import re
import requests
from requests.exceptions import ConnectionError
openai.api_key = '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'
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



def create_prompt(query, key_terms):
    prompt = ''''''
    # prompt_context = f"TASK: Rephrase this question in 3 different concise ways that sound more natural and seem to be written by a fluent English speaker" \
    #     "while preserving the meaning of the original question.\n\n"\
    #     "QUESTION: What is the name of an administrative district and a geographical area with capital and human at it that /location/location/contains St. Augustine?"\
    #         "(Original Triple: United States of America, /location/location/contains, St. Augustine)\n\n"\
    #     "ANSWER:\n"\
    #     "1. What is the name of an administrative district and a geographical area that has a capital and humans living in it which contains St. Augustine?\n"\
    #     "2. What is a geographical area and an administrative district where humans live in and has a capital that contains St. Augustine?\n"\
    #     "3. What is a geographical area and an administrative district with humans living in it that has a capital and contains St. Augustine?\n\n"\
    #     "QUESTION:\n"\
    #     "What is the name of a natural thing and a rocky area with skier and snow at it that is base/location/located in Japan? (Original Triple: Mount Fuji"\
    #     ", base/location/located, Japan)\n\n"\
    #     "1. What is a natural rocky landmark where skiers and snow can be found in it and is located in Japan?\n"\
    #     "2. What is an instance of a natural thing and rocky area with skiers and snow located at it that is in Japan?\n"\
    #     "3. What is a natural rocky area located in Japan that skiers and snow can be found?\n\n"\
    #     "QUESTION:\n"\
    #     f"{query}\n\n"\
    #     "ANSWER:\n"
    prompt_context = f'''"question": "{query}", "key_terms": {key_terms}
    '''
    prompt += prompt_context
    return prompt

def second_prompt(query):
    answer = query['answer']
    return f"considering that the answer of this question is \"{answer}\", which one of the three questions you generated do you think is the best? Only respond with a number from (1, 2, or 3). Please only use a number in your response. Your response must not include any words and any additional characters."
    


def get_response(prompt_system, prompt_user, prompt_assistant, prompt):
    tries = 5
    for i in range(tries):
        message = [{"role": "system", "content": prompt_system}, {"role": "user", "content": prompt_user}, {"role": "assistant", "content": prompt_assistant}, {"role": "user", "content": prompt}]
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = message,
                temperature = 0,
                max_tokens = 300)
            print('success')
            break
            
        except ConnectionError as err:
            if i == tries - 1:
                raise err
            else:
                time.sleep(5)

    return response['choices'][0]['message']['content']

#%%
a = create_prompt('What is the name of a administrative district and a geographical area with capital and humans at it that /location/location/contains St. Augustine? (Original Triple: United States of America, /location/location/contains, St. Augustine)) ')
b = get_response(a)
print(b)

# %%
with open('template questions.txt', 'r') as f:
    questions = f.readlines()

with open('queries_readable_rels.json', 'r') as f:
    queries = json.load(f)
# %%
prompt_system = '''You are an English writing expert.
You will be given a JSON object with 2 properties: “question” a question to rephrase, and “key_terms” a list of key terms from the question. 
Your first task is to rephrase the question in "question" in 3 different concise ways without changing its meaning and using all the terms in "key_terms" or words similar to them. Each option must start with "what" and nothing else.  
Answer in the following format: option1 / option2 / option3 /
Your second task is to pick which of the 3 options you generated sounds the most natural and is closest in meaning to the original question given. 
Answer with 1, 2, or 3. '''
prompt_user = '''“question”: "What is an instance of a natural thing and rocky area with a skier located at it that is in the country producing Toyota?”, “key_terms”: [“natural thing”, “rocky area”, “skier”, “country”, “Toyota”]'''
prompt_assistant = '''What is the name of a natural rocky landmark in the country producing Toyota where people can ski? / What is a natural rocky area in the country that produces Toyota where you can find skiers? / What is a natural rocky landmark where skiers can be found in the country producing Toyota? / 3
'''
with open('GPT questions.txt', 'w') as f1:
    for i, question in enumerate(questions[:10]):
        question = question.split('(Original Triple:')[0]
        q = list(queries.keys())[i]
        query = queries[q]

        key_terms = []
        key_terms.append(query['triples readable'][2])
        for CN_fact in query['CN_facts']:
            key_terms.append(CN_fact[0][2])

        prompt = create_prompt(question, key_terms)

        response = get_response(prompt_system, prompt_user, prompt_assistant,prompt)
        f1.write(response)
        f1.write('\n')
        # followup_prompt = second_prompt(queries[i])
        # final_response = get_response(followup_prompt)
        # f2.write(final_response)
        # f2.write('\n')


# %%
