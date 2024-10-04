#%%  
import os, pickle, sys, time
import torch
import numpy as np
import random
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re
import json
import openai
import requests
from requests.exceptions import ConnectionError

#%%  
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)

#%%  
dataset = 'Recipe-MPR'
from pathlib import Path
import pickle

path = os.path.join(os.getcwd() ,'..', 'data', dataset)
if not os.path.exists(path):
    path = os.path.join(os.getcwd() , 'data', dataset)

root = Path(path)
print(root)
# %%
# Load the Recipe-MPR JSON file
with open(os.path.join(root, "500QA.json"), "r") as file:
    MPR_data = json.load(file)


# Extract keys from the "options" dictionaries
options_ids = set()
for item in MPR_data:
    options = item.get("options", {})
    options_ids.update(options.keys())

print(len(options_ids))
print(list((options_ids))[:6])
# %%
with open(os.path.join(root, "layer1.json"), "r") as file:
    layer1_data = json.load(file)
# %%
id2instructions = {}
for recipe in layer1_data:
    id = recipe['id']
    if id in id2instructions:
        print("recipe duplicate for:\n", id)
        continue
    else:
        instructions = ''
        i = 0
        for instruction in recipe['instructions']:
            i += 1
            text = instruction['text']
            instructions += f'{i}. {text} \n'
        id2instructions[id] = instructions
# %%
# with open(os.path.join(root, "id2instructions.json"), "w") as file:
#     json.dump(id2instructions, file)

with open(os.path.join(root, "id2instructions.json"), "r") as file:
    id2instructions = json.load(file)






# %%
# def create_tools_prompt(instruction):
#     prompt = '''I have a set of kitchen tools: knife, cutting board, cast iron skillet, stainless steel skillet, nonstick fry pan, saucepan, dutch oven, stock pot, baking dish, measuring spoon, measuring cup, sheet pan, grater, mixing bowl, tong, spatula, strainer, peeler, whisk, wooden spoon, colander, blender, scale, can opener, rolling pin, ladle.\n'''
#     prompt += '''\nQuestion: From the set of kitchen tools, which ones does this recipe use? If none of the kitchen tools are used, answer with “nothing."\n'''
#     prompt += '''0. Preheat oven to 350 degrees Fahrenheit.\n 1. Spray pan with non stick cooking spray.\n 2. Heat milk, water and butter to boiling; stir in contents of both pouches of potatoes; let stand one minute.\n 3. Stir in corn.\n 4. Spoon half the potato mixture in pan.\n 5. Sprinkle half each of cheese and onions; top with remaining potatoes.\n 6. Sprinkle with remaining cheese and onions.\n 7. Bake 10 to 15 minutes until cheese is melted.\n 8. Enjoy !\n'''
#     prompt += '''Answer: sauce pan, knife, grater, wooden spoon, cutting board\n'''
#     prompt += '''\nQuestion: From the set of kitchen tools, which ones does this recipe use? If none of the kitchen tools are used, answer with “nothing.”\n'''
#     prompt += f'{instruction}\n'
#     prompt += 'Answer:'

#     return prompt

prompt_system_tools = '''I have a set of kitchen tools: knife, cutting board, cast iron skillet, stainless steel skillet, nonstick fry pan, saucepan, dutch oven, stock pot, baking dish, measuring spoon, measuring cup, sheet pan, grater, mixing bowl, tong, spatula, strainer, peeler, whisk, wooden spoon, colander, blender, scale, can opener, rolling pin, ladle.\n'''
prompt_system_tools += '''\nQuestion: From the set of kitchen tools, which ones does this recipe use? If none of the kitchen tools are used, answer with “nothing.  IMPORTANT: Remember you only can choose among the provided options. You must not choose any other tool."\n'''

prompt_user_tools = '''0. Preheat oven to 350 degrees Fahrenheit.\n 1. Spray pan with non stick cooking spray.\n 2. Heat milk, water and butter to boiling; stir in contents of both pouches of potatoes; let stand one minute.\n 3. Stir in corn.\n 4. Spoon half the potato mixture in pan.\n 5. Sprinkle half each of cheese and onions; top with remaining potatoes.\n 6. Sprinkle with remaining cheese and onions.\n 7. Bake 10 to 15 minutes until cheese is melted.\n 8. Enjoy !\n'''
prompt_assistant_tools = '''Answer: sauce pan, knife, grater, wooden spoon, cutting board\n'''

def create_tools_prompt(instruction):
    prompt = f'{instruction}'
    return prompt


prompt_system_methods = '''I have a set of cooking methods: braise, grill, roast, bake, blanche, stew, poach, steam, fry, boil, sauté, simmer, sous vide, broil, barbecue.\n'''
prompt_system_methods += '''\nQuestion: From the set of cooking methods, which ones does this recipe use? If none of the listed cooking methods are used, answer with “nothing.”\n'''
prompt_user_methods = '''1. Preheat oven to 350 degrees Fahrenheit.\n 2. Spray pan with non stick cooking spray.\n 3. Heat milk, water and butter to boiling; stir in contents of both pouches of potatoes; let stand one minute.\n 4. Stir in corn.\n 5. Spoon half the potato mixture in pan.\n 6. Sprinkle half each of cheese and onions; top with remaining potatoes.\n 7. Sprinkle with remaining cheese and onions.\n 8. Bake 10 to 15 minutes until cheese is melted.\n 9. Enjoy !\n'''
prompt_assistant_methods = '''Answer: bake\n'''
prompt_user_methods2 = '''1. Add the tomatoes to a food processor with a pinch of salt and puree until smooth.\n 2. Combine the onions, bell peppers and cucumbers with the tomato puree in a large bowl.\n 3. Chill at least 1 hour.\n 4. Drizzle with olive oil, garnish with chopped basil and serve.\n'''
prompt_assistant_methods2 = '''Answer: nothing\n'''


def create_method_prompt(instruction):
    prompt = f'{instruction}'
    return prompt

def get_response(prompt_system, prompt_user, prompt_assistant, prompt_user2 = None, prompt_assistant2 = None ,prompt = None):
    tries = 5
    for i in range(tries):
        if prompt_user2 is None and prompt_assistant2 is None:
            message = [{"role": "system", "content": prompt_system}, {"role": "user", "content": prompt_user}, {"role": "assistant", "content": prompt_assistant}, {"role": "user", "content": prompt}]
        elif prompt_user2 is not None and prompt_assistant2 is not None:
            message = [{"role": "system", "content": prompt_system}, {"role": "user", "content": prompt_user}, {"role": "assistant", "content": prompt_assistant}, {"role": "user", "content": prompt_user2}, {"role": "assistant", "content": prompt_assistant2}, {"role": "user", "content": prompt}]
        else:
            raise ValueError("prompt_user2 and prompt_assistant2 must be both None or both not None")
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = message,
                temperature = 0,
                max_tokens = 300)
            break
            
        except ConnectionError as err:
            if i == tries - 1:
                raise err
            else:
                time.sleep(5)

    return response['choices'][0]['message']['content']
    

id2tools = {}
#id2methods = {}
inst_counter = 0
allowed_tools = ['knife', 'cutting board', 'cast iron skillet', 'stainless steel skillet', 'nonstick fry pan', 'saucepan', 'dutch oven', 'stock pot', 'baking dish', 'measuring spoon', 'measuring cup', 'sheet pan', 'grater', 'mixing bowl', 'tong', 'spatula', 'strainer', 'peeler', 'whisk', 'wooden spoon', 'colander', 'blender', 'scale', 'can opener', 'rolling pin', 'ladle']

for id, instruction in id2instructions.items():
    selected_tools = []
    if inst_counter == 5:
        break
    prompt_tool = create_tools_prompt(instruction)
    tools_str = get_response(prompt_system = prompt_system_tools, prompt_user = prompt_user_tools, prompt_assistant = prompt_assistant_tools
    ,prompt_user2 = None, prompt_assistant2 = None, prompt = prompt_tool)
    if tools_str.startswith("Answer: "):
        tools = tools_str.replace("Answer: ", "").split(", ")
    else:
        tools = tools_str

    for tool in tools:
        if tool in allowed_tools:
            selected_tools.append(tool)



    #prompt_method = create_method_prompt(instruction)
    #methods_str = get_response(prompt_system = prompt_system_methods, prompt_user = prompt_user_methods, prompt_assistant = prompt_assistant_methods
    #,prompt_user2 = prompt_user_methods2, prompt_assistant2 = prompt_assistant_methods2, prompt = prompt_method)
    #if methods_str.startswith("Answer: "):
    #    methods = methods_str.replace("Answer: ", "").split(", ")
    #else:
    #    methods = methods_str
    id2tools[id] = selected_tools
    #id2methods[id] = methods
    inst_counter += 1

#print(id2methods)
# %%
with open(os.path.join(root, "id2tools.json"), "w") as file:
    json.dump(id2tools, file)


# %%

with open(os.path.join(root, "id2tools.json"), "r") as file:
    id2tools = json.load(file)

with open(os.path.join(root, "CookingMethods.pickle"), "rb") as file:
    CookingMethods = pickle.load(file)

# %%
with open('KG.txt', 'w') as kg_file:
    for id_key, tools_list in id2tools.items():
        unique_tools = set()
        for tool in tools_list:
            if tool not in unique_tools:
                kg_file.write(f"{id_key}, has_tool, {tool}\n")
                unique_tools.add(tool)

    for id_key, methods_list in CookingMethods.items():
        unique_methods = set()
        for method in methods_list:
            if method not in unique_methods:
                kg_file.write(f"{id_key}, has_method, {method}\n")
                unique_methods.add(method)
# %%

# %%
