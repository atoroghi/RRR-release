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

# This is how "layer1.json" looks like
#[
#{"ingredients": [{"text": "6 ounces penne"}, {"text": "2 cups Beechers Flagship Cheese Sauce (recipe follows)"}, {"text": "1 ounce Cheddar, grated (1/4 cup)"}, {"text": "1 ounce Gruyere cheese, grated (1/4 cup)"}, {"text": "1/4 to 1/2 teaspoon chipotle chili powder (see Note)"}, {"text": "1/4 cup (1/2 stick) unsalted butter"}, {"text": "1/3 cup all-purpose flour"}, {"text": "3 cups milk"}, {"text": "14 ounces semihard cheese (page 23), grated (about 3 1/2 cups)"}, {"text": "2 ounces semisoft cheese (page 23), grated (1/2 cup)"}, {"text": "1/2 teaspoon kosher salt"}, {"text": "1/4 to 1/2 teaspoon chipotle chili powder"}, {"text": "1/8 teaspoon garlic powder"}, {"text": "(makes about 4 cups)"}], "url": "http://www.epicurious.com/recipes/food/views/-world-s-best-mac-and-cheese-387747", "partition": "train", "title": "Worlds Best Mac and Cheese", "id": "000018c8a5", "instructions": [{"text": "Preheat the oven to 350 F. Butter or oil an 8-inch baking dish."}, {"text": "Cook the penne 2 minutes less than package directions."}, {"text": "(It will finish cooking in the oven.)"}, {"text": "Rinse the pasta in cold water and set aside."}, {"text": "Combine the cooked pasta and the sauce in a medium bowl and mix carefully but thoroughly."}, {"text": "Scrape the pasta into the prepared baking dish."}, {"text": "Sprinkle the top with the cheeses and then the chili powder."}, {"text": "Bake, uncovered, for 20 minutes."}, {"text": "Let the mac and cheese sit for 5 minutes before serving."}, {"text": "Melt the butter in a heavy-bottomed saucepan over medium heat and whisk in the flour."}, {"text": "Continue whisking and cooking for 2 minutes."}, {"text": "Slowly add the milk, whisking constantly."}, {"text": "Cook until the sauce thickens, about 10 minutes, stirring frequently."}, {"text": "Remove from the heat."}, {"text": "Add the cheeses, salt, chili powder, and garlic powder."}, {"text": "Stir until the cheese is melted and all ingredients are incorporated, about 3 minutes."}, {"text": "Use immediately, or refrigerate for up to 3 days."}, {"text": "This sauce reheats nicely on the stove in a saucepan over low heat."}, {"text": "Stir frequently so the sauce doesnt scorch."}, {"text": "This recipe can be assembled before baking and frozen for up to 3 monthsjust be sure to use a freezer-to-oven pan and increase the baking time to 50 minutes."}, {"text": "One-half teaspoon of chipotle chili powder makes a spicy mac, so make sure your family and friends can handle it!"}, {"text": "The proportion of pasta to cheese sauce is crucial to the success of the dish."}, {"text": "It will look like a lot of sauce for the pasta, but some of the liquid will be absorbed."}]},
#{"ingredients": [{"text": "1 c. elbow macaroni"}, {"text": "1 c. cubed American cheese (4 ounce.)"}, {"text": "1/2 c. sliced celery"}, {"text": "1/2 c. minced green pepper"}, {"text": "3 tbsp. minced pimento"}, {"text": "1/2 c. mayonnaise or possibly salad dressing"}, {"text": "1 tbsp. vinegar"}, {"text": "3/4 teaspoon salt"}, {"text": "1/2 teaspoon dry dill weed"}], "url": "http://cookeatshare.com/recipes/dilly-macaroni-salad-49166", "partition": "train", "title": "Dilly Macaroni Salad Recipe", "id": "000033e39b", "instructions": [{"text": "Cook macaroni according to package directions; drain well."}, {"text": "Cold."}, {"text": "Combine macaroni, cheese cubes, celery, green pepper and pimento."}, {"text": "Blend together mayonnaise or possibly salad dressing, vinegar, salt and dill weed; add in to macaroni mix."}, {"text": "Toss lightly."}, {"text": "Cover and refrigeratewell."}, {"text": "Serve salad in lettuce lined bowl if you like."}, {"text": "Makes 6 servings."}]},

with open(os.path.join(root, "layer1.json"), "r") as file:
    layer1_data = json.load(file)
id2url = {}

for item in layer1_data:
    if item["id"] in options_ids:
        id2url[item["id"]] = item["url"]
# %%
with open(os.path.join(root, "recipes_with_nutritional_info.json"), "r") as file:
    nutri_data = json.load(file)

for item in nutri_data:
    if item["id"] in options_ids:
        if item["id"] in id2url:
            if item["url"] != id2url[item["id"]]:
                print("URLs do not match")
                print(item["url"])
                print(id2url[item["id"]])
                break
        else:
            id2url[item["id"]] = item["url"]
# nothing prints, so the URLs from the two files match
# %%
# Save the id2url dictionary as a pickle file
with open(os.path.join(root, 'id2url.pickle'), "wb") as pickle_file:
    pickle.dump(id2url, pickle_file)

# %%
kg_path = os.path.join(root, 'foodkg-core.trig')
with open(kg_path, 'r', encoding='utf-8') as f:
    trig_text = f.read()
# %%
url2id = {url: id for id, url in id2url.items()}

    

# %%
# finding a dictionary that maps MPR ids to FoodKG assertion IDs
pattern = r'recipe-kb:provenance-([\w-]+)\s*\{([^}]+)\}'
matches = re.findall(pattern, trig_text, re.DOTALL)
id2assertion = {}
for match in matches:
    assertion_id = match[0]
    assertion_text = match[1]

    # Extract URLs from assertion_text
    urls = re.findall(r'<(https?://[^>]+)>', assertion_text)

    # if the urls is in the url2id, it was a url used in MPR
    for url in urls:
        if url in url2id:
            id2assertion[url2id[url]] = assertion_id




# for assertion_id, link in matches:
#     if link in list(id2url.values()):
#         id2assertion[url2id[link]] = assertion_id
# we don't have assertions for all the recipes (1233 out of 1834)
# %%
with open(os.path.join(root, 'id2assertion.pickle'), "wb") as pickle_file:
    pickle.dump(id2assertion, pickle_file)

# %%
assertion2id = {v: k for k, v in id2assertion.items()}
# %%
# Create an empty dictionary id2ingredients
id2ingredients = {}

# Extract assertion ID and ingredients
pattern = r'recipe-kb:assertions-([\w-]+)\s*\{([^}]+)\}'

matches = re.findall(pattern, trig_text, re.DOTALL)

for match in matches:
    assertion = match[0]
  
    if assertion in id2assertion.values():
        id_value = assertion2id[assertion]

        ingredients_text = match[1]

    # # Extract ingredient names from ingredients_text
        ingredients = re.findall(r'ingredient/([\w%]+)>', ingredients_text)

    # # Add the ingredients to the id2ingredients dictionary
        id2ingredients[id_value] = ingredients
print(len(id2ingredients))
# the discrepancy between len(id2ingredients) and len(id2assertion) is because some ids map to the same assertions
# %%
# we need to compensate for this discrepancy
non_unique_ids = []
for id in id2assertion.keys():
    if id not in id2ingredients.keys():
        non_unique_ids.append(id)
print(len(non_unique_ids))
# %%
nn_ids = []
nn_asss = []
for k,v in id2assertion.items():
    if k in non_unique_ids:
        nn_ids.append(k)
        nn_asss.append(v)

for i in range(len(nn_ids)):
    nn_id = nn_ids[i]
    nn_ass = nn_asss[i]
    id2ingredients[nn_id] = id2ingredients[assertion2id[nn_ass]]

print(len(id2ingredients))

# %%
with open(os.path.join(root, 'id2ingredients.pickle'), "wb") as pickle_file:
    pickle.dump(id2ingredients, pickle_file)
# %%
with open('KG.txt', 'w') as kg_file:

    for id_key, ingredients_list in id2ingredients.items():
        # only store unique ingredients
        unique_ingredients = set()
        
        for ingredient in ingredients_list:

            if ingredient not in unique_ingredients:

                kg_file.write(f"{id_key}, has_ingredient, {ingredient}\n")

                unique_ingredients.add(ingredient)
# food KG seems to miss ingredients for id '3b3fc7998f' with assertion '0ac7bb7879137a7c90f1507f4bf575421400c6c8'
# here are its info from layer1.json:  {'ingredients': [{'text': '1/2 teaspoon Sriracha'}, {'text': '1 hard-cooked egg'}], 'url': 'http://www.myrecipes.com/recipe/egg-with-kick', 'partition': 'train', 'title': 'Egg with a Kick', 'id': '3b3fc7998f', 'instructions': [{'text': 'Drizzle Sriracha over hard-cooked large egg.'}]}
# we have to manually add these lines to the KG.txt file!
#3b3fc7998f, has_ingredient, hardcooked%20egg
#3b3fc7998f, has_ingredient, sriracha%20sauce
# %%

id2ingredients_unique = {}
for id_key, ingredients_list in id2ingredients.items():
    unique_ingredients = set()
    for ingredient in ingredients_list:
        if ingredient not in unique_ingredients:
            unique_ingredients.add(ingredient)
    id2ingredients_unique[id_key] = list(unique_ingredients)


with open(os.path.join(root, 'id2ingredients_unique.pickle'), "wb") as pickle_file:
    pickle.dump(id2ingredients_unique, pickle_file)



# %%
name_to_code = {}
code = 0

with open(os.path.join(root, 'KG.txt'), 'r') as input, open(os.path.join(root, 'KG_coded.txt'), 'w') as output:
    for line in input:
        line = line.strip().split(',')
        if line:
            head, relation, tail = line
            if head not in name_to_code:
                name_to_code[head] = code
                code += 1
            if tail not in name_to_code:
                name_to_code[tail] = code
                code += 1
            output.write(f'{name_to_code[head]}\t0\t{name_to_code[tail]}\n')
with open(os.path.join(root, 'name2code.pickle'), "wb") as pickle_file:
    pickle.dump(name_to_code, pickle_file)

# %%
kg = np.genfromtxt(os.path.join(root, 'KG_coded.txt'), delimiter='\t', dtype=np.int32)
n_e = len(set(kg[:, 0]) | set(kg[:, 2]))
print("number of entities: ", n_e)
n_r = len(set(kg[:, 1]))
print("number of relations: ", n_r)

# %%
def split_kg(kg, split = 0.2):
    np.random.shuffle(kg)
    num_recipes = len(set(kg[:, 0]))
    test_start = int((1-split)*kg.shape[0])
    # we have to ensure that all entities are present in the training set 
    while (len(set(kg[:test_start,0])) < num_recipes):
        np.random.shuffle(kg)
    kg_train = kg[:test_start]
    kg_test = kg[test_start:]
    return kg_train , kg_test


kg_train, kg_test = split_kg(kg, split= 0.3)
#%%
#kg_train, kg_valid = split_kg(kg_train, split = 0.25)
kg_valid = kg_test[:int(0.5*kg_test.shape[0])]
kg_test = kg_test[int(0.5*kg_test.shape[0]):]
print("number of triples in training set: ", kg_train.shape[0])
print("number of triples in validation set: ", kg_valid.shape[0])
print("number of triples in test set: ", kg_test.shape[0])

#%%
# write the train, valid and test data to txt.pickle files
with open(path + '/train.txt.pickle', 'wb') as f:
    pickle.dump(kg_train, f)
with open(path + '/valid.txt.pickle', 'wb') as f:
    pickle.dump(kg_valid, f)
with open(path + '/test.txt.pickle', 'wb') as f:
    pickle.dump(kg_test, f)
# %%
files = ['train.txt.pickle', 'valid.txt.pickle', 'test.txt.pickle']
entities, relations = set(), set()
for f in files:
    file_path = os.path.join(path, f)
    with open(file_path, 'rb') as f:
        to_read = pickle.load(f)
        for line in to_read:
            #lhs, rel, rhs = str(line[0]), str(line[1]), str(line[2])
            lhs, rel, rhs = (line[0]), (line[1]), (line[2])
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            #relations.add(rel+1)
            #relations.add(rel+'_reverse')
entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
id_to_entities = {i:x for (i, x) in enumerate(sorted(entities))}
id_to_relations = {i:x for (i, x) in enumerate(sorted(relations))}


n_relations = len(relations_to_id)
n_entities = len(entities_to_id)
print(f'{n_entities} entities and {n_relations} relations')

# %%
for (dic, f) in zip([entities_to_id, relations_to_id, id_to_entities, id_to_relations], ['ent_id', 'rel_id', 'id_ent', 'id_rel']):
    pickle.dump(dic, open(os.path.join(path, f'{f}.pickle'), 'wb'))

# %%
# compensation for enities not being in order is taken place here

to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}

for file in files:
    file_path = os.path.join(path, file)
    with open(file_path, 'rb') as f:
        to_read = pickle.load(f)
        examples = []
        for line in to_read:
            lhs, rel, rhs = (line[0]), (line[1]), (line[2])
            lhs_id = entities_to_id[lhs]
            rhs_id = entities_to_id[rhs]
            rel_id = relations_to_id[rel]
            examples.append([lhs_id, rel_id, rhs_id])
            # for the inverse case
            #examples.append([rhs_id, rel_id+1, lhs_id])
            to_skip['rhs'][(lhs_id, rel_id)].add(rhs_id)
            # for the inverse case
            #to_skip['lhs'][(rhs_id, rel_id+1)].add(lhs_id)

            to_skip['lhs'][(rhs_id, rel_id)].add(lhs_id)
            
    out = open(os.path.join(path,'new'+ file), 'wb')
    pickle.dump(np.array(examples).astype('uint64'), out)
    out.close()

to_skip_final = {'lhs': {}, 'rhs': {}}
for kk, skip in to_skip.items():
    for k, v in skip.items():
        to_skip_final[kk][k] = sorted(list(v))
out = open(os.path.join(path, 'new_to_skip.pickle'), 'wb')
pickle.dump(to_skip_final, out)
out.close()
# %%
# remember to replace "train.txt.pickle" with "new_train.txt.pickle" in the folder(same for valid and test)
