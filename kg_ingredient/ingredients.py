import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from nltk.stem import WordNetLemmatizer
from flashtext import KeywordProcessor

in_dir = Path('in')
out_dir = Path('out')
out_dir.mkdir(exist_ok=True)

wnl = WordNetLemmatizer()

# read KG.txt
kg_f = in_dir/'KG.txt'
with open(str(kg_f), 'r') as f:
    triples = f.readlines()

# ingredient -> count
ingredients_count = defaultdict(int)

for triple in triples:
    ingredient = triple.split(',')[-1].strip() 
    ingredient = ingredient.split('%20')
    ingredient = [wnl.lemmatize(word) for word in ingredient]
    ingredient = ' '.join(ingredient)

    ingredients_count[ingredient] += 1

# unique ingredients
ingredients_set = list(ingredients_count.keys())

# flashtext 
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(ingredients_set)

# pandas
df = pd.DataFrame.from_dict(ingredients_count, orient='index', columns=['Count'])
df = df.sort_values(by='Count', ascending=False)

df_f = out_dir/'KG_ingredients_count.csv'
df.to_csv(str(df_f), index_label='Ingredient')

# # median 
# median = df['Count'].median() # median is 2
# median_ingredients = list(df.loc[df['Count'] == median].index)
# print(median)
# print(median_ingredients)

# read 500QA.json
mpr_f = in_dir/'500QA.json'
with open(str(mpr_f), 'r') as f:
    mpr_data = json.load(f)

# find KG ingredients in each query 
for data in mpr_data:
    query = data['query']
    query = query.split()
    query = [wnl.lemmatize(word) for word in query]
    query = ' '.join(query)

    ingredients = keyword_processor.extract_keywords(query) 
    data['KG_ingredients'] = ingredients 

mpr_out_f = out_dir/'500QA_KG_ingredients.json'
with open(str(mpr_out_f), 'w') as f:
    json.dump(mpr_data, f, indent=4)