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
import csv

fb_wd = {}
with open('wikidata_freebase.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if len(row) == 2:
            wikidata_id = row[0].strip()
            freebase_id = row[1].strip()
            fb_wd[freebase_id] = wikidata_id

#%%
# head_counts = {}

# with open('Freebase/train.txt', 'r') as file:
#     for line in file:
#         parts = line.strip().split('\t')
#         if len(parts) == 3:
#             h, r, t = parts
#             if (r,t) in head_counts:
#                 head_counts[(r,t)] += 1
#             else:
#                 head_counts[(r,t)] = 1

# # save the head counts as a pickle file
# with open('Freebase/head_counts.pkl', 'wb') as file:
#     pickle.dump(head_counts, file)


#make a dictionary of entities and their connections for multi-hop queries
# ent_connections = {}
# with open('Freebase/train.txt', 'r') as file:
#     for line in file:
#         parts = line.strip().split('\t')
#         if len(parts) == 3:
#             h, r, t = parts
#             if h in ent_connections:
#                 ent_connections[h].append((r,t))
#             else:
#                 ent_connections[h] = [(r,t)]
# # save the head counts as a pickle file
# with open('Freebase/ent_connections.pkl', 'wb') as file:
#     pickle.dump(ent_connections, file)



#%%
# load the head counts from the pickle file
with open('Freebase/head_counts.pkl', 'rb') as file:
    head_counts = pickle.load(file)
    

with open('Freebase/ent_connections.pkl', 'rb') as file:
    ent_connections = pickle.load(file)



#%%
import requests, time
query_type = '1p'
num_queries = 30

CN_facts_threshold = 5

heads_threshold = 10

queries = {}
t1 = time.time()

selected_rels_counter = {}
selected_rels_threshold = 9

with open('Freebase/train.txt', 'r') as file:
    for line_no, line in enumerate(file):
        # if len(queries) == num_queries:
        #     break
        if line_no % 10000 == 0 and line_no > 0:
            print(f"triples processed: {line_no}")
            if line_no % 100000 == 0:
                break

        parts = line.strip().split('\t')
        if len(parts) == 3:
            h, r, t = parts

            # if there are more than heads_threshold heads for the relation, we skip this relation
            if (r,t) in head_counts and head_counts[(r,t)] > heads_threshold:
                continue
            if r in selected_rels_counter:
                if selected_rels_counter[r] > selected_rels_threshold:
                    continue
            try: 
                h_mapped = fb_wd[h]
            except:
                continue
            wikidata_entity_id = h_mapped.strip().split('/')[-1]

            wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_entity_id}.json"
            # Send a GET request to the Wikidata API 
            
            response = requests.get(wikidata_api_url)

            # response was successful
            if response.status_code == 200:

                data = response.json()
                # Extract the label (title) from the JSON data (e.g., Dominican Republic)
                try:
                    label = data["entities"][wikidata_entity_id]["labels"]["en"]["value"]
                except:
                    continue
                # Finding "instance of" claims from wikidata (e.g., country, sovereign state, etc.)
                claims = data["entities"][wikidata_entity_id]["claims"]
                instance_of_qid = []
                if "P31" in claims:
                    for claim in claims["P31"]:
                        instance_of_qid.append(claim["mainsnak"]["datavalue"]["value"]["id"])
                
                # converting labels from Qids to english labels
                instance_of = []
                for instance_qid in instance_of_qid:
                    wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{instance_qid}.json"
                    response_instance = requests.get(wikidata_api_url)
                    if response_instance.status_code == 200:
                        instance_data = response_instance.json()
                        try:
                            instance = instance_data["entities"][instance_qid]["labels"]["en"]["value"].lower()
                            instance_of.append(instance)
                        except:
                            continue


                # now we have the "instace_of" facts which can be used to query from conceptnet
                CN_facts = {}

                head_relations = ['isa', 'hasa', 'usedfor', 'capableof', 'hasproperty', 'madeof']
                tail_relations = ['atlocation', 'partof', ]
                # CN_facts['isa'] = {}; CN_facts['locatedat'] = {}; CN_facts['partof'] = {}; CN_facts['hasa'] = {}
                # CN_facts['usedfor'] = {}; CN_facts['capableof'] = {}; CN_facts['causes'] = {}; CN_facts['hasproperty'] = {}
                # CN_facts['madeof'] = {}
                
                for concept in instance_of:
                    concept = concept.replace(" ", "_")
                    
                    
                    # get the relevant facts from conceptnet(we need to make a 1 second sleep among requests)
                    conceptnet_api_url = f"http://api.conceptnet.io/c/en/{concept}?offset=0&limit=100000"
                    
                    time.sleep(1)
                    while True:
                        try:
                            response = requests.get(conceptnet_api_url)
                            break
                        except:
                            time.sleep(0.5)
                            continue
                    if response.status_code == 200:
                        data = response.json()
                        for edge in data["edges"]:
                            if "rel" in edge:
                                relation = edge["rel"]["label"]
                                # these are "IsA" relations where the head is the concept (e.g, country, IsA, administrative district)
                                if relation.lower() in head_relations:
                                    if concept == edge['start']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower():
                                        if edge['start']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower() != edge['end']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower():
                                            CN_facts[(edge['start']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower(), relation.lower() ,edge['end']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower())] = edge['weight']
                                # these are "AtLocation" relations where the tail is the concept (e.g, a capital, AtLocation, country)
                                elif relation.lower() in tail_relations:
                                        
                                    if concept == (edge['end']['label']).strip().removeprefix('an ').removeprefix('a ').strip().lower():
                                        CN_facts[(edge['start']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower(), relation.lower(), edge['end']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower())] = edge['weight']

                        if len(CN_facts) < CN_facts_threshold:
                            CN_facts = {}
                            continue
                        else: 
                            break
                          

                    else:
                        continue
                if len(CN_facts) < CN_facts_threshold:
                    CN_facts = {}
                    continue

                if query_type == '1p':
                    goal = h
                    hr_goal = label
                    anchor = t
                    try:
                        mapped_anchor = fb_wd[t]
                    except:
                        continue
                    wikidata_id_anchor = mapped_anchor.strip().split('/')[-1]

                    wikidata_api_url_anchor = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id_anchor}.json"
                    anchor_response = requests.get(wikidata_api_url_anchor)
                    if anchor_response.status_code == 200:
                        anchor_data = anchor_response.json()
                        hr_anchor = anchor_data["entities"][wikidata_id_anchor]["labels"]["en"]["value"]

                        CN_sorted = dict(sorted(CN_facts.items(), key=lambda item: item[1], reverse=True))
                        CN_selected = list(CN_sorted.items())[:CN_facts_threshold]

                        # keeping track of how many queries selected for each relation
                        if r in selected_rels_counter:
                            selected_rels_counter[r] += 1
                        else:
                            selected_rels_counter[r] = 1

                        if len(queries) % 100 == 0 and len(queries) > 0:
                            print(f"Queries generated: {len(queries)}")

                        new_query = {}
                        new_query['query type'] = query_type
                        new_query['answer entities'] = goal; new_query['answer'] = hr_goal
                        new_query['variable entities'] = []; new_query['variables'] = []
                        new_query['anchor entities'] = [anchor]; new_query['anchors'] = [hr_anchor]
                        new_query['CN_facts'] = CN_selected; 
                        new_query['relations'] = [r]
                        new_query['triples'] = (h, r, t)
                        new_query['triples readable'] = (hr_goal, r, hr_anchor)
                        queries[f"query {len(queries)}"] = new_query
                        with open('readable_queries_sofar.txt', 'a') as text_file:
                            text_file.write(str(new_query['triples readable'])+ '\n')

                elif query_type == '2p':
                    goal = h
                    hr_goal = label
                    var = t
                    if var not in ent_connections:
                        continue
                    r2 = ent_connections[var][0][0]
                    
                    # avoiding the case where var and anchor are the same
                    i = 0
                    try:
                        while i > -1:
                            anchor = ent_connections[var][i][1]
                            i += 1
                            if anchor != var:
                                mapped_var = fb_wd[t]
                                mapped_anchor = fb_wd[anchor]
                                break


                    except:
                        continue


                    wikidata_id_var = mapped_var.strip().split('/')[-1]
                    wikidata_id_anchor = mapped_anchor.strip().split('/')[-1]

                    wikidata_api_url_var = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id_var}.json"
                    wikidata_api_url_anchor = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id_anchor}.json"
                    var_response = requests.get(wikidata_api_url_var)
                    anchor_response = requests.get(wikidata_api_url_anchor)

                    if var_response.status_code == 200 and anchor_response.status_code == 200:
                        var_data = var_response.json(); anchor_data = anchor_response.json()
                        hr_var = var_data["entities"][wikidata_id_var]["labels"]["en"]["value"]
                        hr_anchor = anchor_data["entities"][wikidata_id_anchor]["labels"]["en"]["value"]

                        IsA_sorted = dict(sorted(IsA_facts.items(), key=lambda item: item[1], reverse=True))
                        AtLocation_sorted = dict(sorted(AtLocation_facts.items(), key=lambda item: item[1], reverse=True))

                        IsA_selected = list(IsA_sorted.items())[:CN_IsA_facts]
                        AtLocation_selected = list(AtLocation_sorted.items())[:CN_AtLocation_facts]

                        new_query = {}
                        new_query['query type'] = query_type
                        new_query['answer entity'] = goal; new_query['answer'] = hr_goal; 
                        new_query['variable entities'] = [var]; new_query['variables'] = [hr_var]
                        new_query['anchor entities'] = [anchor]; new_query['anchors'] = [hr_anchor]
                        new_query['IsA'] = IsA_selected; new_query['AtLocation'] = AtLocation_selected
                        new_query['relations'] = [r, r2]
                        new_query['triples'] = (h, r, t, r2, anchor)
                        new_query['triples readable'] = (hr_goal, r, hr_var, r2, hr_anchor)
                        queries[f"query {len(queries)}"] = new_query
                        with open('readable_queries_sofar.txt', 'a') as text_file:
                            text_file.write(str(new_query['triples readable'])+ '\n')




                    

t2 = time.time()
print("time elapsed:", t2-t1)
# print(queries)

with open('queries.json', 'w') as json_file:
    json.dump(queries, json_file)





#%%

#%%
# import requests, time
# query_type = '1p'
# num_queries = 1000

# CN_IsA_facts = 2
# CN_AtLocation_facts = 2

# heads_threshold = 10

# queries = {}
# t1 = time.time()

# with open('Freebase/train.txt', 'r') as file:
#     for line in file:
#         if len(queries) == num_queries:
#             break
#         if len(queries) % 30 == 0 and len(queries) > 0:
#             print(f"Queries generated: {len(queries)}")
#         parts = line.strip().split('\t')
#         if len(parts) == 3:
#             h, r, t = parts

#             # if there are more than heads_threshold heads for the relation, we skip this relation
#             if (r,t) in head_counts and head_counts[(r,t)] > heads_threshold:
#                 continue
#             try: 
#                 h_mapped = fb_wd[h]
#             except:
#                 continue
#             wikidata_entity_id = h_mapped.strip().split('/')[-1]

#             wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_entity_id}.json"
#             # Send a GET request to the Wikidata API 
            
#             response = requests.get(wikidata_api_url)

#             # response was successful
#             if response.status_code == 200:

#                 data = response.json()
#                 # Extract the label (title) from the JSON data (e.g., Dominican Republic)
#                 try:
#                     label = data["entities"][wikidata_entity_id]["labels"]["en"]["value"]
#                 except:
#                     continue
#                 # Finding "instance of" claims from wikidata (e.g., country, sovereign state, etc.)
#                 claims = data["entities"][wikidata_entity_id]["claims"]
#                 instance_of_qid = []
#                 if "P31" in claims:
#                     for claim in claims["P31"]:
#                         instance_of_qid.append(claim["mainsnak"]["datavalue"]["value"]["id"])
                
#                 # converting labels from Qids to english labels
#                 instance_of = []
#                 for instance_qid in instance_of_qid:
#                     wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{instance_qid}.json"
#                     response_instance = requests.get(wikidata_api_url)
#                     if response_instance.status_code == 200:
#                         instance_data = response_instance.json()
#                         try:
#                             instance = instance_data["entities"][instance_qid]["labels"]["en"]["value"].lower()
#                             instance_of.append(instance)
#                         except:
#                             continue


#                 # now we have the "instace_of" facts which can be used to query from conceptnet
#                 IsA_facts = {}
#                 AtLocation_facts = {}
#                 for concept in instance_of:
#                     concept = concept.replace(" ", "_")
                    
                    
#                     # get the relevant facts from conceptnet(we need to make a 1 second sleep among requests)
#                     conceptnet_api_url = f"http://api.conceptnet.io/c/en/{concept}?offset=0&limit=100000"
                    
#                     time.sleep(1)
#                     while True:
#                         try:
#                             response = requests.get(conceptnet_api_url)
#                             break
#                         except:
#                             time.sleep(0.5)
#                             continue
#                     if response.status_code == 200:
#                         data = response.json()
#                         for edge in data["edges"]:
#                             if "rel" in edge:
#                                 relation = edge["rel"]["label"]
#                                 # these are "IsA" relations where the head is the concept (e.g, country, IsA, administrative district)
#                                 if relation.lower() == "isa":
#                                     if concept == edge['start']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower():
#                                         if edge['start']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower() != edge['end']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower():
#                                             IsA_facts[(edge['start']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower(), edge['end']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower())] = edge['weight']

#                                 # these are "AtLocation" relations where the tail is the concept (e.g, a capital, AtLocation, country)
#                                 elif relation.lower() == "atlocation":
                                        
#                                     if concept == (edge['end']['label']).strip().removeprefix('an ').removeprefix('a ').strip().lower():
#                                         AtLocation_facts[(edge['start']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower(), edge['end']['label'].strip().removeprefix('an ').removeprefix('a ').strip().lower())] = edge['weight']

                                    
#                         if len(IsA_facts) < CN_IsA_facts or len(AtLocation_facts) < CN_AtLocation_facts:
#                             IsA_facts = {}
#                             AtLocation_facts = {}
#                             continue
#                         else: 
#                             break
                          

#                     else:
#                         continue
#                 if len(IsA_facts) < CN_IsA_facts or len(AtLocation_facts) < CN_AtLocation_facts:
#                     IsA_facts = {}
#                     AtLocation_facts = {}
#                     continue

#                 if query_type == '1p':
#                     goal = h
#                     hr_goal = label
#                     anchor = t
#                     try:
#                         mapped_anchor = fb_wd[t]
#                     except:
#                         continue
#                     wikidata_id_anchor = mapped_anchor.strip().split('/')[-1]

#                     wikidata_api_url_anchor = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id_anchor}.json"
#                     anchor_response = requests.get(wikidata_api_url_anchor)
#                     if anchor_response.status_code == 200:
#                         anchor_data = anchor_response.json()
#                         hr_anchor = anchor_data["entities"][wikidata_id_anchor]["labels"]["en"]["value"]

#                         IsA_sorted = dict(sorted(IsA_facts.items(), key=lambda item: item[1], reverse=True))
#                         AtLocation_sorted = dict(sorted(AtLocation_facts.items(), key=lambda item: item[1], reverse=True))

#                         IsA_selected = list(IsA_sorted.items())[:CN_IsA_facts]
#                         AtLocation_selected = list(AtLocation_sorted.items())[:CN_AtLocation_facts]

#                         new_query = {}
#                         new_query['query type'] = query_type
#                         new_query['answer entities'] = goal; new_query['answer'] = hr_goal
#                         new_query['variable entities'] = []; new_query['variables'] = []
#                         new_query['anchor entities'] = [anchor]; new_query['anchors'] = [hr_anchor]
#                         new_query['IsA'] = IsA_selected; new_query['AtLocation'] = AtLocation_selected; 
#                         new_query['relations'] = [r]
#                         new_query['triples'] = (h, r, t)
#                         new_query['triples readable'] = (hr_goal, r, hr_anchor)
#                         queries[f"query {len(queries)}"] = new_query
#                         with open('readable_queries_sofar.txt', 'a') as text_file:
#                             text_file.write(str(new_query['triples readable'])+ '\n')

#                 elif query_type == '2p':
#                     goal = h
#                     hr_goal = label
#                     var = t
#                     if var not in ent_connections:
#                         continue
#                     r2 = ent_connections[var][0][0]
                    
#                     # avoiding the case where var and anchor are the same
#                     i = 0
#                     try:
#                         while i > -1:
#                             anchor = ent_connections[var][i][1]
#                             i += 1
#                             if anchor != var:
#                                 mapped_var = fb_wd[t]
#                                 mapped_anchor = fb_wd[anchor]
#                                 break


#                     except:
#                         continue


#                     wikidata_id_var = mapped_var.strip().split('/')[-1]
#                     wikidata_id_anchor = mapped_anchor.strip().split('/')[-1]

#                     wikidata_api_url_var = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id_var}.json"
#                     wikidata_api_url_anchor = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id_anchor}.json"
#                     var_response = requests.get(wikidata_api_url_var)
#                     anchor_response = requests.get(wikidata_api_url_anchor)

#                     if var_response.status_code == 200 and anchor_response.status_code == 200:
#                         var_data = var_response.json(); anchor_data = anchor_response.json()
#                         hr_var = var_data["entities"][wikidata_id_var]["labels"]["en"]["value"]
#                         hr_anchor = anchor_data["entities"][wikidata_id_anchor]["labels"]["en"]["value"]

#                         IsA_sorted = dict(sorted(IsA_facts.items(), key=lambda item: item[1], reverse=True))
#                         AtLocation_sorted = dict(sorted(AtLocation_facts.items(), key=lambda item: item[1], reverse=True))

#                         IsA_selected = list(IsA_sorted.items())[:CN_IsA_facts]
#                         AtLocation_selected = list(AtLocation_sorted.items())[:CN_AtLocation_facts]

#                         new_query = {}
#                         new_query['query type'] = query_type
#                         new_query['answer entity'] = goal; new_query['answer'] = hr_goal; 
#                         new_query['variable entities'] = [var]; new_query['variables'] = [hr_var]
#                         new_query['anchor entities'] = [anchor]; new_query['anchors'] = [hr_anchor]
#                         new_query['IsA'] = IsA_selected; new_query['AtLocation'] = AtLocation_selected
#                         new_query['relations'] = [r, r2]
#                         new_query['triples'] = (h, r, t, r2, anchor)
#                         new_query['triples readable'] = (hr_goal, r, hr_var, r2, hr_anchor)
#                         queries[f"query {len(queries)}"] = new_query
#                         with open('readable_queries_sofar.txt', 'a') as text_file:
#                             text_file.write(str(new_query['triples readable'])+ '\n')




                    

# t2 = time.time()
# print("time elapsed:", t2-t1)
# # print(queries)

# with open('queries.json', 'w') as json_file:
#     json.dump(queries, json_file)



# %%



# import requests, time
# query_type = '1p'
# num_queries = 500

# CN_facts = 5

# heads_threshold = 10

# queries = {}
# t1 = time.time()

# with open('Freebase/train.txt', 'r') as file:
#     for line in file:
#         if len(queries) == num_queries:
#             break
#         if len(queries) % 10 == 0 and len(queries) > 0:
#             print(f"Queries generated: {len(queries)}")
#         parts = line.strip().split('\t')
#         if len(parts) == 3:
#             h, r, t = parts

#             # if there are more than heads_threshold heads for the relation, we skip this relation
#             if (r,t) in head_counts and head_counts[(r,t)] > heads_threshold:
#                 continue
#             try: 
#                 h_mapped = fb_wd[h]
#             except:
#                 continue
#             wikidata_entity_id = h_mapped.strip().split('/')[-1]

#             wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_entity_id}.json"
#             # Send a GET request to the Wikidata API 
            
#             response = requests.get(wikidata_api_url)

#             # response was successful
#             if response.status_code == 200:

#                 data = response.json()
#                 # Extract the label (title) from the JSON data (e.g., Dominican Republic)
#                 try:
#                     label = data["entities"][wikidata_entity_id]["labels"]["en"]["value"]
#                 except:
#                     continue
#                 # Finding "instance of" claims from wikidata (e.g., country, sovereign state, etc.)
#                 claims = data["entities"][wikidata_entity_id]["claims"]
#                 instance_of_qid = []
#                 if "P31" in claims:
#                     for claim in claims["P31"]:
#                         instance_of_qid.append(claim["mainsnak"]["datavalue"]["value"]["id"])
                
#                 # converting labels from Qids to english labels
#                 instance_of = []
#                 for instance_qid in instance_of_qid:
#                     wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{instance_qid}.json"
#                     response_instance = requests.get(wikidata_api_url)
#                     if response_instance.status_code == 200:
#                         instance_data = response_instance.json()
#                         try:
#                             instance = instance_data["entities"][instance_qid]["labels"]["en"]["value"]
#                             instance_of.append(instance)
#                         except:
#                             continue

#                 # now we have the "instace_of" facts which can be used to query from conceptnet
#                 head_facts = {}
#                 tail_facts = {}
#                 for concept in instance_of:
                    
                    
#                     # get the relevant facts from conceptnet(we need to make a 1 second sleep among requests)
#                     conceptnet_api_url = f"http://api.conceptnet.io/c/en/{concept}?offset=0&limit=100000"
#                     time.sleep(1)
#                     response = requests.get(conceptnet_api_url)
#                     if response.status_code == 200:
#                         data = response.json()
#                         for edge in data["edges"]:
#                             if "rel" in edge:
#                                 relation = edge["rel"]["label"]
#                                 # these are "IsA" relations where the head is the concept (e.g, country, IsA, administrative district)
#                                 if relation.lower() != "synonym" and relation.lower() != "antonym" and relation.lower() != "relatedto":
#                                     if concept == edge['start']['label']:
                                       
#                                         head_facts[(edge['start']['label'], relation, edge['end']['label'])] = edge['weight']
                                   
#                                 # these are relations where the tail is the concept (e.g, a capital, AtLocation, country)
#                                 # we're not allowing "IsA" relations here (e.g, a Africa, IsA, country)
#                                 if relation.lower() != "synonym" and relation.lower() != "isa" and relation.lower() != "antonym" and relation.lower() != "relatedto":


#                                     if concept == (edge['end']['label']).strip().lstrip('a').strip():
#                                         tail_facts[(edge['start']['label'].strip().lstrip('a').strip(), relation, edge['end']['label'].strip().lstrip('a').strip())] = edge['weight']
#                                     if concept == edge['end']['label']:
#                                         tail_facts[(edge['start']['label'], relation, edge['end']['label'])] = edge['weight']
   
                        
#                         if len(tail_facts) < CN_facts or len(head_facts) < CN_facts:
#                             continue
#                         else: 
#                             break
                          

#                     else:
#                         continue
#                 if len(tail_facts) < CN_facts or len(head_facts) < CN_facts:
#                     continue

#                 if query_type == '1p':
#                     goal = h
#                     hr_goal = label
#                     anchor = t
#                     try:
#                         mapped_anchor = fb_wd[t]
#                     except:
#                         continue
#                     wikidata_id_anchor = mapped_anchor.strip().split('/')[-1]

#                     wikidata_api_url_anchor = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id_anchor}.json"
#                     anchor_response = requests.get(wikidata_api_url_anchor)
#                     if anchor_response.status_code == 200:
#                         anchor_data = anchor_response.json()
#                         hr_anchor = anchor_data["entities"][wikidata_id_anchor]["labels"]["en"]["value"]

#                         head_sorted = dict(sorted(head_facts.items(), key=lambda item: item[1], reverse=True))
#                         tail_sorted = dict(sorted(tail_facts.items(), key=lambda item: item[1], reverse=True))
#                         head_selected = list(head_sorted.items())[:CN_facts]
#                         tail_selected = list(tail_sorted.items())[:CN_facts]
#                         new_query = {}
#                         new_query['query type'] = query_type
#                         new_query['goal entity'] = goal; new_query['answer'] = hr_goal; new_query['anchor entity'] = anchor; new_query['answer anchor'] = hr_anchor
#                         new_query['head'] = head_selected; new_query['tail'] = tail_selected; 
#                         new_query['triples'] = (h, r, t)
#                         new_query['triples readable'] = (hr_goal, r, hr_anchor)
#                         queries[f"query {len(queries)}"] = new_query
#                         with open('readable_queries_sofar.txt', 'a') as text_file:
#                             text_file.write(str(new_query['triples readable'])+ '\n')

# t2 = time.time()
# print("time elapsed:", t2-t1)


# with open('queries.json', 'w') as json_file:
#     json.dump(queries, json_file)



# %%
with open('queries.json', 'r') as json_file:
    queries = json.load(json_file)

print(len(queries))

# %%
# with open("template questions.txt", 'w') as text_file:
#     for q in queries:
#         query = queries[q]
#         IsAs = []
#         AtLocs = []
#         for isA_fact in query['IsA']:
#             IsAs.append(isA_fact[0][1])
#         for Atlocation_fact in query['AtLocation']:
#             AtLocs.append(Atlocation_fact[0][0])

#         template_question = f'What is the name of a'
#         for i in range(len(IsAs)):
#             if i == 0:
#                 template_question += f' {IsAs[i]}'
#             else:
#                 template_question += f' and a {IsAs[i]}'

#         for i in range(len(AtLocs)):
#             if i == 0:
#                 template_question += f' with {AtLocs[i]} and'
#             elif i < len(AtLocs) - 1:
#                 template_question += f' {AtLocs[i]} and'
#             elif i == len(AtLocs) - 1:
#                 template_question += f' {AtLocs[i]} at it'

#         answer = query['triples readable'][0]
#         relation = query['triples readable'][1]
#         anchor = query['triples readable'][2]
#         template_question += f' that {relation} {anchor}?'
        
#         original_triple = f' (Original Triple: {answer}, {relation}, {anchor})) \n'
#         template_question += original_triple
#         text_file.write(template_question)


# text_file.close()




# %%
# %%
with open('queries_readable_rels.json', 'r') as json_file:
    queries_readable_rel = json.load(json_file)

print(len(queries_readable_rel))

with open("template questions.txt", 'w') as text_file:
    for q in queries_readable_rel:
        query = queries_readable_rel[q]

        template_question = f'What is the name of the entity that'
        for CN_fact in query['CN_facts']:
            head = CN_fact[0][0]
            rel = CN_fact[0][1]
            tail = CN_fact[0][2]
            if rel == 'hasa':
                template_question += f' has {tail} and'
            elif rel == 'capableof':
                template_question += f' is capable of {tail} and'
            elif rel == 'isa':
                template_question += f' is a {tail} and'
            elif rel == 'usedfor':
                template_question += f' is used for {tail} and'
            elif rel == 'atlocation':
                template_question += f'{tail} is located at it and'
            elif rel == 'partof':
                template_question += f' {head} is a part of it and'
            elif rel == 'hasproperty':
                template_question += f' has the property of {tail} and'
        # removing the excessive "and" at the end
        template_question = template_question[:-4]


        answer = query['triples readable'][0]
        relation = query['triples readable'][1].replace("'", '')
        anchor = query['triples readable'][2]

        template_question += f' that {relation} {anchor}?'
        original_triple = f' (Original Triple: {answer}, {relation}, {anchor})) \n'
        
        template_question += original_triple
        text_file.write(template_question)
text_file.close()





# %%

#CSKG experiment

# file_path = "cskg.tsv"
# with open("continents.txt", 'w') as text_file:

# # Open the file and read its lines
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#         # Check if the line contains the word "continent" but does not end with "CN"
#             if "continent" in line.lower() and not line.strip().endswith("CN"):
#                 text_file.write(line)
# text_file.close()
# %%

with open('queries.json', 'r') as json_file:
    queries = json.load(json_file)

print(len(queries))
counter = 0
insuff = set()
for q in queries:
    query = queries[q]
    CN_facts = query['CN_facts']
    weights = []
    for CN_fact in CN_facts:

        weights.append(CN_fact[1])
    #if max(weights) <= 2:
        counter += 1
        insuff.add(CN_facts[0][0][0])


print(counter)
print(len(insuff))
# %%
