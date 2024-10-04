import os.path as osp
import argparse
import pickle
import json
import sys
import os
import requests

class WikidataRetriever:
    def __init__(self, entity_name, dataset):
        self.entity_name = entity_name
        self.dataset = dataset





        
    def get_wikidata_description(entity_name_or_id, name=True):
        if name:
            search_url = "https://www.wikidata.org/w/api.php"
            search_params = {
                'action': 'wbsearchentities',
                'format': 'json',
                'language': 'en',
                'search': entity_name_or_id
            }
            response = requests.get(search_url, params=search_params)
            if response.status_code != 200:
                return "Failed to find page"
            search_results = response.json()['search']
            if not search_results:
                return "No results found for this entity"
            entity_id = search_results[0]['id']
        else:
            entity_id = entity_name_or_id
        entity_url = "https://www.wikidata.org/wiki/Special:EntityData/{}.json".format(entity_id)
        response = requests.get(entity_url)
        if response.status_code != 200:
            return "Failed to get entity data"
        entity_info = response.json()['entities'][entity_id]
        if not entity_info.get('claims'):
            return "No claims found for this entity"
        properties_to_remove = [ 'P18', 'P373', 'P6375', 'P1442', 'P1476', 'P625', 'P646', 'P18',
         'P373', 'P6375', 'P1442', 'P1476', 'P625', 'P947', 'P1343', 'P227', 'P214', 'P268', 'P269', 'P349', 'P244',
         'P213', 'P691', 'P906', 'P245', 'P409', 'P910', 'P935', 'P1006', 'P949', 'P1017', 'P1005', 'P950', 'P1273', 'P1207','P3744', 'P2002'
         ]
        for prop in properties_to_remove:
            if prop in entity_info['claims']:
                del entity_info['claims'][prop]
        filtered_labels = {}
        for lang, label in entity_info['labels'].items():
            if lang == 'en':
                filtered_labels[lang] = label['value']
        filtered_entity_info = {
            'labels': filtered_labels,
            'claims': entity_info['claims']
        }
        try:
            description = entity_info['descriptions']['en']['value']
        except:
            description = 'No description found'

        return description




    def get_wikidata_facts(entity_name_or_id, name=True):
        if name:
            search_url = "https://www.wikidata.org/w/api.php"
            search_params = {
                'action': 'wbsearchentities',
                'format': 'json',
                'language': 'en',
                'search': entity_name_or_id
            }
            response = requests.get(search_url, params=search_params)
            if response.status_code != 200:
                return "Failed to find page"
            try:
                search_results = response.json()['search']
                if not search_results:
                    return "No results found for this entity"
            except:
                return "No results found for this entity"
            entity_id = search_results[0]['id']
        else:
            entity_id = entity_name_or_id
        tail_id2labels = {}
        entity_url = "https://www.wikidata.org/wiki/Special:EntityData/{}.json".format(entity_id)
        response = requests.get(entity_url)
        if response.status_code != 200:
            return "Failed to get entity data"
        entity_info = response.json()['entities'][entity_id]
        if not entity_info.get('claims'):
            return "No claims found for this entity"
        properties_to_remove = [ 'P18', 'P373', 'P6375', 'P1442', 'P1476', 'P625', 'P646', 'P18',
         'P373', 'P6375', 'P1442', 'P1476', 'P625', 'P947', 'P1343', 'P227', 'P214', 'P268', 'P269', 'P349', 'P244',
         'P213', 'P691', 'P906', 'P245', 'P409', 'P910', 'P935', 'P1006', 'P949', 'P1017', 'P1005', 'P950', 'P1273', 'P1207','P3744', 'P2002'
         ]
        for prop in properties_to_remove:
            if prop in entity_info['claims']:
                del entity_info['claims'][prop]
        filtered_labels = {}
        for lang, label in entity_info['labels'].items():
            if lang == 'en':
                filtered_labels[lang] = label['value']
        filtered_entity_info = {
            'labels': filtered_labels,
            'claims': entity_info['claims']
        }
        try:
            description = entity_info['descriptions']['en']['value']
        except:
            description = 'No description found'
        try:
            head = filtered_entity_info['labels']['en']
        except:
            return [], {}
        kept_props = {}
        hr_triples = []
        final_triples = []

        hr_triples.append((head, 'is', description))

        for kept_prop, values in filtered_entity_info['claims'].items():

            kept_prop_url = f'https://www.wikidata.org/w/api.php?action=wbgetentities&ids={kept_prop}&format=json&languages=en&props=labels'
            hr_response = requests.get(kept_prop_url)
            hr_rel = hr_response.json()['entities'][kept_prop]['labels']['en']['value']
            if ' ID' in hr_rel:
                continue
            kept_props[hr_rel] = kept_prop
        

            for value in values:

            
                try:
                    tail_label = value['mainsnak']['datavalue']['value']['id']
                except:
                    if value['mainsnak']['snaktype'] == 'novalue':
                        continue
                    else:
                        try:
                            tail_label = value['mainsnak']['datavalue']['value']
                        except:
                            continue

                tail_url = f'https://www.wikidata.org/w/api.php?action=wbgetentities&ids={tail_label}&format=json&languages=en&props=labels'
                tail_response = requests.get(tail_url)
                try:
                    hr_tail = tail_response.json()['entities'][tail_label]['labels']['en']['value']
                except:
                    try:
                        hr_tail = tail_response.json()['error']['id']
                    except:
                        continue
                if hr_tail not in tail_id2labels:
                    tail_id2labels[hr_tail] = tail_label

                # adding qualifiers that are for when we have additional information about the triple tail
                if 'qualifiers' in value:
                    for qualifier_prop in list(value['qualifiers'].keys()):
                        qualifier_prop_url = f'https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qualifier_prop}&format=json&languages=en&props=labels'
                        hr_response = requests.get(qualifier_prop_url)
                        hr_qualifier_rel = hr_response.json()['entities'][qualifier_prop]['labels']['en']['value']
                        try:
                            qualifier_tail = value['qualifiers'][qualifier_prop][0]['datavalue']['value']['id']
                        except:
                            if value['qualifiers'][qualifier_prop][0]['snaktype'] == 'novalue':
                                continue
                            else:
                                try:
                                    qualifier_tail = value['qualifiers'][qualifier_prop][0]['datavalue']['value']
                                except:
                                    continue
                        qualifier_tail_url = f'https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qualifier_tail}&format=json&languages=en&props=labels'
                        qualifier_tail_response = requests.get(qualifier_tail_url)
                        try:
                            hr_qualifier_tail = qualifier_tail_response.json()['entities'][qualifier_tail]['labels']['en']['value']
                        except:
                            try:
                                hr_qualifier_tail = qualifier_tail_response.json()['error']['id']
                            except:
                                continue
                        hr_triples.append((head, hr_rel, hr_tail, hr_qualifier_rel, hr_qualifier_tail))
                else:
                    hr_triples.append((head, hr_rel, hr_tail))
        for triple in hr_triples:
            final_triples.append(str(triple).replace("\\", "").replace("'", ""))
        return final_triples, tail_id2labels
    
    def write_to_file(facts, name):
        with open(f'{name}.txt', 'w', encoding="utf-8") as f:
            for fact in facts:
                f.write(str(fact) + '\n')
    
    