import ast
from kbc.wikidata import Wikidata
from kbc.WDRetriever import WikidataRetriever
def truncate_to_last_occurrence(input_string, delimiter):
    last_occurrence_index = input_string.rfind(delimiter)
    if last_occurrence_index == -1:
        last_occurrence_index = input_string.rfind(")\"")

    truncated_string = input_string[:last_occurrence_index]
    return truncated_string

def truncate_string_to_words(original_facts, max_words):
    
    words = str(original_facts).split()
    truncated_words = words[:max_words]
    truncated_string = ' '.join(truncated_words)
    
    if truncated_string[1]=="\'":
        processed_string = truncate_to_last_occurrence(truncated_string, ")\'")
        processed_string += ")\']"
    else:
        processed_string = truncate_to_last_occurrence(truncated_string, ")\"")
        processed_string += ")\"]"
    #processed_string += ")\']"

    try:
        facts_string = ast.literal_eval(truncated_string)
    except:
        facts_string = ast.literal_eval(processed_string)
    facts = list(facts_string)
    return facts
class KGassistant:

    def __init__(self, dataset, agent, question):
        self.question = question
        self.dataset = dataset
        self.reasoner = agent

        self.wikidata = Wikidata()


    def description_finder(self, entities, name=True, label=None):
        entity2descriptions = {}
        input_name = name
        for entity in entities:
            if entity.startswith("Q") and entity[1:].isdigit():
                name = False
            else:
                name = input_name
            description = self.wikidata.get_description(entity, name)

            if description:
                if not name:
                    label = None
                    if label is None:
                        label = self.wikidata._id_to_label(entity)
                    entity2descriptions[label] = description
                if name:
                    entity2descriptions[entity] = description

        return entity2descriptions
    
    def facts_finder(self, entities, name=True, label=None, truncated=True):
        entities2facts = {}
        entities2tail_labels = {}
        input_name = name
        for entity in entities:
            
            if entity.startswith("Q") and entity[1:].isdigit():
                name = False
            else:
                name = input_name
            
            if not name:
                label = None
                if label is None:
                    label = self.wikidata._id_to_label(entity)
                try:
                    entities2facts[label], entities2tail_labels[label] = self.wikidata.get_triples(entity, is_label=name)
                except:
                    continue
                facts_str = self.wikidata.verbalize_triples(entities2facts[label])
                if truncated:
                    entities2facts[label] = truncate_string_to_words(facts_str, 100)
                else:
                    entities2facts[label] = facts_str

            
            if name:
                try:
                    entities2facts[entity], entities2tail_labels[entity] = self.wikidata.get_triples(entity, is_label=name)
                except:
                    continue

                facts_str = self.wikidata.verbalize_triples(entities2facts[entity])
                if truncated:
                    entities2facts[entity] = truncate_string_to_words(facts_str, 100)
                else:
                    entities2facts[entity] = facts_str

        return entities2facts, entities2tail_labels

    def qid2labels(self, selected_entities):
        selected_entities_names = set()
        qids = set()

        for ent in selected_entities:
            name = self.wikidata._id_to_label(ent)
            if name is None:
                name = ent
                qid = self.wikidata._label_to_id(ent)
                # invalid entity picked by llm
                if qid is None:
                    continue
                qids.add(qid)
            else:
                qids.add(ent)
            selected_entities_names.add(name)
                
        
        return list(selected_entities_names)
