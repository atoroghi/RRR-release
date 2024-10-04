import os, pickle, sys, time, re, copy
import numpy as np
import csv
import tqdm
import yaml
import openai
import requests
import argparse
import json
from requests.exceptions import ConnectionError
from kbc.WDRetriever import WikidataRetriever
import requests
from wikidataintegrator import wdi_core
import signal

openai.api_key_path = 'api_key.txt'

def get_response(message):
    tries = 5
    for i in range(tries):
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



def truncate_string_to_words(original_string, max_words):
    words = original_string.split()
    truncated_words = words[:max_words]
    truncated_string = ' '.join(truncated_words)
    return truncated_string


# Function to get QID for an entity
def get_entid(entity_label):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={entity_label}&language=en&format=json"
    response = requests.get(url)
    data = response.json()
    if len(data['search'])>0:
          return data['search'][0]['id']
    
    else:
          return None


# Function to get property ID for a property label
def get_property_id(property_label):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={property_label}&type=property&language=en&format=json"
    response = requests.get(url)
    data = response.json()
    
    # Check if search results are available
    if data['search']:
        # Iterate through results to find the correct property ID
        for result in data['search']:
            if 'description' in result and result['description'] == 'Wikimedia disambiguation page':
                continue  # Skip disambiguation pages
            if 'id' in result and result['id'].startswith('P'):
                return result['id']
    return None

# Function to replace entities and properties in the SPARQL query
def replace_entities_and_properties(query):
    property_pattern = r'wdt:\[([^]]*)\]'
    entity_pattern = r'wd:\[([^]]*)\]'

    properties = re.findall(property_pattern, query)
    entities = re.findall(entity_pattern, query)

    prop2ids = {}
    ent2ids = {}
    for prop_label in properties:
      prop2ids[prop_label] = get_property_id(prop_label)
    
    for ent_label in entities:
      ent2ids[ent_label] = get_entid(ent_label)

    for key, value in ent2ids.items():
      escaped_key = re.escape(key)
      ent_label_pattern = re.compile(r'\[' + escaped_key + r'\]')
      if value:
        query = re.sub(ent_label_pattern, value , query)

    for key2, value2 in prop2ids.items():
      escaped_key2 = re.escape(key2)
      prop_label_pattern = re.compile(r'\[' + escaped_key2 + r'\]')
      if value2:
        query = re.sub(prop_label_pattern, value2 , query)


    return query

def handler(signum, frame):
    raise TimeoutError("Execution timed out")
#@timeout(30, use_signals=False)
def run_sparql(modified_query):
    q_answers = []
    q_answer_labels = []
    result = wdi_core.WDFunctionsEngine.execute_sparql_query(query=modified_query)

    try:
        for binding in result['results']['bindings']:
            for key, value in binding.items():
                q_answers.append(value['value'])
            if 'answerLabel' in binding.keys():
                q_answer_labels.append( binding['answerLabel']['value'])
            else:
                q_answer_labels.append(q_answers[-1])
    except:
        return ["no additional information found for this query."]
    
    return q_answer_labels

def execute_run_sparql_with_timeout(query):
    try:
        return run_sparql(query)
    except Exception as e:
        
        return ["no additional information found for this query."]

def extract_first_number(input_string):
    # extracts the first number in a string(used to extract the sub-questions and the respective queries)
    pattern = r'\b\d+\b'
    matches = re.findall(pattern, input_string)
    if matches:
        return int(matches[0])
    else:
        return None
    

def log_to_csv(query_outcome, csv_file_path):
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(query_outcome)


def main():
    parser = argparse.ArgumentParser(description='Extract cooking methods from recipes.')
    parser.add_argument('path', help='Path to directory containing queries')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', default='StrategyQA', choices=['Recipe-MPR', 'StrategyQA', 'Creek'])
    parser.add_argument('--mode', type=str, help='Mode of the experiment', default='modified', choices=['original', 'modified'])
    args = parser.parse_args()



    dataset = args.dataset_name

    dataset_path = os.path.join(os.getcwd() , args.path, dataset)
    if dataset == 'Recipe-MPR':
        #data_path = os.path.join(dataset_path, '500QA.json')
        #with open(data_path, "r") as file:
        #    data = json.load(file)
        raise NotImplementedError
    elif dataset == 'StrategyQA':
        data_path = os.path.join(dataset_path, f'{args.dataset_name}_modified.csv')
        with open(data_path, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = list(csv_reader)

    answer_col_num = 3
    if args.mode == 'original':
        q_col_num = 1
    elif args.mode == 'modified':
        q_col_num = 2

    with open(os.path.join('prompts' , 'KBBinder_prompt.yaml'), 'r', encoding="utf-8") as yaml_file:
        yaml_str = yaml.load(yaml_file, Loader=yaml.FullLoader)

    SPARQL_prompt_template = yaml_str['SPARQL_prompt']
    Final_Answer_prompt_template = yaml_str['Final_Answer_prompt']
    

    output_file_path = os.path.join('results', f'KBBinder_{dataset}_{args.mode}')
    os.makedirs(output_file_path, exist_ok=True)

    query_counter = 0; unanswered_counter = 0; correct_counter = 0; 

    results_file_path = os.path.join(os.getcwd(), 'results', 'KBBinder', dataset, args.mode)

    last_answered_query = 0
    csv_file_path = os.path.join(results_file_path, 'Outcomes_Per_Query.csv')
    if os.path.exists(csv_file_path):
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            last_line = None
            for row in reader:
                last_line = row
            if last_line is not None:
                last_answered_query = int(last_line[0])
                print(f'Last previously-answered query: {last_answered_query}')

    for i, row in tqdm.tqdm(enumerate(data[last_answered_query+1:])):
        conversation_history = []

        if len(row) >= 4:
            current_q_num = i+1+last_answered_query
            query_counter += 1
            true_answer = (row[answer_col_num]).strip().lower().startswith("true")
            # obtaining the initial response
            question = row[q_col_num]
            conversation_history.append({"query ID": row[0]})
            conversation_history.append({"question": question})


            SPARQL_prompt = copy.deepcopy(SPARQL_prompt_template)
            SPARQL_prompt[-1]['input3'] = SPARQL_prompt[-1]['input3'].format(QUESTION = question)
            SPARQL_message = [{"role": "system", "content": SPARQL_prompt[0]['system']}, {"role": "user", "content":SPARQL_prompt[1]['input1']},
                              {"role": "assistant", "content": SPARQL_prompt[2]['output1']}, {"role": "user", "content":SPARQL_prompt[3]['input2']},
                              {"role": "assistant", "content": SPARQL_prompt[4]['output2']}, {"role": "user", "content":SPARQL_prompt[-1]['input3']}]
            conversation_history += SPARQL_message
            Generated_output = get_response(SPARQL_message)
            conversation_history.append({"role": "assistant", "content": Generated_output})

            additional_knowledge = {}

            for line in Generated_output.split('\n'):
                if line.startswith('Sub-question'):
                    #sub_questions.append(line[(line.find(':') + 1):].strip())
                    additional_knowledge[extract_first_number(line)] = {}
                    additional_knowledge[extract_first_number(line)]['sub_question'] = line[(line.find(':') + 1):].strip()
                elif line.startswith('SPARQL'):
                    #queries.append(line[(line.find(':') + 1):].strip())
                    additional_knowledge[extract_first_number(line)]['query'] = line[(line.find(':') + 1):].strip()

            

            #query_answers = {}
            #query_answer_labels = {}
            # replacing qids and pids and running queries

            for i, val in additional_knowledge.items():
                if 'query' in val.keys():
                    query = val['query']

                    modified_query = replace_entities_and_properties(query)
                    #q_answer_labels = run_sparql(modified_query)

                    timeout_seconds = 30
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(timeout_seconds)
                    try:
                        q_answer_labels = run_sparql(modified_query)
                        signal.alarm(0)
                    except TimeoutError as e:
                        print("timed out")
                        q_answer_labels = ["No additional information found for this sub-question."]
 

                        # query_answers[i+1] = q_answers
                        # query_answer_labels[i+1] = q_answer_labels

                    additional_knowledge[i]['answers'] = q_answer_labels
                else:
                    additional_knowledge[i]['answers'] = []
                
            

            # obtaining final answer
            Final_Answer_prompt = copy.deepcopy(Final_Answer_prompt_template)
            
            sub_questions_string = """"""

            for i, value in additional_knowledge.items():
                ans = additional_knowledge[i]['answers']
                if len(ans) == 0:
                    ans_string = "No results found for this sub-question."
                else:
                    ans_string = str(ans)

                sub_questions_string += f"Sub-question {i}: " + value['sub_question'] + " Answer: " + ans_string + "\n"
            
            Final_Answer_prompt[-1]['input3'] = Final_Answer_prompt[-1]['input3'].format(QUESTION = question, SUBQUESTIONS = sub_questions_string)

            Final_Answer_message = [{"role": "system", "content": Final_Answer_prompt[0]['system']}, {"role": "user", "content":Final_Answer_prompt[1]['input1']},
                                    {"role": "assistant", "content":Final_Answer_prompt[2]['output1']}, {"role": "user", "content":Final_Answer_prompt[3]['input2']},
                                    {"role": "assistant", "content":Final_Answer_prompt[4]['output2']}, {"role": "user", "content":Final_Answer_prompt[-1]['input3']}]

            conversation_history += Final_Answer_message
            Generated_Answer = get_response(Final_Answer_message)
            conversation_history.append({"role": "assistant", "content": Generated_Answer})
            Final_Answer = Generated_Answer[Generated_Answer.find(':') + 1:]
            answered = not(Final_Answer.strip().lower().startswith("i don't know"))

            if not answered:
                unanswered_counter += 1
                log_to_csv([current_q_num, "idk", true_answer], csv_file_path)
                with open(os.path.join(output_file_path, f'conversation_history.yaml'),'a') as yaml_file_out:
                    yaml.dump(conversation_history, yaml_file_out, sort_keys=False, default_flow_style=False)
                continue
                
            else:
                

                binary_final_answer = Final_Answer.strip().lower().startswith("yes")

                result = (binary_final_answer == true_answer)
                if result:
                    correct_counter += 1
                    log_to_csv([current_q_num, result, true_answer], csv_file_path)

                with open(os.path.join(output_file_path, f'conversation_history.yaml'),'a') as yaml_file_out:
                    yaml.dump(conversation_history, yaml_file_out, sort_keys=False, default_flow_style=False)
                



    accuracy = correct_counter / query_counter
    unanswered_rate = unanswered_counter / query_counter

    print('accuracy:', accuracy)
    print('unanswered rate:', unanswered_rate)

    with open(os.path.join(output_file_path, f'conversation_history.yaml'),'a') as yaml_file_out:
        yaml.dump(conversation_history, yaml_file_out, sort_keys=False, default_flow_style=False)

    print(f"Conversation history saved to {output_file_path} in conversation_history.yaml")

    with open(os.path.join(output_file_path, 'final_results.txt'), 'w') as txt_file:
        txt_file.write(f'Accuracy: {accuracy}\n')
        txt_file.write(f'Unanswered Rate: {unanswered_rate}\n')
    
    print(f"Results saved to {output_file_path} in final_results.txt")


if __name__ == '__main__':
    main()