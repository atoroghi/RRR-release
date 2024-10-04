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

openai.api_key_path = 'api_key.txt'
def log_to_csv(query_outcome, csv_file_path):
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(query_outcome)

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


def main():
    parser = argparse.ArgumentParser(description='Extract cooking methods from recipes.')
    parser.add_argument('path', help='Path to directory containing queries')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', default='StrategyQA', choices=['Recipe-MPR', 'StrategyQA', 'Creak'])
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
    elif dataset == 'Creak':
        data_path = os.path.join(dataset_path, f'{args.dataset_name}_modified.csv')
        with open(data_path, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = list(csv_reader)
    answer_col_num = 3
    if args.mode == 'original':
        q_col_num = 1
    elif args.mode == 'modified':
        q_col_num = 2

    if dataset == 'StrategyQA':
        prompts_folder = 'prompts'
    elif dataset == 'Creak':
        prompts_folder = 'prompts_Creak_rev1'

    with open(os.path.join(f'{prompts_folder}' , 'KGR_prompt.yaml'), 'r', encoding="utf-8") as yaml_file:
        yaml_str = yaml.load(yaml_file, Loader=yaml.FullLoader)

    Initial_Answer_prompt_template = yaml_str['Initial_Answer']
    Claim_Extraction_prompt_template = yaml_str['Claim_Extraction']
    Entity_Extraction_prompt_template = yaml_str['Entity_Extraction']
    Fact_Selection_prompt_template = yaml_str['Fact_Selection']
    Claim_Verification_prompt_template = yaml_str['Claim_Verification']
    Final_Answer_prompt_template = yaml_str['Final_Answer']
    

    output_file_path = os.path.join('results', f'KGR_{dataset}_{args.mode}')
    os.makedirs(output_file_path, exist_ok=True)
    last_answered_query = 0
    csv_file_path = os.path.join(output_file_path, 'Outcomes_Per_Query.csv')
    if os.path.exists(csv_file_path):
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            last_line = None
            for row in reader:
                last_line = row
            if last_line is not None:
                last_answered_query = int(last_line[0])
                print(f'Last previously-answered query: {last_answered_query}')

    query_counter = 0; unanswered_counter = 0; correct_counter = 0; 



    for i, row in tqdm.tqdm(enumerate(data[last_answered_query+1:])):
        conversation_history = []
        current_q_num = i+1+last_answered_query


        if len(row) >= 4:
            query_counter += 1
            true_answer = (row[answer_col_num]).strip().lower().startswith("true")
            # obtaining the initial response
            question = row[q_col_num]
            conversation_history.append({"query ID": row[0]})
            conversation_history.append({"query": question})


            Initial_Answer_prompt = copy.deepcopy(Initial_Answer_prompt_template)
            Initial_Answer_prompt[-1]['input1'] = Initial_Answer_prompt[-1]['input1'].format(QUESTION = question)
            Initial_Answer_message = [{"role": "system", "content": Initial_Answer_prompt[0]['system']}, {"role": "user", "content":Initial_Answer_prompt[1]['input1']}]
            conversation_history += Initial_Answer_message
            Initial_Answer = get_response(Initial_Answer_message)
            conversation_history.append({"role": "assistant", "content": Initial_Answer})
            
            answered = not(Initial_Answer.strip().lower().startswith("i don't know"))

            if not answered:
                unanswered_counter += 1
                with open(os.path.join(output_file_path, 'conversation_history.yaml'),'a') as yaml_file_out:
                    yaml.dump(conversation_history, yaml_file_out, sort_keys=False, default_flow_style=False)
                continue
                
            else:
                # extracting claims
                Claim_Extraction_prompt = copy.deepcopy(Claim_Extraction_prompt_template)
                Claim_Extraction_prompt[-1]['input3'] = Claim_Extraction_prompt[-1]['input3'].format(SENTENCE = Initial_Answer)
            

                Claim_Extraction_message = [{"role": "system", "content": Claim_Extraction_prompt[0]['system']}, {"role": "user", "content":Claim_Extraction_prompt[1]['input1']}, 
                                            {"role": "assistant", "content": Claim_Extraction_prompt[2]['output1']}, {"role": "user", "content": Claim_Extraction_prompt[3]['input2']},
                                              {"role": "assistant", "content": Claim_Extraction_prompt[4]['output2']},
                                                {"role": "user", "content": Claim_Extraction_prompt[-1]['input3']}]
                conversation_history += Claim_Extraction_message
                claims = get_response(Claim_Extraction_message)
                conversation_history.append({"role": "assistant", "content": claims})

                claims_start_index = claims.find(':') + 1
                claims_string = claims[claims_start_index:].strip()
                claims_list = [re.sub(r'^\d+-\s*', '', claim.strip()) for claim in claims_string.split('/')]

            
                # retrofitting claims

                verified_claims = []

                for claim in claims_list:
                    Entity_Extraction_prompt = copy.deepcopy(Entity_Extraction_prompt_template)
                    Entity_Extraction_prompt[-1]['input3'] = Entity_Extraction_prompt[-1]['input3'].format(SENTENCE = claim)

                    Entity_Extraction_message = [{"role": "system", "content": Entity_Extraction_prompt[0]['system']}, {"role": "user", "content":Entity_Extraction_prompt[1]['input1']},
                                                {"role": "assistant", "content": Entity_Extraction_prompt[2]['output1']}, {"role": "user", "content": Entity_Extraction_prompt[3]['input2']},
                                                {"role": "assistant", "content": Entity_Extraction_prompt[4]['output2']},
                                                {"role": "user", "content": Entity_Extraction_prompt[-1]['input3']}]
                    conversation_history += Entity_Extraction_message
                    entities = get_response(Entity_Extraction_message)
                    conversation_history.append({"role": "assistant", "content": entities})

                    entities_start_index = entities.find(':') + 1
                    entities_string = entities[entities_start_index:].strip()
                    entities_list = [re.sub(r'^\d+-\s*', '', entity.strip()) for entity in entities_string.split('/')]
                    relevant_facts_list = []

                    for entity in entities_list:
                        facts = WikidataRetriever.get_wikidata_facts(entity)
                        if len(facts) == 0 or type(facts) == str:
                            continue
                        facts_str = str(facts)
                        facts_str_truncated = truncate_string_to_words(facts_str, 500)

                        Fact_Selection_prompt = copy.deepcopy(Fact_Selection_prompt_template)
                        
                        Fact_Selection_prompt[-1]['input3'] = Fact_Selection_prompt[-1]['input3'].format(CLAIM = claim, FACTS = facts_str_truncated)

                        Fact_Selection_message = [{"role": "system", "content": Fact_Selection_prompt[0]['system']}, {"role": "user", "content":Fact_Selection_prompt[1]['input1']},
                                                {"role": "assistant", "content": Fact_Selection_prompt[2]['output1']}, {"role": "user", "content": Fact_Selection_prompt[3]['input2']},
                                                {"role": "assistant", "content": Fact_Selection_prompt[4]['output2']},
                                                {"role": "user", "content": Fact_Selection_prompt[-1]['input3']}]
                        conversation_history += Fact_Selection_message
                        
                        relevant_facts = get_response(Fact_Selection_message)
                        conversation_history.append({"role": "assistant", "content": relevant_facts})
                        relevant_facts_list.append(relevant_facts)
                    
                    #verifying the claim
                    Claim_Verification_prompt = copy.deepcopy(Claim_Verification_prompt_template)
                    Claim_Verification_prompt[-1]['input'] = Claim_Verification_prompt[-1]['input'].format(CLAIM = claim, FACTS = str(relevant_facts_list))
                    Claim_Verification_message = [{"role": "system", "content": Claim_Verification_prompt[0]['system']}, {"role": "user", "content":Claim_Verification_prompt[1]['input']}]
                    conversation_history += Claim_Verification_message
                    
                    verified_claim = get_response(Claim_Verification_message)
                    conversation_history.append({"role": "assistant", "content": verified_claim})
                    verified_claims.append(verified_claim)
                
                # generating the final answer
                Final_Answer_prompt = copy.deepcopy(Final_Answer_prompt_template)
                Final_Answer_prompt[-1]['input'] = Final_Answer_prompt[-1]['input'].format(QUESTION = question , CLAIMS = str(verified_claims))
                Final_Answer_message = [{"role": "system", "content": Final_Answer_prompt[0]['system']}, {"role": "user", "content":Final_Answer_prompt[1]['input']}]
                conversation_history += Final_Answer_message
                
                final_answer = get_response(Final_Answer_message)
                conversation_history.append({"role": "assistant", "content": final_answer})

                if not (final_answer.strip().lower().startswith("yes") or final_answer.strip().lower().startswith("no")):
                    unanswered_counter += 1
                    log_to_csv([current_q_num, 'idk', true_answer], csv_file_path)
                    with open(os.path.join(output_file_path, 'conversation_history.yaml'),'a') as yaml_file_out:
                        yaml.dump(conversation_history, yaml_file_out, sort_keys=False, default_flow_style=False)
                    continue

                binary_final_answer = final_answer.strip().lower().startswith("yes")
                with open(os.path.join(output_file_path, 'conversation_history.yaml'),'a') as yaml_file_out:
                    yaml.dump(conversation_history, yaml_file_out, sort_keys=False, default_flow_style=False)


                
                result = (binary_final_answer == true_answer)
                if result:
                    correct_counter += 1
                log_to_csv([current_q_num, binary_final_answer, true_answer], csv_file_path)
                



    accuracy = correct_counter / query_counter
    unanswered_rate = unanswered_counter / query_counter

    print('accuracy:', accuracy)
    print('unanswered rate:', unanswered_rate)

    with open(os.path.join(output_file_path, 'conversation_history.yaml'),'a') as yaml_file_out:
        yaml.dump(conversation_history, yaml_file_out, sort_keys=False, default_flow_style=False)

    print(f"Conversation history saved to {output_file_path} in conversation_history.yaml")

    with open(os.path.join(output_file_path, 'final_results.txt'), 'w') as txt_file:
        txt_file.write(f'Accuracy: {accuracy}\n')
        txt_file.write(f'Unanswered Rate: {unanswered_rate}\n')
    
    print(f"Results saved to {output_file_path} in final_results.txt")


if __name__ == '__main__':
    main()