import os, pickle, sys, time, copy
from utils.chat import Chat
import numpy as np
import tqdm
import pandas as pd
from collections import defaultdict
import yaml, csv, json
import argparse
from kbc.ReasonerMPR import ReasonerMPR



def log_to_csv(query_outcome, csv_file_path):
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(query_outcome)


def str2bool(answer):
    if answer.lower()[-2:] == 'se':
        return False
    elif answer.lower()[-2:] == 'ue':
        return True

def run_experiments(data, agent, dataset, csv_file_path, chat, max_breadth, max_depth, last_answered_query, method_name):
    outcomes = []
    #for i, q in tqdm.tqdm(enumerate(data[:1])):






    #begin_index = last_answered_query+1
    #end_index = 101
    begin_index = 0
    end_index = 101
    for i, row in tqdm.tqdm(enumerate(data[begin_index:end_index])):
        
        #current_q_num = i+1+last_answered_query
        current_q_num = i+begin_index
        id = "M"+str(current_q_num)
    

        query = row['query']
        recipes = row['recipes']

        answered = False
        first_attempt = True
        axioms = []

        current_depth = 0
        current_breadth = 0
        while not answered:
            if first_attempt:
                first_attempt = False

                # Generate Axiom
                
                print(query)
                
                print(outcomes)


                entity2facts = {'user':row['triples']}
                if method_name == 'KAPING':
                    selected_option, answer, message = agent.attempt_answering_kaping(query, recipes, entity2facts)
                    chat.add_step(id, message, answer)
                    if selected_option == row['answer']:
                        outcomes.append(True)
                        log_to_csv([id, selected_option, row['answer']], csv_file_path)
                        chat.save()
                        break
                    else:
                        outcomes.append(False)
                        log_to_csv([id, selected_option, row['answer']], csv_file_path)
                        chat.save()
                        break


                if current_breadth == 0:
                    axiom, prompt, response = agent.generate_fact(query, recipes, entity2facts)
                else:
                    # generating new axioms and not the old ones
                    axiom, prompt, response = agent.generate_fact_next(query, entity2facts, axioms)
                #candidate_axioms.append(axiom)
                chat.add_step(id, prompt, response)
                
                selected_axiom = axiom
                axioms.append(selected_axiom)
                
                relevant_facts = copy.deepcopy(entity2facts)


                selected_option, answer, message = agent.attempt_answering(query, recipes, axioms, relevant_facts)
                chat.add_step(id, message, answer)
                if selected_option is not None:
                    answered = True
               
                if answered:                    
                    #if binary_answer == 'idk' or binary_answer is None:
                    #    outcomes.append("idk")
                    #    log_to_csv([current_q_num, "idk", row[ans_col_num]], csv_file_path)
                    #else:
                    if selected_option == 'idk':
                        print("Couldn't process response for query: ", query)
                        outcomes.append('idk')
                    else:
                        outcomes.append(selected_option == row['answer'])
                    log_to_csv([id, selected_option, row['answer']], csv_file_path)
                    chat.save()
                    break
                

            if not (answered or first_attempt):

                current_breadth += 1
                first_attempt = True
                if current_breadth == max_breadth :
                    chat.save()
                    outcomes.append('idk')
                    log_to_csv([id, "idk", row['answer']], csv_file_path)
                    break





    return outcomes






def main():
    parser = argparse.ArgumentParser(description='Extract cooking methods from recipes.')
    parser.add_argument('path', help='Path to directory containing queries')
    parser.add_argument('--method_name', type=str, help='name of the method', default='RRR', choices=['KAPING', 'RRR'])
    parser.add_argument('--max_breadth', type=int, help='Breadth of search', default=1)
    parser.add_argument('--max_depth', type=int, help='Depth of search', default=1)
    args = parser.parse_args()



    dataset = 'Recipe-MPR'

    dataset_path = os.path.join(os.getcwd() , args.path, dataset, 'commonsense_dict_final.json')
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    if args.method_name == 'KAPING':
        prompts_folder = "prompts_MPR_KAPING"
    elif args.method_name == 'RRR':
        prompts_folder = "prompts_MPR_RRR"

    if args.method_name == 'KAPING':
        assert args.max_breadth == 1 and args.max_depth == 1, "KAPING only supports breadth and depth of 1"


    agent = ReasonerMPR(prompts_folder = prompts_folder)

    chat_f = f"chats/Recipe-MPR/{args.method_name}/MPR_Experiments.yaml"
    os.makedirs(os.path.join(os.getcwd(), 'chats', 'Recipe-MPR', args.method_name), exist_ok=True)
    chat = Chat(chat_f)

    results_file_path = os.path.join(os.getcwd(), 'results', 'MPR_Experiments', args.method_name)
    os.makedirs(results_file_path, exist_ok=True)

    last_answered_query = -1
    csv_file_path = os.path.join(results_file_path, 'Outcomes_Per_Query.csv')
    if os.path.exists(csv_file_path):
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            last_line = None
            for row in reader:
                last_line = row
            if last_line is not None:
                try:
                    last_answered_query = int(last_line[0])
                except:
                    last_answered_query = int(last_line[0][1:])
                print(f'Last previously-answered query: {last_answered_query}')




    outcomes = run_experiments(data, agent, dataset, csv_file_path, chat, args.max_breadth, args.max_depth, last_answered_query, args.method_name)
    true_count = outcomes.count(True)
    false_count = outcomes.count(False)

    print(f'True: {true_count}\n')
    print(f'False: {false_count}\n')
    print(f'Accuracy: {true_count/(true_count+false_count)}\n')

    with open(os.path.join(results_file_path, 'final_results.txt'), 'w') as txt_file:
        txt_file.write(f'True: {true_count}\n')
        txt_file.write(f'False: {false_count}\n')
        txt_file.write(f'Accuracy: {true_count/(true_count+false_count)}\n')
        for i, outcome in enumerate(outcomes):
            txt_file.write(f'S{i+1}: {outcome}\n')

if __name__ == '__main__':
    main()

