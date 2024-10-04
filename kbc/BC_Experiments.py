import os, pickle, sys, time, copy
from utils.chat import Chat
import numpy as np
import tqdm
import pandas as pd
from collections import defaultdict
import yaml, csv
import openai
import requests
import argparse
from requests.exceptions import ConnectionError
from kbc.Reasoner import Reasoner
from kbc.KGAssistant import KGassistant
from kbc.KnowledgeRetriever import KnowledgeRetriever


def log_to_csv(query_outcome, csv_file_path):
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(query_outcome)


def str2bool(answer):
    if answer.lower()[-2:] == 'se':
        return False
    elif answer.lower()[-2:] == 'ue':
        return True

def run_experiments(data, agent, dataset, csv_file_path, mode, chat, k_facts, max_breadth, max_depth, q_col_num,ans_col_num, last_answered_query):
    outcomes = []
    #for i, q in tqdm.tqdm(enumerate(data[:1])):
    k_facts = 20



    knowledge_retriever = KnowledgeRetriever(os.path.join(os.getcwd(), 'models'))


    #for i, row in tqdm.tqdm(enumerate(data[last_answered_query+1:3])):
    for i, row in tqdm.tqdm(enumerate(data[1:151])):
        

        current_q_num = i+1+last_answered_query
        
        id = row[0]

        question = row[q_col_num]
        kgassistant = KGassistant(dataset, agent, question)

        answered = False
        first_attempt = True
        axioms = []
        selected_entities = set()

        current_depth = 0
        current_breadth = 0
        while not answered:
            candidate_axioms = []
            if first_attempt:
                first_attempt = False

                # Generate Axiom
                entities, prompt, response = agent.identify_entities(question)

                selected_entities.update(entities)
                
                chat.add_step(id, prompt, response)
                print(question)
                
                print(outcomes)

                if agent.refined is not None:
                    
                    entities = kgassistant.qid2labels(entities)
                    #entity2descriptions = kgassistant.description_finder(entities)
                    #entity2descriptions = kgassistant.description_finder(entities, name=False)
                selected_entities.update(entities)
                entities = list(selected_entities)
                #entities = [entity for entity in entities if ('Q' not in entity and entity != 'New York' and entity != 'People\'s Republic of China' and entity!='surf break') ]


                print(entities)


                entity2descriptions = kgassistant.description_finder(entities)
                               
                raw_entity2facts, enetity2tail_labels = kgassistant.facts_finder(entities, truncated = True)

                entity2facts = {}

                
                for entity in raw_entity2facts:
                    # subgraph pruning ablation (uncomment next line)
                    entity2facts[entity] = []
                    #entity2facts[entity] = knowledge_retriever(question, raw_entity2facts[entity], k_facts)[:k_facts]

                if current_breadth == 0:
                    axiom, prompt, response = agent.generate_fact(question, entity2facts)
                else:
                    # generating new axioms and not the old ones
                    axiom, prompt, response = agent.generate_fact_next(question, entity2facts, axioms)
                #candidate_axioms.append(axiom)
                chat.add_step(id, prompt, response)
                #for i in range(2):
                #    new_axiom, prompt, response = agent.generate_fact_next(question, entity2facts, axioms)
                #    candidate_axioms.append(new_axiom)
                    
                #selected_axiom, prompt, response = agent.select_axiom(question, entity2facts, candidate_axioms)
                #verified_axiom, prompt, response = agent.verify_axiom(question, axiom)
                
                selected_axiom = axiom
                chat.add_step(id, prompt, response)
                axioms.append(selected_axiom)
                
                relevant_facts = copy.deepcopy(entity2facts)

                # subgraph pruning ablation (comment next six lines and uncomment 7th)
                # try:
                #     raw_entity2facts_truncated, _ = kgassistant.facts_finder(entities, truncated = False) 
                #     new_relevant_facts, chat = agent.identify_relevant_kgfacts(question, entity2descriptions, axioms, raw_entity2facts_truncated, id, chat)
                # except:
                #     raw_entity2facts_truncated, _ = kgassistant.facts_finder(entities, truncated = True) 
                #     new_relevant_facts, chat = agent.identify_relevant_kgfacts(question, entity2descriptions, axioms, raw_entity2facts_truncated, id, chat)

                new_relevant_facts, _ = kgassistant.facts_finder(entities, truncated = True) 
            
            

                for key, value in new_relevant_facts.items():
                    if key in relevant_facts:
                        relevant_facts[key].extend(value)
                    else:
                        relevant_facts[key] = value
                

                answered, binary_answer, answer, message = agent.attempt_answering(question, entity2descriptions, axioms, relevant_facts)
                chat.add_step(id, message, answer)
                #verified, evaluation, message = agent.verify_answer(question, entity2descriptions, axioms, relevant_facts, answer)
                #verified = True
                #chat.add_step(id, message, evaluation)
                #if not verified:
                #    answered, binary_answer, answer, message = agent.correct_answer(question, entity2descriptions, axioms, relevant_facts, answer, evaluation)
                #    chat.add_step(id, message, answer)
               
                if answered:                    
                    if binary_answer == 'idk' or binary_answer is None:
                        outcomes.append("idk")
                        log_to_csv([current_q_num, "idk", row[ans_col_num]], csv_file_path)
                    else:
                        outcomes.append(binary_answer == str2bool(row[ans_col_num]))
                        log_to_csv([current_q_num, binary_answer, row[ans_col_num]], csv_file_path)
                    chat.save()
                    break
                # up to line 159 added for no iteration
                # else:
                #     # if the answer is not found, we need to decide on the next entity to consider
                #     current_breadth += 1
                #     if current_breadth == max_breadth :
                #         chat.save()
                #         outcomes.append('idk')
                #         log_to_csv([current_q_num, "idk", row[ans_col_num]], csv_file_path)
                #         break

                

            if not (answered or first_attempt):
                # decide between second hop and new fact
                next_entity_decision = True
                
                while next_entity_decision and current_depth <= max_depth:
                    # TODO: If identify missing isn't working properly, we need to add the evaluation in the prompt
                    answerability, answer_missing, message_missing = agent.identify_missing(question, answer, entity2descriptions, axioms, relevant_facts, selected_entities)
                    chat.add_step(id, message_missing, answer_missing)
                    if answerability:
                        answered, binary_answer, answer, message = agent.attempt_answering(question, entity2descriptions, axioms, relevant_facts)
                        chat.add_step(id, message, answer)
                        #polarized_answer, message = agent.decide_polarity(question, answer)
                        #if message is not None:
                        #    chat.add_step(id, message, polarized_answer)
                        #binary_answer = agent.binarize_answer(polarized_answer)
                        if binary_answer == 'idk' or binary_answer is None:
                            outcomes.append("idk")
                            log_to_csv([current_q_num, "idk", row[ans_col_num]], csv_file_path)
                        else:
                            outcomes.append(binary_answer == str2bool(row[ans_col_num]))
                            log_to_csv([current_q_num, binary_answer, row[ans_col_num]], csv_file_path)
                        chat.save()
                        break

                    next_entity, prompt = agent.decide_next_entity(question, answer_missing, relevant_facts, selected_entities)
                    chat.add_step(id, prompt, next_entity)
                    if next_entity and next_entity != 'None':
                        # we already have the most relevant facts. we need to add the next most relevant ones.
                        if next_entity in entity2facts:
                            new_facts = knowledge_retriever(answer_missing, raw_entity2facts[next_entity], (current_depth+2)*k_facts)[(current_depth+1)*k_facts:(current_depth+2)*k_facts]
                            new_next_relevant_facts, chat = agent.identify_relevant_kgfacts(answer_missing, entity2descriptions, axioms, raw_entity2facts, id, chat, next_entity=next_entity)
                            new_facts += new_next_relevant_facts[next_entity]
                            entity2facts[next_entity].extend(new_facts)

                        else:
                            # obtain tail label
                            next_entity2descriptions = {}
                            next_entity_id = None
                            for entity in enetity2tail_labels:
                                if next_entity in enetity2tail_labels[entity]:
                                    next_entity_id = enetity2tail_labels[entity][next_entity]
                                    next_entity2descriptions = kgassistant.description_finder([next_entity_id], name=False, label=next_entity)
                                    raw_next_entity2facts, secondhop_tail2labels = kgassistant.facts_finder([next_entity_id], name=False, label=next_entity)
                                    #raw_entity2facts[next_entity] = raw_next_entity2facts[next_entity_id]
                                    raw_entity2facts[next_entity] = raw_next_entity2facts[next_entity]
                                    #enetity2tail_labels[next_entity] = secondhop_tail2labels[next_entity_id]
                                    enetity2tail_labels[next_entity] = secondhop_tail2labels[next_entity]
                                    break
                            # if we don't have its tail id
                            if not next_entity_id:
                                 next_entity2descriptions = kgassistant.description_finder([next_entity], name=True)
                                 raw_next_entity2facts, secondhop_tail2labels = kgassistant.facts_finder([next_entity], name=True)
                                 if bool(secondhop_tail2labels):
                                    enetity2tail_labels[next_entity] = secondhop_tail2labels[next_entity]
                                    raw_entity2facts[next_entity] = raw_next_entity2facts[next_entity]
                                

                        

                            # replacing the id with the name
                            try:
                                raw_next_entity2facts[next_entity] = raw_next_entity2facts.pop(next_entity_id)
                                next_entity2descriptions[next_entity] = next_entity2descriptions.pop(next_entity_id)
                                
                            except:
                                pass

                            next_entity2facts = {}
                            for entity in raw_next_entity2facts:
                                # TODO: Why does the number of ranked facts limit to 10?
                                facts = raw_next_entity2facts[entity]
                                ranked_facts = knowledge_retriever(answer_missing, facts, k_facts)
                                
                                next_entity2facts[entity] = ranked_facts[:k_facts]
                            
                            # new_next_relevant_facts, chat = agent.identify_relevant_kgfacts(answer_missing, next_entity2descriptions, axioms, raw_next_entity2facts, id, chat, next_entity=next_entity)
                            # for key, value in new_next_relevant_facts.items():
                            #     if key in next_entity2facts:
                            #         next_entity2facts[key].extend(value)
                            #     else:
                            #         next_entity2facts[key] = value

                            next_relevant_facts = next_entity2facts
                            #next_relevant_facts, chat = agent.identify_relevant_kgfacts(question, next_entity2descriptions, axioms, next_entity2facts, id, chat)
                            
                            # the next entity may not have any description
                            try:
                                relevant_facts[next_entity] = next_relevant_facts[next_entity]
                                entity2descriptions[next_entity] = next_entity2descriptions[next_entity]
                            except:
                                pass
                    else:
                        next_entity_decision = False
                        # this branch has failed, try another commonsense axiom
                        break

                    answered, binary_answer, answer, message = agent.attempt_answering(question, entity2descriptions, axioms, relevant_facts)
                    chat.add_step(id, message, answer)          
                    # verified, evaluation, message = agent.verify_answer(question, entity2descriptions, axioms, relevant_facts, answer)              
                    # chat.add_step(id, message, evaluation)
                    # if not verified:
                    #     answered, binary_answer, answer, message = agent.correct_answer(question, entity2descriptions, axioms, relevant_facts, answer, evaluation)
                    #     chat.add_step(id, message, answer)

                    if answered:

                        if binary_answer == 'idk' or binary_answer is None:
                            outcomes.append("idk")
                            log_to_csv([current_q_num, "idk", row[ans_col_num]], csv_file_path)
                        else:
                            outcomes.append(binary_answer == str2bool(row[ans_col_num]))
                            log_to_csv([current_q_num, binary_answer, row[ans_col_num]], csv_file_path)
                        chat.save()
                        break
                    


                    current_depth += 1

                if answered or answerability:
                    break
                current_breadth += 1
                first_attempt = True
                if current_breadth == max_breadth :
                    chat.save()
                    outcomes.append('idk')
                    log_to_csv([current_q_num, "idk", row[ans_col_num]], csv_file_path)
                    break





    return outcomes






def main():
    parser = argparse.ArgumentParser(description='Extract cooking methods from recipes.')
    parser.add_argument('path', help='Path to directory containing queries')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', default='StrategyQA', choices=['Recipe-MPR', 'StrategyQA', 'Creak'])
    parser.add_argument('--direction', type=str, help='Direction of the experiment', default='bidirectional', choices=['forward', 'backward', 'bidirectional'])
    parser.add_argument('--mode', type=str, help='Mode of the experiment', default='valid', choices=['original', 'modified'])
    parser.add_argument('--facts_num', type=int, help='Number of top scoring facts to keep', default=10)
    parser.add_argument('--use_refined', type=str, help='Use refined entity linker', default="True")
    parser.add_argument('--max_breadth', type=int, help='Breadth of search', default=1)
    parser.add_argument('--max_depth', type=int, help='Depth of search', default=2)
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment', default='no_name')
    args = parser.parse_args()



    dataset = args.dataset_name

    dataset_path = os.path.join(os.getcwd() , args.path, dataset)
    # if dataset == 'Recipe-MPR':
    #     data_path = os.path.join(dataset_path, '500QA.json')
    #     with open(data_path, "r") as file:
    #         data = json.load(file)
    if dataset == 'StrategyQA':
        data_path = os.path.join(dataset_path, f'{args.dataset_name}_modified.csv')
        with open(data_path, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = list(csv_reader)
        if args.mode == 'original':
            prompts_folder = "prompts_rev1_original"
        else:
            prompts_folder = "prompts_rev1"
    elif dataset == 'Creak':
        data_path = os.path.join(dataset_path, f'{args.dataset_name}_modified.csv')
        with open(data_path, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = list(csv_reader)
        prompts_folder = "prompts_Creak_rev2"

    
    if args.mode == 'original':
        q_col_num = 1
        answer_col_num = 4
    elif args.mode == 'modified':
        q_col_num = 2
        answer_col_num = 3

    use_refined = str2bool(args.use_refined)

    agent = Reasoner(args.direction, use_refined, prompts_folder)

    chat_f = f"chats/{args.dataset_name}/{args.mode}/BC_Experiments_{args.experiment_name}.yaml"
    os.makedirs(os.path.join(os.getcwd(), 'chats', args.dataset_name, args.mode), exist_ok=True)
    chat = Chat(chat_f)

    results_file_path = os.path.join(os.getcwd(), 'results', 'BC_Experiments', dataset, args.mode, args.experiment_name)

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
    else:
        os.makedirs(os.path.join(results_file_path), exist_ok=True)



    outcomes = run_experiments(data, agent, dataset, csv_file_path, args.mode, chat, args.facts_num, args.max_breadth, args.max_depth,q_col_num, answer_col_num, last_answered_query)
    true_count = outcomes.count(True)
    false_count = outcomes.count(False)
    idk_count = sum(isinstance(entry, str) for entry in outcomes)

    print(f'True: {true_count}\n')
    print(f'False: {false_count}\n')
    print(f'IDK: {idk_count}\n')
    print(f'Accuracy: {true_count/(true_count+false_count+idk_count)}\n')
    print(f'Unanswered rate: {idk_count/(true_count+false_count+idk_count)}\n')

    os.makedirs(results_file_path, exist_ok=True)

    with open(os.path.join(results_file_path, 'final_results.txt'), 'w') as txt_file:
        txt_file.write(f'True: {true_count}\n')
        txt_file.write(f'False: {false_count}\n')
        txt_file.write(f'IDK: {idk_count}\n')
        txt_file.write(f'Accuracy: {true_count/(true_count+false_count+idk_count)}\n')
        txt_file.write(f'Unanswered rate: {idk_count/(true_count+false_count+idk_count)}\n')
        for i, outcome in enumerate(outcomes):
            txt_file.write(f'S{i+1}: {outcome}\n')

if __name__ == '__main__':
    main()

