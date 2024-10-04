import os
import csv
import sys

from collections import defaultdict, Counter
from ruamel.yaml import YAML

sys.path.append("..")

from utils.prompt import Prompt
from utils.llm import LLM
from utils.chat import Chat


def run_strategy_qa():
    openai = LLM()

    llm_prompt = Prompt("../prompts/llm.yaml")
    # llm_prompt = Prompt("../prompts/llm_cot.yaml")

    chat_f = "../chats/llm_0_shot.yaml"
    # chat_f = "../chats/llm_0_shot_cot.yaml"
    # chat_f = "../chats/llm_few_shot_cot.yaml"

    chat = Chat(chat_f)

    strategy_qa_f = "../data/StrategyQA/StrategyQA_Modified.csv"

    with open(strategy_qa_f, "r") as f:
        strategy_qa = csv.reader(f)
        next(strategy_qa)

        for row in strategy_qa:
            id = row[0]
            question = row[2]
            answer = row[3] == 'TRUE'

            if id in chat.chats:
                print(f'skip {id}')
                continue

            # prompt
            prompt_params = {"QUESTION": question}
            prompt = llm_prompt(prompt_params)
            # prompt = llm_prompt(prompt_params, few_shot=True)

            # call llm
            response = openai(prompt)

            # log
            chat.add_step(id, prompt, response)

            chat.save()

def zero_shot_parse_response(response):
    response = response.strip().lower()

    if "i don't know" in response:
        return None
    elif "yes" in response:
        return True
    elif "no" in response:
        return False
    else:
        print(response)
        return None

def cot_parse_response(response):
    response = response.strip().lower()

    sentences = response.split('.')

    if sentences[-1]:
        last = sentences[-1].strip()
    else:
        last = sentences[-2].strip()

    if "i don't know" in last:
        return None

    if "yes" in last:
        return True
    elif "no" in last:
        return False
    else:
        print(last)
        return None

def eval_strategy_qa():

    result = Counter()

    # chat_f = "../chats/llm_0_shot.yaml"
    # chat_f = "../chats/llm_0_shot_cot.yaml"
    chat_f = "../chats/llm_few_shot_cot.yaml"

    with open(chat_f, 'r') as f:
        chats = dict(YAML().load(f))


    strategy_qa_f = "../data/StrategyQA/StrategyQA_Modified.csv"

    with open(strategy_qa_f, "r") as f:
        strategy_qa = csv.reader(f)
        next(strategy_qa)

        for row in strategy_qa:
            id = row[0]
            question = row[2]
            answer = row[3] == 'TRUE'

            llm_response = chats[id][-1][-1]['assistant']

            # llm_answer = zero_shot_parse_response(llm_response)
            llm_answer = cot_parse_response(llm_response)

            if llm_answer is None:
                result.update(["None"])
                continue

            if answer == llm_answer:
                result.update(["Correct"])
            else:
                print(llm_response)
                result.update(["Incorrect"])
    
    print(result)
            




if __name__ == "__main__":
    # TODO: argument parser

    # run_strategy_qa()
    eval_strategy_qa()
