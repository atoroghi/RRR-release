import os
import sys
import csv
from collections import Counter

from ruamel.yaml import YAML
from sentence_transformers import SentenceTransformer, util
from refined.inference.processor import Refined

sys.path.append("..")
from utils.wikidata import Wikidata
from utils.prompt import Prompt
from utils.chat import Chat
from utils.llm import LLM

class Kaping:
    def __init__(self, model_dir):
        self.entity_extractor = EntityExtractor(model_dir)
        self.wikidata = Wikidata()
        self.knowledge_retriever = KnowledgeRetriever(model_dir)

        self.prompt = Prompt("../prompts/kaping.yaml")

    def __call__(self, question):
        # extract question entities
        question_entities = self.entity_extractor(question)

        if question_entities == [None]:
            prompt_params = {"FACTS": [], "QUESTION": question}
            return self.prompt(prompt_params)

        # retrieve wikidata triples
        wikidata_triples = []
        for entity in question_entities:
            entity_triples = self.wikidata.from_id(entity)
            if entity_triples:
                wikidata_triples += self.wikidata.from_id(entity)

        verbalized_wikidata_triples = self.wikidata.verbalize_triples(wikidata_triples)

        # get top wikidata triples
        knowledge = self.knowledge_retriever(question, verbalized_wikidata_triples)

        prompt_params = {"FACTS": knowledge, "QUESTION": question}
        return self.prompt(prompt_params)


class EntityExtractor:
    def __init__(self, model_dir):
        self.refined = Refined.from_pretrained(
            model_name="wikipedia_model",
            entity_set="wikidata",
            data_dir=f"{model_dir}/refined",
        )

    def __call__(self, s):
        spans = self.refined.process_text(s)
        return [span.predicted_entity.wikidata_entity_id for span in spans]


class KnowledgeRetriever:
    def __init__(self, model_dir):
        self.mpnet = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            cache_folder=f"{model_dir}/sentence_transformers",
        )

    def __call__(self, question, facts):
        question_embedding = self.encode([question])
        facts_embeddings = self.encode(facts)

        results = util.semantic_search(
            question_embedding, facts_embeddings, score_function=util.dot_score
        )[0]

        return [facts[result["corpus_id"]] for result in results]

    def encode(self, s):
        return self.mpnet.encode(s, normalize_embeddings=True, convert_to_tensor=True)

def run_strategy_qa():
    kaping = Kaping("/scratch/guowill/kaping")

    openai = LLM()

    prompt = Prompt("../prompts/kaping.yaml")

    chat_f = "../chats/kaping.yaml"
    chat = Chat(chat_f)

    strategy_qa_f = "../data/StrategyQA/StrategyQA_Modified.csv"

    with open(strategy_qa_f, "r") as f:
        strategy_qa = csv.reader(f)
        next(strategy_qa)

        for row in strategy_qa:
            id = row[0]
            question = row[2]
            answer = row[3] == "TRUE"

            if id in chat.chats:
                print(f'skip {id}')
                continue

            prompt = kaping(question)

            # call llm
            response = openai(prompt)

            # log
            chat.add_step(id, prompt, response)
    
            chat.save()

def kaping_parse_response(response):
    content = response.strip().lower()

    answer = content.split(".")[0]

    if "i don't know" in answer:
        return None
    elif "yes" in answer:
        return True
    elif "no" in answer:
        return False
    else:
        print(answer)
        return None

def eval_strategy_qa():
    result = Counter()

    chat_f = "../chats/kaping.yaml"

    with open(chat_f, "r") as f:
        chats = dict(YAML().load(f))

    strategy_qa_f = "../data/StrategyQA/StrategyQA_Modified.csv"

    with open(strategy_qa_f, "r") as f:
        strategy_qa = csv.reader(f)
        next(strategy_qa)

        for row in strategy_qa:
            id = row[0]
            question = row[2]
            answer = row[3].strip() == "TRUE"


            llm_response = chats[id][-1][-1]["assistant"]

            llm_answer = kaping_parse_response(llm_response)

            if llm_answer is None:
                result.update(["None"])
                continue

            if answer == llm_answer:
                result.update(["Correct"])
            else:
                result.update(["Incorrect"])

    print(result)

if __name__ == "__main__":
    # TODO: argument parser

    # run_strategy_qa()
    eval_strategy_qa()
    

    
