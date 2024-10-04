import openai
import requests, time, sys, re, random
from requests.exceptions import ConnectionError
from utils.prompt import Prompt
import ruamel.yaml

from refined.inference.processor import Refined

class EntityExtractor:
    def __init__(self):

        self.refined = Refined.from_pretrained(
            model_name="wikipedia_model",
            entity_set="wikidata"
        )
        

    def __call__(self, s):
        
        spans = self.refined.process_text(s)
        return [span.predicted_entity.wikidata_entity_id for span in spans]


openai.api_key_path = 'api_key.txt'

def extract_entity_name(answer):
    match = re.search(r'Selected Entity: (.+)', answer)
    if match:
        entity_name = match.group(1)
        return entity_name
    else:
        return None


class ReasonerMPR:
    def __init__(self, prompts_folder):
        self.prompts_folder = prompts_folder


    def get_response(self,message, temp=0):
        tries = 5
        for i in range(tries):
            try:
                response = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    messages = message,
                    temperature = temp,
                    max_tokens = 300)
                break
            except ConnectionError as err:
                if i == tries - 1:
                    raise err
                else:
                    time.sleep(5)
        return response['choices'][0]['message']['content']


  


    def generate_fact(self, query, recipes, entity2facts):

        message = self.fact_generation_prompt(query, recipes, entity2facts)
        response = self.get_response(message)
        selected_rule_index = response.find("Rule:")
        if selected_rule_index == -1:
            fact = response.strip()
        else:
            fact = response[selected_rule_index + len("Rule:"):].strip()
        return fact, message, response
    
    def fact_generation_prompt(self, query, recipes, entity2facts):

        llm_prompt = Prompt(f"{self.prompts_folder}/inference_rule_v4.yaml")
        facts_numbered = []
        for entity in entity2facts:
            for i, fact in enumerate(entity2facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        recipes_numbered = []
        for i, recipe in enumerate(recipes):
            recipes_numbered.append(f"Recipe {i+1}- ingredients: {recipes[recipe]['ingredients']}\n instructions: {recipes[recipe]['instructions']}")

        #prompt_params = {"QUERY": query, "FACTS": facts_numbered, "RECIPES": recipes_numbered}
        prompt_params = {"QUERY": query, "FACTS": facts_numbered}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt
    
    def generate_fact_next(self, query, recipes, entity2facts):
        message = self.fact_generation_next_prompt(query, recipes, entity2facts)
        response = self.get_response(message, temp=0.5)
        selected_rule_index = response.find("Rule:")
        if selected_rule_index == -1:
            fact = response.strip()
        else:
            fact = response[selected_rule_index + len("Rule:"):].strip()
        return fact, message, response
    
    def fact_generation_next_prompt(self,query, recipes, entity2facts):
        llm_prompt = Prompt(f"{self.prompts_folder}/inference_rule_v4_next.yaml")
        facts_numbered = []
        for entity in entity2facts:
            for i, fact in enumerate(entity2facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        recipes_numbered = []
        for i, recipe in enumerate(recipes):
            recipes_numbered.append(f"Recipe {i+1}- ingredients: {recipes[recipe]['ingredients']}\n instructions: {recipes[recipe]['instructions']}")
        prompt_params = {"QUERY": query, "FACTS": facts_numbered, "RECIPES": recipes_numbered}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt
    
  

    
    def attempt_answering(self, sub_question, recipes, axioms, relevant_facts):
        options = list(recipes.keys())
        random.shuffle(options)
        num2id = {i+1: id for i, id in enumerate(options)}
        message = self.decide_answer_prompt(sub_question, recipes, axioms, relevant_facts, num2id)
        answered = False
        new_message = message.copy()

        while not answered:
            try:

                response = self.get_response(message)
                answered = True
            except:
                
                new_message = new_message[:-3]
                new_message.append(message[-1])
                response = self.get_response(new_message)
                answered = True


        selected_option_index = response.find("recipe is:")
        if selected_option_index != -1:
            selected_option_num = response[selected_option_index+ len("recipe is:"):].strip()[0]
            selected_option = num2id[int(selected_option_num)]
        else:
            selected_option = 'idk'
        return selected_option, response, message
    
    def decide_answer_prompt(self, query, recipes, axioms, entity2facts, num2id):
        llm_prompt = Prompt(f"{self.prompts_folder}/decide_answer2.yaml")
        axioms_numbered = []
        for i, axiom in enumerate(axioms):
            axioms_numbered.append(f"{i+1}- {axiom}")
        facts_numbered = []
        for entity in entity2facts:
            for i, fact in enumerate(entity2facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        recipes_numbered = []
        for i in num2id:
            recipes_numbered.append(f"Recipe {i}- ingredients: {recipes[num2id[i]]['ingredients']}\n instructions: {recipes[num2id[i]]['instructions']}")
        #prompt_params = {"QUERY": query, "GENERAL_RULE": axioms_numbered, "FACTS": facts_numbered, "RECIPES": recipes_numbered}
        prompt_params = {"QUERY": query, "GENERAL_RULE": axioms_numbered, "RECIPES": recipes_numbered}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt


    def attempt_answering_kaping(self, query, recipes, entity2facts):

        options = list(recipes.keys())
        random.shuffle(options)
        num2id = {i+1: id for i, id in enumerate(options)}
        message = self.decide_answer_kaping_prompt(query, recipes, options, entity2facts, num2id)
        response = self.get_response(message)
        selected_option_index = response.find("Recipe")
        selected_option_num = response[selected_option_index+ len("Recipe"):].strip()[0]
        selected_option = num2id[int(selected_option_num)]
        return selected_option, response, message
    
    def decide_answer_kaping_prompt(self, query, recipes, options, entity2facts, num2id):
        llm_prompt = Prompt(f"{self.prompts_folder}/decide_answer.yaml")
        facts_numbered = []
        for entity in entity2facts:
            for i, fact in enumerate(entity2facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        recipes_numbered = []
        for i in num2id:
            recipes_numbered.append(f"Recipe {i}- ingredients: {recipes[num2id[i]]['ingredients']}\n instructions: {recipes[num2id[i]]['instructions']}")

        prompt_params = {"QUERY": query, "FACTS": facts_numbered, "RECIPES": recipes_numbered}
        prompt = llm_prompt(prompt_params, few_shot=False)
        return prompt


