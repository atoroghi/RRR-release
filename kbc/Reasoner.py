#import openai
from openai import OpenAI
from dotenv import load_dotenv
import requests, time, sys, re, os
from requests.exceptions import ConnectionError
from utils.prompt import Prompt
import ruamel.yaml
import copy

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


#openai.api_key_path = 'api_key.txt'
load_dotenv()


def extract_entity_name(answer):
    match = re.search(r'Selected Entity: (.+)', answer)
    if match:
        entity_name = match.group(1)
        return entity_name
    else:
        return None


class Reasoner:
    def __init__(self, direction, use_refined, prompts_folder):
        self.direction = direction

        # refined = entity linker 
        if use_refined:
            self.refined = EntityExtractor()
        else:
            self.refined = None
        self.prompts_folder = prompts_folder


    def get_response(self,message, temp=0):

        tries = 5
        client = OpenAI()
        for i in range(tries):
            try:
                # response = openai.ChatCompletion.create(
                #     model = "gpt-3.5-turbo",
                #     messages = message,
                #     temperature = temp,
                #     max_tokens = 300)
                response =client.chat.completions.create(model = "gpt-3.5-turbo",
                    messages = message,
                    temperature = temp,
                    max_tokens = 300)
                break
            except ConnectionError as err:
                if i == tries - 1:
                    raise err
                else:
                    time.sleep(5)
        #return response['choices'][0]['message']['content']
        return response.choices[0].message.content

    def identify_entities_prompt(self, q):
        #llm_prompt = Prompt("prompts/identify_entities.yaml")
        llm_prompt = Prompt(f"{self.prompts_folder}/identify_entities.yaml")
        prompt_params = {"QUESTION": q}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt


    def fact_generation_prompt(self,q, entity2descriptions):

        #llm_prompt = Prompt("prompts_Creak_rev1/inference_rule_v4.yaml")
        llm_prompt = Prompt(f"{self.prompts_folder}/inference_rule_v4.yaml")
        entities = list(entity2descriptions.keys())
        #rev1
        #hints = []
        #for i, entity in enumerate(entities):
        #    hints.append(f"{i+1}- {entity}: {entity2descriptions[entity]}")
        relevant_facts = entity2descriptions
        entities_numbered = []
        facts_numbered = []
        for entity in relevant_facts:
            entities_numbered.append(f"{1+len(entities_numbered)}- {entity}")
            for i, fact in enumerate(relevant_facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        #rev1
        #prompt_params = {"QUESTION": q, "HINTS": hints}
        prompt_params = {"QUESTION": q,"ENTITIES": entities_numbered, "FACTS": facts_numbered}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt
    
    def fact_generation_next_prompt(self,q, entity2descriptions, axioms):
        #llm_prompt = Prompt("prompts/inference_rule_v4_next.yaml")
        llm_prompt = Prompt(f"{self.prompts_folder}/inference_rule_v4_next.yaml")
        entities = list(entity2descriptions.keys())
        relevant_facts = entity2descriptions
        entities_numbered = []
        facts_numbered = []
        for entity in relevant_facts:
            entities_numbered.append(f"{1+len(entities_numbered)}- {entity}")
            for i, fact in enumerate(relevant_facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        axioms_numbered = []
        for i, axiom in enumerate(axioms):
            axioms_numbered.append(f"{i+1}- {axiom}")
        prompt_params = {"QUESTION": q, "ENTITIES": entities_numbered,"FACTS": facts_numbered, "AXIOMS": axioms_numbered}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt
    
    def select_axiom_prompt(self,q, entity2descriptions, axioms):
        llm_prompt = Prompt(f"{self.prompts_folder}/select_axiom.yaml")
        entities = list(entity2descriptions.keys())
        relevant_facts = entity2descriptions
        facts_numbered = []
        for entity in relevant_facts:
            for i, fact in enumerate(relevant_facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        axioms_numbered = []
        for i, axiom in enumerate(axioms):
            axioms_numbered.append(f"{i+1}- {axiom}")
        prompt_params = {"QUESTION": q, "FACTS": facts_numbered, "AXIOMS": axioms_numbered}
        prompt = llm_prompt(prompt_params, few_shot=True)
        #prompt = llm_prompt(prompt_params)
        return prompt

    def verify_axiom_prompt(self,q, axioms):
        llm_prompt = Prompt(f"{self.prompts_folder}/verify_axiom.yaml")

        prompt_params = {"QUESTION": q, "GENERAL_RULE": axioms}
        prompt = llm_prompt(prompt_params, few_shot=True)
        #prompt = llm_prompt(prompt_params)
        return prompt

    def relevant_facts_prompt(self, q, entity2descriptions, axioms, kg_facts, next_entity=None):
        
        #llm_prompt = Prompt("prompts/relevant_facts.yaml")
        llm_prompt = Prompt(f"{self.prompts_folder}/relevant_facts.yaml")
        entities = list(entity2descriptions.keys())
        hints = []
        for i, entity in enumerate(entities):
            hints.append(f"{i+1}- {entity}: {entity2descriptions[entity]}")
        entities_numbered = []
        facts_numbered = []
        for entity in kg_facts:
            entities_numbered.append(f"{1+len(entities_numbered)}- {entity}")
            for i, fact in enumerate(kg_facts[entity]):
                #fact_string = fact[2] + "has relation" + fact[1] + "with" + fact[0]
                #facts_numbered.append(f"{1+len(facts_numbered)}- {fact_string}")
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
               
        #prompt_params = {"QUESTION": q, "HINTS": hints, "GENERAL_RULE": axioms, "FACTS": kg_facts}
        if next_entity is None:
            prompt_params = {"QUESTION": q, "GENERAL_RULE": axioms, "FACTS": facts_numbered}
        else:
            prompt_params = {"QUESTION": q, "FACTS": facts_numbered}
            
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt

        


    def identify_missing_prompt(self, question, proposed_answer ,entity2descriptions, axioms, relevant_facts, selected_entities):
        
        #llm_prompt = Prompt("prompts/identify_missing.yaml")
        llm_prompt = Prompt(f"{self.prompts_folder}/identify_missing.yaml")
        entities = list(entity2descriptions.keys())
        hints = []
        for i, entity in enumerate(entities):
            hints.append(f"{i+1}- {entity}: {entity2descriptions[entity]}")
        facts_numbered = []
        for entity in relevant_facts:
            for i, fact in enumerate(relevant_facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        #prompt_params = {"QUESTION": question, "HINTS": hints, "GENERAL_RULE": axioms, "FACTS": facts_numbered, "PREVIOUS_ENTITIES": selected_entities}
        prompt_params = {"QUESTION": question,"HINTS": hints, "GENERAL_RULE": axioms, "FACTS": facts_numbered, "PROPOSED_ANSWER": proposed_answer}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt



    def decide_answer_prompt(self, question, entity2descriptions, axioms, relevant_facts):
        #llm_prompt = Prompt("prompts/decide_answer.yaml")

        #llm_prompt = Prompt(f"{self.prompts_folder}/decide_answer.yaml")
        llm_prompt = Prompt(f"{self.prompts_folder}/decide_answer.yaml")
        entities = list(entity2descriptions.keys())
        hints = []
        for i, entity in enumerate(entities):
            hints.append(f"{i+1}- {entity}: {entity2descriptions[entity]}")
        facts_numbered = []
        for entity in relevant_facts:
            for i, fact in enumerate(relevant_facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        axioms_numbered = []
        for i, axiom in enumerate(axioms):
            axioms_numbered.append(f"{i+1}- {axiom}")
        #prompt_params = {"QUESTION": question, "HINTS": hints, "GENERAL_RULE": axioms_numbered, "FACTS": facts_numbered}
        prompt_params = {"QUESTION": question, "GENERAL_RULE": axioms_numbered, "FACTS": facts_numbered}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt

    def verify_answer_prompt(self, question, entity2descriptions, axioms, relevant_facts, answer):
        #llm_prompt = Prompt("prompts/verify_answer.yaml")
        llm_prompt = Prompt(f"{self.prompts_folder}/verify_answer.yaml")
        entities = list(entity2descriptions.keys())
        hints = []
        for i, entity in enumerate(entities):
            hints.append(f"{i+1}- {entity}: {entity2descriptions[entity]}")
        facts_numbered = []
        for entity in relevant_facts:
            for i, fact in enumerate(relevant_facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        axioms_numbered = []
        for i, axiom in enumerate(axioms):
            axioms_numbered.append(f"{i+1}- {axiom}")
        prompt_params = {"QUESTION": question, "HINTS": hints, "GENERAL_RULE": axioms_numbered, "FACTS": facts_numbered, "PROPOSED_ANSWER": answer}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt

    def correct_answer_prompt(self, question, entity2descriptions, axioms, relevant_facts, answer, evaluation):

        #evaluation = evaluation.split(',', 1)[1] 
        #llm_prompt = Prompt("prompts/correct_answer.yaml")
        llm_prompt = Prompt(f"{self.prompts_folder}/correct_answer.yaml")
        entities = list(entity2descriptions.keys())
        hints = []
        for i, entity in enumerate(entities):
            hints.append(f"{i+1}- {entity}: {entity2descriptions[entity]}")
        facts_numbered = []
        for entity in relevant_facts:
            for i, fact in enumerate(relevant_facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
        axioms_numbered = []
        for i, axiom in enumerate(axioms):
            axioms_numbered.append(f"{i+1}- {axiom}")
        prompt_params = {"QUESTION": question, "GENERAL_RULE": axioms_numbered, "FACTS": facts_numbered, "PROPOSED_ANSWER": answer, "EVALUATION": evaluation}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt

    def decide_next_entity_prompt(self, question, answer_missing, relevant_facts, selected_entities):
        #llm_prompt = Prompt("prompts/decide_next_entity.yaml")
        llm_prompt = Prompt(f"{self.prompts_folder}/decide_next_entity.yaml")
        facts_numbered = []
        
        for entity in relevant_facts:
            for i, fact in enumerate(relevant_facts[entity]):
                facts_numbered.append(f"{1+len(facts_numbered)}- {fact}")
                
        prompt_params = {"QUESTION": question, "NEED": answer_missing, "FACTS": facts_numbered, "PREVIOUS_ENTITIES": selected_entities}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt
    
    
    def decide_polarity_prompt(self, sub_question, proposed_answer):
        #llm_prompt = Prompt("prompts/polarity_detection.yaml")
        llm_prompt = Prompt(f"{self.prompts_folder}/polarity_detection.yaml")
        prompt_params = {"QUESTION": sub_question, "PROPOSED": proposed_answer}
        prompt = llm_prompt(prompt_params, few_shot=True)
        return prompt
    
    #uncomment lines identified with $ for ablation study of entity linkers
    def identify_entities(self, q):

        

        entities = []

        message = self.identify_entities_prompt(q)
        response = self.get_response(message)
        selected_entities_index = response.find("Selected entity/entities:")
        entities = response[selected_entities_index + len("Selected entity/entities:"):].strip().split("\n")
        llm_entities = copy.deepcopy(entities)
        
        if self.refined is not None:
            
            refined_entities = self.refined(q)
            entities += refined_entities
        entities = list(set(entities))
        entities_final = [ent for ent in entities if ent is not None]

        with open("entities.txt", "a") as file:
            entity_Text = f"question:{q}\n LLM entities: {llm_entities}\n refined entities: {refined_entities}\n\n"
            file.write(entity_Text)


        return entities_final, message, response
        #return entities_final, None, None




    def generate_fact(self, q, entity2descriptions):
        # q is a signle question
        # return a single fact

        message = self.fact_generation_prompt(q, entity2descriptions)
        response = self.get_response(message)
        selected_rule_index = response.find("Rule:")
        if selected_rule_index == -1:
            fact = response.strip()
        else:
            fact = response[selected_rule_index + len("Rule:"):].strip()
        however_index = fact.find("However,")
        if however_index != -1:
            fact = fact[:however_index ].strip()
        unfortunately_index = fact.find("Unfortunately,")
        if unfortunately_index != -1:
            fact = fact[:unfortunately_index ].strip()
        therefore_index = fact.find("Therefore, it is not possible")
        if therefore_index != -1:
            fact = fact[:therefore_index ].strip()
        cannot_determine_index = fact.find("cannot determine")
        if cannot_determine_index != -1:
            fact = fact[:cannot_determine_index ].strip()
        the_facts_donot_support_index = fact.find("The facts provided do not")
        if the_facts_donot_support_index != -1:
            fact = fact[:the_facts_donot_support_index ].strip()
        return fact, message, response
    
    def generate_fact_next(self, q, entity2descriptions, axioms):
        # q is a signle question
        # return a single fact

        message = self.fact_generation_next_prompt(q, entity2descriptions, axioms)
        fact_prefix = "Rule: "
        response = self.get_response(message, temp=0.7)
        selected_rule_index = response.find("Rule:")
        fact = response[selected_rule_index + len("Rule:"):].strip()
        however_index = fact.find("However,")
        if however_index != -1:
            fact = fact[:however_index ].strip()
        return fact, message, response
    
    def select_axiom(self, q, entity2descriptions, axioms):
        message = self.select_axiom_prompt(q, entity2descriptions, axioms)
        response = self.get_response(message)
        selected_rule_index = response.find("Selected rule:")
        fact = response[selected_rule_index + len("Selected rule:"):].strip()
        return fact, message, response

    def verify_axiom(self, q, axioms):
        message = self.verify_axiom_prompt(q, axioms)
        response = self.get_response(message)
        selected_rule_index = response.find("is:")
        axiom = response[selected_rule_index + len("is:"):].strip()
        return axiom, message, response

    def identify_relevant_kgfacts(self, q, entity2descriptions, axioms, entity2facts, id, chat, next_entity=None):
        entity2relevant_facts = {}
        message = self.relevant_facts_prompt(q, entity2descriptions, axioms, entity2facts, next_entity)
        new_message = message.copy()
        answered = False
        while not answered:
            try:
                response = self.get_response(new_message)
                answered = True
            except:
                new_message = new_message[:-5]
                new_message.append(message[-1])
                response = self.get_response(new_message)
                answered = True
        chat.add_step(id, new_message, response)
        selected_answer_index = response.find("facts are:")
        final_facts = response[selected_answer_index+ len("facts are:"):].strip().split("\n")
        if selected_answer_index == -1:
            selected_answer_index = response.find("fact is:")
            final_facts = response[selected_answer_index+ len("fact is:"):].strip().split("\n")
        if selected_answer_index == -1:
            final_facts = []
        for entity in entity2facts:
            entity2relevant_facts[entity] = []
            for fact in final_facts:
                if "none" not in fact.lower():
                    if entity in fact:
                        entity2relevant_facts[entity].append(fact)       
                    
        return entity2relevant_facts, chat
    
    def attempt_answering(self, sub_question, entity2descriptions, axioms, relevant_facts):
        message = self.decide_answer_prompt(sub_question, entity2descriptions, axioms, relevant_facts)

        response = self.get_response(message)
        #answered = ((answer.lower().startswith("answer: yes")) or (answer.lower().startswith("answer: no"))) and ("i don't know" not in answer.lower() and "cannot determine" not in answer.lower())
        selected_answer_index = response.find("answer is")
        answer = response[selected_answer_index+ len("answer is"):].strip()
        answered = (": yes" in answer.lower() or ": no" in answer.lower()) and ("i don't know" not in answer.lower() and "cannot determine" not in answer.lower() and "don't have enough information" not in answer.lower())
        if ": yes" in answer.lower():
            binary_answer = True
        elif ": no" in answer.lower():
            binary_answer = False
        else:
            binary_answer = None
        return answered, binary_answer , response, message
    
    def verify_answer(self, sub_question, entity2descriptions, axioms, relevant_facts, answer):
        message = self.verify_answer_prompt(sub_question, entity2descriptions, axioms, relevant_facts, answer)
        #verification = self.get_response(message)
        #verified = verification.lower().startswith("evaluation: yes")
        response = self.get_response(message)
        selected_verification_index = response.find("is:")
        verification = response[selected_verification_index+ len("is:"):].strip()
        verified = verification.lower().startswith("yes") or ("yes" in verification.lower())
        return verified, response, message

    def correct_answer(self, sub_question, entity2descriptions, axioms, relevant_facts, answer, evaluation):
        message = self.correct_answer_prompt(sub_question, entity2descriptions, axioms, relevant_facts, answer, evaluation)
        response = self.get_response(message)
        modification = response.find("modified answer is:")
        answer = response[modification + len("modified answer is"):].strip()
        answered = ((": yes" in answer.lower()) or ("no" in answer.lower())) and ("i don't know" not in answer.lower() and "cannot determine" not in answer.lower() and "don't have enough information" not in answer.lower())
        if ": yes" in answer.lower():
            binary_answer = True
        elif ": no" in answer.lower():
            binary_answer = False
        else:
            binary_answer = None
        return answered, binary_answer, response, message


    def binarize_answer(self, answer):
        if ("i don't know" in answer.lower() or "cannot determine" in answer.lower()):
            return "idk"
        if answer.lower().startswith('answer: yes'):
            return True
        elif answer.lower().startswith('answer: no'):
            return False
        else:
            return None

    def identify_missing(self, sub_question, proposed_answer , entity2descriptions, axioms, relevant_facts, selected_entities):
        message = self.identify_missing_prompt(sub_question, proposed_answer, entity2descriptions, axioms, relevant_facts, selected_entities)
        answer = self.get_response(message)
        answerability= answer.lower().startswith("answer: nothing")
        answer_missing = answer.lstrip("Answer: ")
        return answerability, answer_missing, message
    
    def decide_next_entity(self, question, answer_missing, relevant_facts, selected_entities):
        message = self.decide_next_entity_prompt(question, answer_missing, relevant_facts, selected_entities)
        answer = self.get_response(message)
        next_entity = extract_entity_name(answer)
        return next_entity, message

    def decide_answer(self, sub_question, axioms, relevant_facts):
        message = self.decide_answer_prompt(sub_question, axioms, relevant_facts)
        answer = self.get_response(message).lower().startswith("yes")
        return answer, message

    def decide_next_hop(self, sub_question, relevant_facts, conversation_history):
        # q is a list of statements
        # return a list of statements
        message = self.next_hop_prompt(sub_question, relevant_facts, conversation_history)
        response = self.get_response(message)
        next_hop_decision = (response.lower().startswith("no") == False)
        return next_hop_decision, response, message

    def decide_polarity(self, sub_question, proposed_answer):
        prefixes = ["No,", "Yes,", "I don't know."]
        for prefix in prefixes:
            if proposed_answer.startswith(prefix):
                return proposed_answer[len(prefix):].lstrip(), None
        message = self.decide_polarity_prompt(sub_question, proposed_answer.split(',', 1)[1])
        response = self.get_response(message)
        return response, message
    