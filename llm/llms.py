

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time


class LLM():
    def __init__(self):
        pass
    

    def generate(self,prompt,max_new_tokens=256):

        p = "[INST] "+ prompt+ " [/INST]"
        t0 = time.time()
        inputs = self.tokenizer(p, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        text = text.replace(p,'')

        return text



class Mistral(LLM):

    def __init__(self):
        
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        # print(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to('cuda')
        print('model device:',self.model.device)





class Mixtral(LLM):

    def __init__(self):
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        # print(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        print('model device:',self.model.device)


