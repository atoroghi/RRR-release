from transformers import AutoTokenizer
import transformers
import torch
import time

from llms import Mistral, Mixtral
# model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True,"bnb_4bit_compute_dtype":torch.float16},
# )
# messages = [{"role": "user", "content": "Explain what a Mixture of Experts is in less than 100 words."}]
model = Mixtral()
t0 = time.time()
# prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
r = model.generate("Explain what a Mixture of Experts is in less than 100 words.")
print(time.time()-t0)
print(r)
# print(outputs[0]["generated_text"])