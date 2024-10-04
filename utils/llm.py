import os

import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class LLM:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    #def __call__(self, messages, model="gpt-3.5-turbo-1106"):
    def __call__(self, messages, model="gpt-3.5-turbo-0613"):
        completion = openai.ChatCompletion.create(
            model=model, messages=messages)

        return completion.choices[0].message
