import os
from collections import defaultdict

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


class Chat:
    def __init__(self, save_file):
        self.save_file = save_file

        self.yaml = YAML()

        if os.path.isfile(save_file):
            with open(save_file, 'r', encoding="utf-8") as f:
                self.chats = defaultdict(list, YAML().load(f))
        else:
            self.chats = defaultdict(list)
    

    def add_step(self, id, prompt, response):
        if isinstance(response, str):
            response_dict = {"role": "assistant", "content": response}
            messages = prompt + [response_dict]
        else:
            if response is None:
                messages = []
            else:
                messages = prompt + [response]
        for i in range(len(messages)):
            message = dict(messages[i])
            message = {message["role"]: LiteralScalarString(message["content"])}
            messages[i] = message
            

        self.chats[id].append(messages)

    def save(self):
        with open(self.save_file, "w", encoding="utf-8") as f:
            self.yaml.dump(dict(self.chats), f)


if __name__ == "__main__":
    convo = Chat("../conversations/chat.yaml")

    prompt = [
        {"role": "system", "content": "system 1"},
        {"role": "user", "content": "user 1"},
        {"role": "assistant", "content": "assistant 1"},
        {"role": "user", "content": "user 2"},
    ]
    response = {"role": "assistant", "content": "I don't know."}

    convo.add_step("Q1", prompt, response)

    prompt = [
        {"role": "system", "content": "system 1"},
        {"role": "user", "content": "user 1"},
        {"role": "assistant", "content": "assistant 1"},
        {"role": "user", "content": "user 2"},
    ]
    response = {"role": "assistant", "content": "I don't know."}

    convo.add_step("Q1", prompt, response)

    prompt = [
        {"role": "system", "content": "system 1"},
        {"role": "user", "content": "user 1"},
        {"role": "assistant", "content": "assistant 1"},
        {"role": "user", "content": "user 2"},
        {"role": "assistant", "content": "assistant 2"},
        {"role": "user", "content": "user 3"},
    ]
    response = {"role": "assistant", "content": "I don't know."}

    convo.add_step("Q2", prompt, response)

    convo.save()
