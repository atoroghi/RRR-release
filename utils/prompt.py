from ruamel.yaml import YAML
from jinja2 import Template

# TODO: make input_format use Jinja for more flexibility


class Prompt:
    def __init__(self, prompt_file):
        yaml = YAML()

        with open(prompt_file, 'r') as f:
            self.prompt_info = yaml.load(f)

        self.input_template = Template(
            self.prompt_info['input_template'], trim_blocks=True, lstrip_blocks=True)

    def __call__(self, input, few_shot=False):
        messages = []

        messages.append(self.create_system_message())

        if few_shot:
            messages += self.create_few_shot_messages()

        input_message = {'role': 'user', 'content': self.format_input(input)}
        messages.append(input_message)

        return messages

    def format_input(self, input):
        return self.input_template.render(**input)

    def create_system_message(self):
        return {'role': 'system', 'content': self.prompt_info['system']}

    def create_few_shot_messages(self):
        messages = []

        for example in self.prompt_info['few-shot']:
            input = self.format_input(example['input'])
            output = example['output']

            user_msg = {'role': 'user', 'content': input}
            assistant_msg = {'role': 'assistant', 'content': output}

            messages += [user_msg, assistant_msg]

        return messages


if __name__ == '__main__':
    prompt = Prompt('../prompts/prompt.yaml')

    input = {'QUESTION': 'Is Bob older than 20?', 'FACTS': [
        '(Bob, age, 20)', '(Bob, country, Canada)']}

    messages = prompt(input, few_shot=True)

    for msg in messages:
        print(msg['role'])
        print(msg['content'])
        print('***')
