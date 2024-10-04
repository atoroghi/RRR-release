import json
import argparse
import re, sys, os
import pickle
import nltk  # Import nltk module
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('punkt')

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Perform stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]

    return lemmatized_tokens

def extract_cooking_methods(recipe_text):
    cooking_methods = ['braise', 'grill', 'roast', 'bake', 'blanche', 'stew', 'poach', 'steam', 'fry', 'boil', 'sauté', 'simmer', 'sous vide', 'broil', 'barbecue']
    methods_used = []

    # Preprocess recipe text
    preprocessed_text = preprocess_text(recipe_text)

    for method in cooking_methods:
        # Preprocess cooking method
        preprocessed_method = preprocess_text(method)

        # Check if any form of the cooking method is present in the preprocessed text
        if any(form in preprocessed_text for form in preprocessed_method):
            methods_used.append(method)

    return methods_used

def process_recipes(json_file):

    with open(os.path.join('data/Recipe-MPR', json_file), 'r') as file:
        recipes_data = json.load(file)

    cooking_methods = {}

    for i, recipe in enumerate(recipes_data):
        if i % 100 == 0:
            print(f'Processing recipe {i} of {len(recipes_data)}')
        recipe_id = recipe.get('id')
        if recipe_id:
            methods_used = set()
            recipe_instructions = recipe['instructions']
            for instruction in recipe_instructions:
                instruction_text = instruction['text']
                methods = extract_cooking_methods(instruction_text)
                for method in methods:
                    methods_used.add(method)

            cooking_methods[recipe_id] = methods_used

    return cooking_methods

def main():
    parser = argparse.ArgumentParser(description='Extract cooking methods from recipes.')
    parser.add_argument('--recipes_file', type=str, help='Path to the JSON file containing recipes.')
    parser.add_argument('--output_file', type=str, help='Path to the output pickle file.')
    args = parser.parse_args()

    cooking_methods = process_recipes(args.recipes_file)

    # for recipe_id, methods_used in cooking_methods.items():
    #     print(f'Recipe ID {recipe_id}: {methods_used}')
    with open(args.output_file, 'wb') as file:
        pickle.dump(cooking_methods, file)

if __name__ == '__main__':
    main()
