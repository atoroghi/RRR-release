
import sys, os, json, argparse

def main():
    parser = argparse.ArgumentParser(description='Extract cooking methods from recipes.')
    parser.add_argument('--recipes_file', type=str, help='Path to the JSON file containing recipes.')
    parser.add_argument('--recipe_id', type=str, help='Path to the output pickle file.')

    args = parser.parse_args()
    print(args)

    with open(os.path.join('data/Recipe-MPR', args.recipes_file), 'r') as file:
        recipes_data = json.load(file)
    print(recipes_data[0]['id'])
    i = 0
    for recipe in recipes_data:
        recipe_id = recipe.get('id')
        if recipe_id == args.recipe_id:
            recipe_instructions = recipe['instructions']
            for instruction in recipe_instructions:
                instruction_text = instruction['text']
                print(f'{i}. {instruction_text}')
                i += 1

if __name__ == '__main__':
    main()