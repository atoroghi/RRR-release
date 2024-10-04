import csv
import json, os, re
import argparse

# Open the CSV file for reading
parser = argparse.ArgumentParser(description='Extract cooking methods from recipes.')
parser.add_argument('--dataset_name', type=str, help='Path to the input CSV file')
args = parser.parse_args()
csv_file_path = os.path.join(os.getcwd() , 'data' , args.dataset_name, f'{args.dataset_name}_modified_new.csv')
json_file_path = 'CR-LT-SQA.json'

with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Initialize an empty list to store JSON entries
    json_entries = []

    # Iterate through each row in the CSV file
    for row in csv_reader:
        # Assuming the CSV has at least 2 columns
        if len(row) >= 2 and row[0].lower() != 'id':
            # Create a dictionary for each entry
            entry = {'id': row[0], 'query': row[2],'answer':row[3], 'KG Entities':row[5], 'Inference Rule': row[7]}
                     
            match = re.match(r"\((.*),\s*(.*),\s*\[(.*)\]\)", row[6])
            if match:
                head = match.group(1).strip()
                rel = match.group(2).strip()
                tails = [tail.strip() for tail in match.group(3).split(',')]
                output_triples = [f"({head}, {rel}, {tail})" for tail in tails]
                entry['KG Triples'] = ', '.join(output_triples)
            else:
                match = re.match(r"\(\[(.*)\],\s*(.*),\s*(.*)\)", row[6])
                if match:
                    heads = [head.strip() for head in match.group(1).split(',')]
                    rel = match.group(2).strip()
                    tail = match.group(3).strip()
                    output_triples = [f"({head}, {rel}, {tail})" for head in heads]
                    entry['KG Triples'] = ', '.join(output_triples)
                else:
                    entry['KG Triples'] = row[6]
            entry['Reasoning Steps'] = []
            # Append the entry to the list
            json_entries.append(entry)

# Open the JSON file for writing
with open(json_file_path, 'w') as json_file:
    # Write the list of entries as JSON to the file
    json.dump(json_entries, json_file, indent=2, ensure_ascii= False)

print(f'Conversion complete. JSON file saved at {json_file_path}')
