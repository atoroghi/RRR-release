import os.path as osp
import argparse
import pickle
import json
from tqdm import tqdm as tqdm
import sys
import numpy as np
import torch
from kbc.learn import kbc_model_load
import kbc.models
import copy



def get_args():
    t_norms = ['min', 'product', 'luka']
    parser = argparse.ArgumentParser(
    description="Complex Query Decomposition - Beam"
    )
    parser.add_argument('--query_path', help='Path to directory containing queries')
    parser.add_argument('--data_path', help='Path to directory containing KG')
    parser.add_argument(
    '--model_path',
    help="The path to the KBC model. Can be both relative and full"
    )
    parser.add_argument(
    '--candidates', default=5,
    help="Candidate amount for beam search"
    )
    parser.add_argument(
    '--t_norm', choices=t_norms, default='product',
    help="T-norms available are ".format(t_norms)
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    with open(osp.join(args.query_path, "500QA_KG_ingredients.json"), "r") as file:
        queries = json.load(file)

    with open(osp.join(args.data_path, 'name2code.pickle'), 'rb') as f:
        name2code = pickle.load(f)
    with open(osp.join(args.data_path, 'ent_id.pickle'), 'rb') as f:
        ent_id = pickle.load(f)

    kg_coded = np.genfromtxt(osp.join(args.data_path, 'KG_coded.txt'), delimiter='\t', dtype=np.int32)
    non_item_ents = [ent_id[ent] for ent in kg_coded[:,2]]
    non_item_ents = set(non_item_ents)
    item_ents = [ent_id[ent] for ent in kg_coded[:,0]]
    item_ents = set(item_ents)

    ing2recipe = {}
    for ing in non_item_ents:
        ing2recipe[ing] = set(kg_coded[kg_coded[:,2] == ing][:,0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Load model
    
    kbc, epoch, loss = kbc_model_load(args.model_path)

    for parameter in kbc.model.parameters():
        parameter.requires_grad = False
    # print(kbc.model.model_type())
    # print(kbc.model.entity_embeddings(torch.tensor([0]).to(device)))
    # a = kbc.model.entity_embeddings(torch.tensor([0]).to(device))
    # b = kbc.model.rel_embeddings(torch.tensor([0]).to(device))
    # print((kbc.model.backward_emb( a, b)).shape)


    pos_queries = ([q for q in queries if q['query_type']['Negated'] == 0])

    # Specific
    #pos_queries = ([q for q in pos_queries if q['query_type']['Specific'] == 1])

    # Commonsense
    #pos_queries = ([q for q in pos_queries if q['query_type']['Commonsense'] == 1])

    # Analogical
    #pos_queries = ([q for q in pos_queries if q['query_type']['Analogical'] == 1])

    # Temporal
    pos_queries = ([q for q in pos_queries if q['query_type']['Temporal'] == 1])

    

    rel_emb = kbc.model.rel_embeddings(torch.tensor([0]).to(device))

    answer_ranks = []
    answer_ranks_among_options = []

    for pos_query in tqdm(pos_queries):
        coded_options = [ent_id[name2code[option]] for option in pos_query['options']]
        coded_answer = ent_id[name2code[pos_query['answer']]]
        coded_kg_ings = []
        for ing in pos_query['KG_ingredients']:
            modified_ing = " "+ing.replace(" ", "%20")
            if modified_ing in name2code:
                coded_kg_ings.append(ent_id[name2code[modified_ing]])

        scores = torch.ones((1, len(ent_id))).to(device)

        for ing in coded_kg_ings:
            scores = torch.mul(scores, kbc.model.backward_emb(kbc.model.entity_embeddings(torch.tensor([ing]).to(device)), rel_emb))

        
        # filtering the set of non_item entities from the ranking (we're just comparing against other items not ings)
        for ent in non_item_ents:
            scores[0][ent] = -1e6
        # get the rank of answer among all items
        ranked_scores = torch.argsort(scores[0], descending=True)


        # filtering other recipes that satisfy the KG_ingredients
        
        # filtered_items = copy.deepcopy(item_ents)
        # for ing in coded_kg_ings:
        #     satisfying_items = ing2recipe[ing]
        #     filtered_items = filtered_items.intersection(satisfying_items)

        # for ent in filtered_items:
        #     mask[ent] = False

        answer_rank = (ranked_scores == coded_answer).nonzero().item()
        answer_ranks.append(answer_rank)

        options_scores = {opt: scores[0][opt] for opt in coded_options}
        rank_among_options = 1
        answer_score = scores[0][coded_answer]
        for option in coded_options:
            option_score = scores[0][option]
            if option_score > answer_score:
                rank_among_options += 1
        answer_ranks_among_options.append(rank_among_options)

        
    
    print(" hits at one:" , sum([1 for rank in answer_ranks if rank < 1])/len(answer_ranks))
    print(" hits at three:" , sum([1 for rank in answer_ranks if rank < 3])/len(answer_ranks))
    print(" hits at ten:" , sum([1 for rank in answer_ranks if rank < 10])/len(answer_ranks))

        
    print(" hits at one among options:" , sum([1 for rank in answer_ranks_among_options if rank < 2])/len(answer_ranks_among_options))

