import os, sys
import os.path as osp
import json
import time
import enum
from collections import defaultdict
import subprocess
import pickle


from typing import List, Tuple

from collections import OrderedDict
import xml.etree.ElementTree
import numpy as np
import torch


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batches(size: int, batch_size: int) -> List[Tuple[int, int]]:
    max_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, max_batch)]
    return res
def create_instructions_bpl(chains,graph_type):
    instructions = []
    try:

            if graph_type == '1_2':
                #instructions.append("hop_0_1")
                instructions.append("hop_0")

            elif graph_type == '2_2':
                #instructions.append("intersect_0_1_2")
                instructions.append("intersect_0_1")
            elif graph_type == '2_3':
                #instructions.append("intersect_0_1_2_3")
                instructions.append("intersect_0_1_2")
            elif graph_type == '1_3':
                #instructions.append("hop_0_1")
                #instructions.append("hop_1_2")
                instructions.append("hop_0_1")
            elif graph_type == '1_4':
                #instructions.append("hop_0_1")
                #instructions.append("hop_1_2")
                #instructions.append("hop_2_3")
                instructions.append("hop_0_1_2")
                #instructions.append("hop_1_2")
            elif graph_type == '3_3':
                #instructions.append("hop_0_1")
                #instructions.append("intersect_1_2_3")
                instructions.append("hop_0_1")
                instructions.append("intersect_1_2")
            elif graph_type == '4_3':
                #instructions.append("intersect_0_1")
                #instructions.append("intersect_2_3")
                instructions.append("intersect_0_1")
                instructions.append("hop_1_2")

            else:
                raise NotImplementedError

    except RuntimeError as e:
        print(e)
        return instructions
    return instructions


def create_instructions(chains):
    instructions = []
    try:

        prev_start = None
        prev_end = None

        path_stack = []
        start_flag = True
        for chain_ind, chain in enumerate(chains):

            if start_flag:
                prev_end = chain[-1]
                start_flag = False
                continue
            if prev_end == chain[0]:
                instructions.append(f"hop_{chain_ind-1}_{chain_ind}")
                prev_end = chain[-1]
                prev_start = chain[0]

            elif prev_end == chain[-1]:

                prev_start = chain[0]
                prev_end = chain[-1]

                instructions.append(f"intersect_{chain_ind-1}_{chain_ind}")
            else:
                path_stack.append(([prev_start, prev_end],chain_ind-1))
                prev_start = chain[0]
                prev_end = chain[-1]
                start_flag = False
                continue

            if len(path_stack) > 0:

                path_prev_start = path_stack[-1][0][0]
                path_prev_end = path_stack[-1][0][-1]

                if path_prev_end == chain[-1]:

                    prev_start = chain[0]
                    prev_end = chain[-1]

                    instructions.append(f"intersect_{path_stack[-1][1]}_{chain_ind}")
                    path_stack.pop()
                    continue

        ans = []
        for inst in instructions:
            if ans:

                if 'inter' in inst and ('inter' in ans[-1]):
                        last_ind = inst.split("_")[-1]
                        ans[-1] = ans[-1]+f"_{last_ind}"
                else:
                    ans.append(inst)

            else:
                ans.append(inst)
        instructions = ans

    except RuntimeError as e:
        print(e)
        return instructions
    return instructions


def extract(elem, tag, drop_s):
  text = elem.find(tag).text
  if drop_s not in text: raise Exception(text)
  text = text.replace(drop_s, "")
  try:
    return int(text)
  except ValueError:
    return float(text)


def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))

def check_gpu():
    d = OrderedDict()
    d["time"] = time.time()

    cmd = ['nvidia-smi', '-q', '-x']
    cmd_out = subprocess.check_output(cmd)
    gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

    util = gpu.find("utilization")
    d["gpu_util"] = extract(util, "gpu_util", "%")

    d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
    d["mem_used_per"] = d["mem_used"] * 100 / 11171

    if d["gpu_util"] < 15 and d["mem_used"] < 2816 :
        msg = 'GPU status: Idle \n'
    else:
        msg = 'GPU status: Busy \n'

    now = time.strftime("%c")
    return ('\nUpdated at %s\nGPU utilization: %s %%\nVRAM used: %s %%\n%s\n' % (now, d["gpu_util"],d["mem_used_per"], msg))


class QuerDAG(enum.Enum):
    TYPE1_1 = "1_1"
    TYPE1_2 = "1_2"
    TYPE2_2 = "2_2"
    TYPE2_2_disj = "2_2_disj"
    TYPE1_3 = "1_3"
    TYPE1_4 = "1_4"
    TYPE2_3 = "2_3"
    TYPE3_3 = "3_3"
    TYPE4_3 = "4_3"
    TYPE4_3_disj = "4_3_disj"
    TYPE1_3_joint = '1_3_joint'
    TYPE1_2_seq = "1_2_seq"
    TYPE1_3_seq = "1_3_seq"
    TYPE2_2_seq = "2_2_seq"
    TYPE2_2_disj_seq = "2_2_disj_seq"
    TYPE2_3_seq = "2_3_seq"
    TYPE3_3_seq = "3_3_seq"
    TYPE4_3_seq = "4_3_seq"
    TYPE4_3_disj_seq = "4_3_disj_seq"
    TYPE1_1_seq = "1_1_seq"


class DynKBCSingleton:
    __instance = None
    # Singleton pattern to ensure that only one instance of the class is created
    @staticmethod
    def getInstance():
        """ Static access method. """
        if DynKBCSingleton.__instance == None:
            DynKBCSingleton()
        return DynKBCSingleton.__instance


    def set_attr(self, raw, kbc, chains, parts, intact_parts, target_ids_hard, keys_hard,
                 target_ids_complete, keys_complete,chain_instructions,
                 graph_type, lhs_norm, cuda, ent_id2fb, rel_id2fb, fb2name,
                possible_heads_emb, possible_tails_emb, users, items, ent_id):
        self.raw = raw
        self.kbc = kbc
        self.chains = chains
        self.parts = parts
        self.intact_parts = intact_parts


        self.target_ids_hard = target_ids_hard
        self.keys_hard = keys_hard


        self.target_ids_complete = target_ids_complete
        self.keys_complete = keys_complete

        self.cuda = True
        self.lhs_norm = lhs_norm
        self.chain_instructions = chain_instructions
        self.graph_type = graph_type
        self.ent_id2fb = ent_id2fb
        self.rel_id2fb = rel_id2fb
        self.fb2name = fb2name
        self.possible_heads_emb = possible_heads_emb
        self.possible_tails_emb = possible_tails_emb
        self.users = users
        self.items = items
        self.ent_id = ent_id
        self.__instance = self

    def __init__(self,raw = None, kbc = None, chains = None , parts = None, \
    target_ids_hard = None, keys_hard = None, target_ids_complete = None, keys_complete = None, \
    lhs_norm = None, chain_instructions = None, graph_type = None, cuda = None, possible_heads_emb=None,
    possible_tails_emb = None, users = None, items = None, ent_id = None):
        """ Virtually private constructor. """
        if DynKBCSingleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DynKBCSingleton.raw = raw
            DynKBCSingleton.kbc = kbc
            DynKBCSingleton.chains = chains
            DynKBCSingleton.parts = parts

            DynKBCSingleton.target_ids_hard = target_ids_hard
            DynKBCSingleton.keys_hard = keys_hard

            DynKBCSingleton.target_ids_complete = target_ids_complete
            DynKBCSingleton.keys_complete = keys_complete

            DynKBCSingleton.cuda = True
            DynKBCSingleton.lhs_norm = lhs_norm
            DynKBCSingleton.graph_type = graph_type
            DynKBCSingleton.chain_instructions = chain_instructions
            DynKBCSingleton.possible_heads_emb = possible_heads_emb
            DynKBCSingleton.possible_tails_emb = possible_tails_emb
            DynKBCSingleton.users = users
            DynKBCSingleton.items = items
            DynKBCSingleton.ent_id = ent_id
            DynKBCSingleton.__instance = self

    def set_eval_complete(self,target_ids_complete, keys_complete):
            self.target_ids_complete = target_ids_complete
            self.keys_complete = keys_complete
            self.__instance = self

def get_keys_and_targets_bpl(parts, users, items, graph_type):
    if len(parts) == 1:
        part1 = parts[0]
        part2 = None
        part3 = None
        part4 = None
    if len(parts) == 2:
        part1, part2 = parts
        part3 = None
        part4 = None
    if len(parts) == 3:
        part1, part2, part3 = parts
        part4 = None
    elif len(parts) == 4:
        part1, part2, part3, part4 = parts

    # "keys" are the unique chain queries we want to answer in a str format(e.g., (21,2,4)_(5,6,8))
    # target_ids is a dict that maps these keys to the target id of the chain query
    user_ids = {}
    item_ids = {}
    keys = []

    for chain_iter in range(len(part1)):
        # parts of the chain are concatenated to form a key
        if len(parts) == 4:
            key = part1[chain_iter] + part2[chain_iter] + part3[chain_iter] + part4[chain_iter]
        if len(parts) == 3:
            key = part1[chain_iter] + part2[chain_iter] + part3[chain_iter]
        if len(parts) == 2:
            key = part1[chain_iter] + part2[chain_iter]
        elif len(parts) == 1:
            key = part1[chain_iter]
        # then joins the key elements with underscores to create a string.
        key = '_'.join(str(e) for e in key)

        if key not in user_ids:
            user_ids[key] = []
            keys.append(key)
        if key not in item_ids:
            item_ids[key] = []
            keys.append(key)

        user_ids[key] = users[chain_iter]
        item_ids[key] = items[chain_iter]

    return user_ids, item_ids, keys


def get_keys_and_targets(parts, targets, graph_type):
    if len(parts) == 1:
        part1 = parts[0]
        part2, part3, part4, part5, part6, part7, part8, part9 = None, None, None, None, None,None, None, None
    if len(parts) == 2:
        part1, part2 = parts
        part3, part4, part5, part6, part7, part8, part9 = None, None, None, None,None, None, None
    elif len(parts) == 3:
        part1, part2, part3 = parts
        part4, part5, part6, part7, part8, part9 = None, None, None, None, None, None
    elif len(parts) == 4:
        part1, part2, part3, part4 = parts
        part5, part6, part7, part8, part9 = None, None, None, None, None
    elif len(parts) == 5:
        part1, part2, part3, part4, part5= parts
        part6, part7, part8, part9 = None, None, None, None
    elif len(parts) == 6:
        part1, part2, part3, part4, part5, part6 = parts
        part7, part8, part9 = None, None, None
    elif len(parts) == 9:
        part1, part2, part3, part4, part5, part6, part7, part8, part9 = parts
    # "keys" are the unique chain queries we want to answer in a str format(e.g., (21,2,4)_(5,6,8))
    # target_ids is a dict that maps these keys to the target id of the chain query
    target_ids = {}
    keys = []

    for chain_iter in range(len(part1)):
        # parts of the chain are concatenated to form a key
        if len(parts) == 9:
            key = part1[chain_iter] + part2[chain_iter] + part3[chain_iter] + part4[chain_iter] + part5[chain_iter] + part6[chain_iter] + part7[chain_iter] + part8[chain_iter] + part9[chain_iter]
        if len(parts) == 6:
            key = part1[chain_iter] + part2[chain_iter] + part3[chain_iter] + part4[chain_iter] + part5[chain_iter] + part6[chain_iter]
        elif len(parts) == 5:
            key = part1[chain_iter] + part2[chain_iter] + part3[chain_iter] + part4[chain_iter] + part5[chain_iter]
        elif len(parts) == 4:
            key = part1[chain_iter] + part2[chain_iter] + part3[chain_iter] + part4[chain_iter]
        elif len(parts) == 3:
            key = part1[chain_iter] + part2[chain_iter] + part3[chain_iter]
        elif len(parts) == 2:
            key = part1[chain_iter] + part2[chain_iter]
        elif len(parts) == 1:
            key = part1[chain_iter]
        # then joins the key elements with underscores to create a string.
        key = '_'.join(str(e) for e in key)

        if key not in target_ids:
            target_ids[key] = []
            keys.append(key)

        target_ids[key] = targets[chain_iter]

    return target_ids, keys

# takes each part of the chains and gets embeddings for each part

def preload_env(kbc_path, dataset, graph_type, mode="complete", kg_path=None,
                explain=False, valid_heads=None, valid_tails=None, ent_id=None):

    from kbc.learn import kbc_model_load

    env = DynKBCSingleton.getInstance()

    chain_instructions = []
    try:


        if env.kbc is not None:
            kbc = env.kbc
        else:
            kbc, epoch, loss = kbc_model_load(kbc_path)


        for parameter in kbc.model.parameters():
            parameter.requires_grad = False

        keys = []
        target_ids = {}
        # sees which type of query we are dealing with
        if QuerDAG.TYPE1_1.value == graph_type:

            raw = dataset.type1_1chain

            type1_1chain = []
            for i in range(len(raw)):
                type1_1chain.append(raw[i].data)

            part1 = [x['raw_chain'] for x in type1_1chain]

            flattened_part1 = []

            # [[A,b,C][C,d,[Es]]

            targets = []
            for chain_iter in range(len(part1)):
                flattened_part1.append([part1[chain_iter][0], part1[chain_iter][1],-(chain_iter+1234)])
                targets.append(part1[chain_iter][2])

            part1 = flattened_part1

            target_ids, keys = get_keys_and_targets([part1], targets, graph_type)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)

            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm+=torch.norm(lhs_emb)

            lhs_norm/= len(chain1[0])

            chains = [chain1]
            parts = [part1]
            part1_heads_emb = torch.zeros(chain1[0].shape, device=device)
            part1_tails_emb = torch.zeros(chain1[0].shape, device=device)
             
            for i in range(part1.shape[0]):
                # gets the relation id
                rel = int(part1[i][1])
                # gets the possible heads and tails of it
                possible_heads = torch.tensor(np.array(valid_heads[rel]).astype('int64') , device=device)
                # possible tails of this relation should also be possible heads of the rel of the second part of the chain
                possible_tails = torch.tensor(np.array(valid_tails[rel]).astype('int64'), device=device)
                possible_heads_embeddings = kbc.model.entity_embeddings(possible_heads)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)

                # gets the mean of the heads and tails
                mean_head = torch.mean(possible_heads_embeddings, dim=0)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part1_heads_emb[i] = mean_head
                part1_tails_emb[i] = mean_tail
            possible_heads_emb = [part1_heads_emb]
            possible_tails_emb = [part1_tails_emb]



        elif QuerDAG.TYPE1_2.value == graph_type:

            raw = dataset.type1_2chain
            # a list of all type1_2chains (only their data)
            type1_2chain = []
            for i in range(len(raw)):
                type1_2chain.append(raw[i].data)

            # type 1_2 chain has two parts. it separates them
            # part1: [[user, likes, item]], part2: [[item, rel, tail]]
            part1 = [x['raw_chain'][0] for x in type1_2chain]
            part2 = [x['raw_chain'][1] for x in type1_2chain]

            intact_part1 = part1.copy()
            intact_part2 = part2.copy()
            intact_parts = [intact_part1, intact_part2]

            flattened_part1 =[]
            flattened_part2 = []

            # [[A,b,C][C,d,[Es]]

            #targets = []
            #for chain_iter in range(len(part2)):
            #    # masks the target node (tail of the second part of the chain) with some code so that it is not used in the embedding
            #    # but part 1 remains the same
            #    flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
            #    flattened_part1.append(part1[chain_iter])
            #    # that target node is added to the targets list
            #    targets.append(part2[chain_iter][2])

            items = []
            users = []
            for chain_iter in range(len(part2)):
                # masks the target node (head of the second part of the chain) with some code so that it is not used in the embedding
                # but part 1 remains the same
                flattened_part2.append([-(chain_iter+1234) ,part2[chain_iter][1], part2[chain_iter][2][0]])
                flattened_part1.append([part1[chain_iter][0], part1[chain_iter][1], -(chain_iter+1234)])
                #flattened_part1.append([-(chain_iter+1234) , -(chain_iter+1234), -(chain_iter+1234)])
                # let's consider the item node as the target node for now
                items.append(part1[chain_iter][2])
                users.append(part1[chain_iter][0])

            part1 = flattened_part1
            part2 = flattened_part2

            # part1 = [115359, 47, -1234]
            # part2 = [-1234, 28, 17745]

            #targets = targets
            users = users
            items = items

            # "keys" is a list contanits unique chain queries we want to answer in a str format(e.g., (21,2,-1)_(-1,6,-1234))
            # target_ids is a dict that maps these keys to the target id of the chain query
            #target_ids, keys = get_keys_and_targets([part1, part2], targets, graph_type)
            user_ids, item_ids, keys = get_keys_and_targets_bpl([part1, part2], users, items, graph_type)
            # for this chain type it's ['hop_0_1']
            if not chain_instructions:
                chain_instructions = create_instructions_bpl([part1[0], part2[0]], graph_type)
           
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            # gets the embeddings for part 1s of the chain. It is a tuple of three tensors. each tensor is 5000* (emb_dim)
            # if the model is simple or complex, it's 5000* (2*emb_dim). chain1[0] is None. (now chain1[0] and chain1[2] are none)
            chain1 = kbc.model.get_full_embeddigns(part1)

            # gets the embeddings for part 2s of the chain (remember that the target node is masked)
            # now chain2[0] is None
            chain2 = kbc.model.get_full_embeddigns(part2)
            #lhs_norm seems not to be used
            #lhs_norm = 0.0
            #for lhs_emb in chain1[0]:
            #    lhs_norm+=torch.norm(lhs_emb)

            #lhs_norm/= len(chain1[0])
            # chains is the list of embeddings of nodes in chain parts
            chains = [chain1,chain2]
            # parts is the list of the nodes themselves (node ids)
            parts = [part1, part2]

            # from here on, my code for getting the mean of the head and tail for each part
            # for part 1:
            part1_heads_emb = torch.zeros(chain2[2].shape, device=device)
            part1_tails_emb = torch.zeros(chain2[2].shape, device=device)
             
            # for i in range(part1.shape[0]):
            #     # gets the relation id
            #     rel = int(part1[i][1])
            #     # we also need the relation of the second part of the chain for possible tails
            #     rel_2 = int(part2[i][1])
            #     # gets the possible heads and tails of it
            #     #valid_heads_ent = [ent_id[x] for x in valid_heads[rel]]
            #     valid_heads_ent = [x for x in valid_heads[rel]]
            #     possible_heads = torch.tensor(np.array(valid_heads_ent).astype('int64') , device=device)
            #     # possible tails of this relation should also be possible heads of the rel of the second part of the chain
            #     intersect = np.intersect1d(np.array(valid_tails[rel]), np.array(valid_heads[rel_2]))
            #     #valid_tails_ent = [ent_id[x] for x in intersect]
            #     valid_tails_ent = [x for x in intersect]
            #     possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)

            #     possible_heads_embeddings = kbc.model.entity_embeddings(possible_heads)
            #     possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
            #     # gets the mean of the heads and tails
            #     mean_head = torch.mean(possible_heads_embeddings, dim=0)
            #     mean_tail = torch.mean(possible_tails_embeddings, dim=0)
            #     part1_heads_emb[i] = mean_head
            #     part1_tails_emb[i] = mean_tail
             

            # for part 2:
            # possible heads for part 2 is the same as possible tails of part 1
            part2_heads_emb = part1_tails_emb.clone()
            part2_tails_emb = torch.zeros(chain2[1].shape, device=device)

            for i in range(part2.shape[0]):
                rel = int(part2[i][1])
                valid_tails_ent = [x for x in valid_tails[rel]]
                #valid_tails_ent = [ent_id[x] for x in valid_tails[rel]]
                possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)
                try:
                    possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                except:
                    print(possible_tails)
                    print(rel)
                    sys.exit()
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part2_tails_emb[i] = mean_tail

            possible_heads_emb = [part1_heads_emb, part2_heads_emb]
            possible_tails_emb = [part1_tails_emb, part2_tails_emb]


        elif QuerDAG.TYPE2_2.value == graph_type:
            raw = dataset.type2_2chain

            type2_2chain = []
            for i in range(len(raw)):
                type2_2chain.append(raw[i].data)

            # part1: [[user, likes, item]], part2: [[item, rel, tail]], part3: [[item, rel, tail]]
            part1 = [x['raw_chain'][0] for x in type2_2chain]
            part2 = [x['raw_chain'][1] for x in type2_2chain]
            part3 = [x['raw_chain'][2] for x in type2_2chain]
            intact_part1 = part1.copy()
            intact_part2 = part2.copy()
            intact_part3 = part3.copy()
            intact_parts = [intact_part1, intact_part2, intact_part3]

            flattened_part1 =[]
            flattened_part2 = []
            flattened_part3 = []

            users = []
            items = []
            for chain_iter in range(len(part2)):
                flattened_part3.append([-(chain_iter+1234),part3[chain_iter][1],part3[chain_iter][2]])
                flattened_part2.append([-(chain_iter+1234),part2[chain_iter][1],part2[chain_iter][2]])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                items.append(part1[chain_iter][2])
                users.append(part1[chain_iter][0])

            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3
            users = users
            items = items

            user_ids, item_ids, keys = get_keys_and_targets_bpl([part1, part2, part3], users, items, graph_type)

            if not chain_instructions:
                chain_instructions = create_instructions_bpl([part1[0], part2[0], part3[0]], graph_type)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)

            #lhs_norm = 0.0
            #for lhs_emb in chain1[0]:
            #    lhs_norm+=torch.norm(lhs_emb)

            #lhs_norm/= len(chain1[0])
            chains = [chain1,chain2, chain3]
            parts = [part1, part2, part3]

            # from here on, my code for getting the mean of the head and tail for each part
            # for part 1 (user, likes, item):
            part1_heads_emb = torch.zeros(chain1[0].shape, device=device)
            part1_tails_emb = torch.zeros(chain1[0].shape, device=device)

            for i in range(part1.shape[0]):
                # gets the relation id
                rel_1 = int(part1[i][1])
                # we also need the relation of the second part of the chain for possible tails
                rel_2 = int(part2[i][1])
                rel_3 = int(part3[i][1])
                valid_heads_ent = [x for x in valid_heads[rel_1]]
                possible_heads = torch.tensor(np.array(valid_heads_ent).astype('int64') , device=device)
                # possible tails must be possible for both relations in the case of QA
                # for recommendation, since the first head is user, its tail (item) must be valid for the second relation's head
                #first_intersect = np.intersect1d(np.array(valid_tails[rel_1]), np.array(valid_tails[rel_2]))
                first_intersect = np.intersect1d(np.array(valid_tails[rel_1]), np.array(valid_heads[rel_2]))
                second_intersect = np.intersect1d(np.array(valid_heads[rel_3]), first_intersect)
                #valid_tails_ent = [ent_id[x] for x in second_intersect]
                valid_tails_ent = second_intersect
                possible_tails = torch.tensor(second_intersect.astype('int64'), device=device)
                possible_heads_embeddings = kbc.model.entity_embeddings(possible_heads)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                # get means of the tails
                mean_head = torch.mean(possible_heads_embeddings, dim=0)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part1_heads_emb[i] = mean_head
                part1_tails_emb[i] = mean_tail

            possible_tails_emb = [part1_tails_emb]
            possible_heads_emb = [part1_heads_emb]
           

        elif QuerDAG.TYPE2_2_disj.value == graph_type:
            raw = dataset.type2_2_disj_chain

            type2_2chain = []
            for i in range(len(raw)):
                type2_2chain.append(raw[i].data)

            part1 = [x['raw_chain'][0] for x in type2_2chain]
            part2 = [x['raw_chain'][1] for x in type2_2chain]

            flattened_part1 =[]
            flattened_part2 = []

            targets = []
            for chain_iter in range(len(part2)):
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                targets.append(part2[chain_iter][2])


            part1 = flattened_part1
            part2 = flattened_part2
            targets = targets

            target_ids, keys = get_keys_and_targets([part1, part2], targets, graph_type)


            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0]])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)

            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm+=torch.norm(lhs_emb)

            lhs_norm/= len(chain1[0])
            chains = [chain1,chain2]
            parts = [part1,part2]
            # from here on, my code for getting the mean of the head and tail for each part
            # for part 1:
            part1_heads_emb = torch.zeros(chain1[0].shape, device=device)
            part1_tails_emb = torch.zeros(chain1[0].shape, device=device)

            for i in range(part1.shape[0]):
                # gets the relation id
                rel_1 = int(part1[i][1])
                # we also need the relation of the second part of the chain for possible tails
                rel_2 = int(part2[i][1])
                # possible tails must be possible for both relations
                possible_tails = torch.tensor(np.union1d(np.array(valid_tails[rel_1]), np.array(valid_tails[rel_2])).astype('int64'), device=device)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                # get means of the tails
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part1_tails_emb[i] = mean_tail

            possible_tails_emb = [part1_tails_emb]
            possible_heads_emb = [part1_heads_emb]


        elif QuerDAG.TYPE1_3.value == graph_type:
            raw = dataset.type1_3chain

            type1_3chain = []
            for i in range(len(raw)):
                type1_3chain.append(raw[i].data)

            # part1: [[user, likes, item]], part2: [[item, rel, tail1]], part3: [[tail1, rel, tail2]]
            part1 = [x['raw_chain'][0] for x in type1_3chain]
            part2 = [x['raw_chain'][1] for x in type1_3chain]
            part3 = [x['raw_chain'][2] for x in type1_3chain]
            intact_part1 = part1.copy()
            intact_part2 = part2.copy() 
            intact_part3 = part3.copy()
            intact_parts = [intact_part1, intact_part2, intact_part3]

            flattened_part1 =[]
            flattened_part2 = []
            flattened_part3 = []

            # [A,b,C][C,d,[Es]]
            items = []
            targets = []
            users = []
            for chain_iter in range(len(part3)):
                flattened_part3.append([-(chain_iter+1234),part3[chain_iter][1],part3[chain_iter][2]])
                flattened_part2.append([-(chain_iter+1234),part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                items.append(part1[chain_iter][2])
                users.append(part1[chain_iter][0])

            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3
            users = users
            items = items

            user_ids, item_ids, keys = get_keys_and_targets_bpl([part1, part2, part3], users, items, graph_type)

            if not chain_instructions:
                chain_instructions = create_instructions_bpl([part1[0], part2[0], part3[0]], graph_type)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)

            #lhs_norm = 0.0
            #for lhs_emb in chain1[0]:
            #    lhs_norm+=torch.norm(lhs_emb)

            #lhs_norm/= len(chain1[0])

            chains = [chain1,chain2,chain3]
            parts = [part1, part2, part3]

            # from here on, my code for getting the mean of the head and tail for each part
            # for part 1:
            part1_heads_emb = torch.zeros(chain1[0].shape, device=device)
            part1_tails_emb = torch.zeros(chain1[0].shape, device=device)

            for i in range(part1.shape[0]):
                # gets the relation id
                rel = int(part1[i][1])
                # we also need the relation of the second part of the chain for possible tails
                rel_2 = int(part2[i][1])
                # gets the possible heads and tails of it
                valid_heads_ent = [x for x in valid_heads[rel]]
                possible_heads = torch.tensor(np.array(valid_heads_ent).astype('int64') , device=device)
                # possible tails of this relation should also be possible heads of the rel of the second part of the chain
                intersect = np.intersect1d(np.array(valid_tails[rel]), np.array(valid_heads[rel_2]))
                valid_tails_ent = [x for x in intersect]
                possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)
                
                possible_heads_embeddings = kbc.model.entity_embeddings(possible_heads)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)

                # gets the mean of the heads and tails
                mean_head = torch.mean(possible_heads_embeddings, dim=0)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part1_heads_emb[i] = mean_head
                part1_tails_emb[i] = mean_tail

            # for part 2:
            # possible heads for part 2 is the same as possible tails of part 1
            part2_heads_emb = part1_tails_emb.clone()
            part2_tails_emb = torch.zeros(chain2[1].shape, device=device)

            for i in range(part2.shape[0]):
                # gets the relation id
                rel_2 = int(part2[i][1])
                # we also need the relation of the third part of the chain for possible tails
                rel_3 = int(part3[i][1])
                intersect = np.intersect1d(np.array(valid_tails[rel_2]), np.array(valid_heads[rel_3]))
                valid_tails_ent = [x for x in intersect]
                possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)
                # possible tails of this relation should also be possible heads of the rel of the third part of the chain
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)

                # gets the mean of the heads and tails
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part2_tails_emb[i] = mean_tail
            
            # for part 3:
            # possible heads for part 3 is the same as possible tails of part 2
            part3_heads_emb = part2_tails_emb.clone()
            part3_tails_emb = torch.zeros(chain3[1].shape, device=device)

            for i in range(part3.shape[0]):
                # gets the relation id
                rel_3 = int(part3[i][1])
                valid_tails_ent = [x for x in valid_tails[rel_3]]
                possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part3_tails_emb[i] = mean_tail
            
            possible_heads_emb = [part1_heads_emb, part2_heads_emb, part3_heads_emb]
            possible_tails_emb = [part1_tails_emb, part2_tails_emb, part3_tails_emb]

        elif QuerDAG.TYPE1_4.value == graph_type:
            raw = dataset.type1_4chain
            type1_4chain = []
            for i in range(len(raw)):
                type1_4chain.append(raw[i].data)

            # raw_chain: [ [user, likes, item], [item, rel1, tail1], [tail1, rel, tail2], [tail2, rel, tail3] ]
            part1 = [x['raw_chain'][0] for x in type1_4chain]
            part2 = [x['raw_chain'][1] for x in type1_4chain]
            part3 = [x['raw_chain'][2] for x in type1_4chain]
            part4 = [x['raw_chain'][3] for x in type1_4chain]
            intact_part1 = part1.copy()
            intact_part2 = part2.copy()
            intact_part3 = part3.copy()
            intact_part4 = part4.copy()
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4]
            flattened_part1 = []
            flattened_part2 = []
            flattened_part3 = []
            flattened_part4 = []

            items = []
            targets = []
            users = []
            for chain_iter in range(len(part4)):
                flattened_part4.append([-(chain_iter+1234),part4[chain_iter][1],part4[chain_iter][2]])
                flattened_part3.append([-(chain_iter+1234),part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([-(chain_iter+1234),part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                items.append(part1[chain_iter][2])
                users.append(part1[chain_iter][0])
            
            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3
            part4 = flattened_part4

            user_ids, item_ids, keys = get_keys_and_targets_bpl([part1, part2, part3, part4], users, items, graph_type)
            if not chain_instructions:
                chain_instructions = create_instructions_bpl([part1[0], part2[0], part3[0], part4[0]], graph_type)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)
            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)
            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)
            part4 = np.array(part4)
            part4 = torch.tensor(part4.astype('int64'), device=device)
            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)
            chain4 = kbc.model.get_full_embeddigns(part4)

            chains = [chain1,chain2,chain3, chain4]
            parts = [part1, part2, part3, part4]
            # for part 1:
            part1_heads_emb = torch.zeros(chain1[0].shape, device=device)
            part1_tails_emb = torch.zeros(chain1[0].shape, device=device)

            for i in range(part1.shape[0]):
                # gets the relation id
                rel = int(part1[i][1])
                # we also need the relation of the second part of the chain for possible tails
                rel_2 = int(part2[i][1])
                # gets the possible heads and tails of it
                valid_heads_ent = [x for x in valid_heads[rel]]
                possible_heads = torch.tensor(np.array(valid_heads_ent).astype('int64') , device=device)
                # possible tails of this relation should also be possible heads of the rel of the second part of the chain
                intersect = np.intersect1d(np.array(valid_tails[rel]), np.array(valid_heads[rel_2]))
                valid_tails_ent = [x for x in intersect]
                possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)
                
                possible_heads_embeddings = kbc.model.entity_embeddings(possible_heads)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)

                # gets the mean of the heads and tails
                mean_head = torch.mean(possible_heads_embeddings, dim=0)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part1_heads_emb[i] = mean_head
                part1_tails_emb[i] = mean_tail

            # for part 2:
            # possible heads for part 2 is the same as possible tails of part 1
            part2_heads_emb = part1_tails_emb.clone()
            part2_tails_emb = torch.zeros(chain2[1].shape, device=device)

            for i in range(part2.shape[0]):
                # gets the relation id
                rel_2 = int(part2[i][1])
                # we also need the relation of the third part of the chain for possible tails
                rel_3 = int(part3[i][1])
                intersect = np.intersect1d(np.array(valid_tails[rel_2]), np.array(valid_heads[rel_3]))
                valid_tails_ent = [x for x in intersect]
                possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)
                # possible tails of this relation should also be possible heads of the rel of the third part of the chain
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)

                # gets the mean of the heads and tails
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part2_tails_emb[i] = mean_tail

            # for part 3:
            # possible heads for part 3 is the same as possible tails of part 2
            part3_heads_emb = part2_tails_emb.clone()
            part3_tails_emb = torch.zeros(chain3[1].shape, device=device)
            for i in range(part3.shape[0]):
                rel_3 = int(part3[i][1])
                rel_4 = int(part4[i][1])
                intersect = np.intersect1d(np.array(valid_tails[rel_3]), np.array(valid_heads[rel_4]))
                valid_tails_ent = [x for x in intersect]
                possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part3_tails_emb[i] = mean_tail

            # for part 4:
            # possible heads for part 4 is the same as possible tails of part 3
            part4_heads_emb = part3_tails_emb.clone()
            part4_tails_emb = torch.zeros(chain4[1].shape, device=device)
            for i in range(part4.shape[0]):
                rel_4 = int(part4[i][1])
                valid_tails_ent = [x for x in valid_tails[rel_4]]
                possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part3_tails_emb[i] = mean_tail
            
            possible_heads_emb = [part1_heads_emb, part2_heads_emb, part3_heads_emb, part4_heads_emb]
            possible_tails_emb = [part1_tails_emb, part2_tails_emb, part3_tails_emb, part4_tails_emb]


        elif QuerDAG.TYPE2_3.value == graph_type:
            raw = dataset.type2_3chain

            type2_3chain = []
            for i in range(len(raw)):
                type2_3chain.append(raw[i].data)
            # part1: [[user, likes, item]], part2: [[item, rel, tail]], part3: [[item, rel, tail]], part4: [[item, rel, tail]]
            part1 = [x['raw_chain'][0] for x in type2_3chain]
            part2 = [x['raw_chain'][1] for x in type2_3chain]
            part3 = [x['raw_chain'][2] for x in type2_3chain]
            part4 = [x['raw_chain'][3] for x in type2_3chain]

            intact_part1 = part1.copy()
            intact_part2 = part2.copy()
            intact_part3 = part3.copy()
            intact_part4 = part4.copy()
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4]

            flattened_part1 = []
            flattened_part2 = []
            flattened_part3 = []
            flattened_part4 = []

            users = []
            items = []
            for chain_iter in range(len(part3)):
                flattened_part4.append([-(chain_iter+1234),part4[chain_iter][1],part4[chain_iter][2]])
                flattened_part3.append([-(chain_iter+1234),part3[chain_iter][1],part3[chain_iter][2]])
                flattened_part2.append([-(chain_iter+1234),part2[chain_iter][1],part2[chain_iter][2]])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                items.append(part1[chain_iter][2])
                users.append(part1[chain_iter][0])

            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3
            part4 = flattened_part4
            users = users
            items = items

            user_ids, item_ids, keys = get_keys_and_targets_bpl([part1, part2, part3, part4], users, items, graph_type)

            if not chain_instructions:
                chain_instructions = create_instructions_bpl([part1[0], part2[0], part3[0], part4[0]], graph_type)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)

            part4 = np.array(part4)
            part4 = torch.tensor(part4.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)
            chain4 = kbc.model.get_full_embeddigns(part4)


            #lhs_norm = 0.0
            #for lhs_emb in chain1[0]:
            #    lhs_norm+=torch.norm(lhs_emb)

            #lhs_norm/= len(chain1[0])

            chains = [chain1,chain2,chain3,chain4]
            parts = [part1,part2,part3,part4]
            # from here on, my code for getting the mean of the head and tail for each part
            part1_heads_emb = torch.zeros(chain1[0].shape, device=device)
            part1_tails_emb = torch.zeros(chain1[0].shape, device=device)

            for i in range(part1.shape[0]):
                # gets the relation id
                rel_1 = int(part1[i][1])
                # we also need the relation of the second part of the chain for possible tails
                rel_2 = int(part2[i][1])
                rel_3 = int(part3[i][1])
                rel_4 = int(part4[i][1])
                #first_intersect = np.intersect1d(np.array(valid_tails[rel_1]), np.array(valid_tails[rel_2])).astype('int64')
                first_intersect = np.intersect1d(np.array(valid_tails[rel_1]), np.array(valid_heads[rel_2]))
                second_intersect = np.intersect1d(first_intersect, np.array(valid_heads[rel_3])).astype('int64')
                third_intersect = np.intersect1d(second_intersect, np.array(valid_heads[rel_4])).astype('int64')
                #valid_tails_ent = [ent_id[x] for x in third_intersect]
                valid_tails_ent = third_intersect
                possible_tails = torch.tensor(valid_tails_ent, device=device)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                # gets the mean of the tails
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part1_tails_emb[i] = mean_tail
            
            possible_tails_emb = [part1_tails_emb]
            possible_heads_emb = [part1_heads_emb]


        elif QuerDAG.TYPE3_3.value == graph_type:
            # raw_chains = [[[user, likes, item], [item, rel, anchor], [item, rel, tail], [tail, rel , anchor]]]

            raw = dataset.type3_3chain

            type3_3chain = []
            for i in range(len(raw)):
                type3_3chain.append(raw[i].data)
            # part1: [user, likes, item], part2: [item, rel, anchor], part3: [item, rel, tail], part4: [tail, rel , anchor]

            part1 = [x['raw_chain'][0] for x in type3_3chain]
            part2 = [x['raw_chain'][1] for x in type3_3chain]
            part3 = [x['raw_chain'][2] for x in type3_3chain]
            part4 = [x['raw_chain'][3] for x in type3_3chain]
            intact_part1 = part1.copy()
            intact_part2 = part2.copy()
            intact_part3 = part3.copy()
            intact_part4 = part4.copy()
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4]

            flattened_part1 =[]
            flattened_part2 = []
            flattened_part3 = []
            flattened_part4 = []

            users = []
            items = []
            for chain_iter in range(len(part4)):
                flattened_part4.append([-(chain_iter+1234),part4[chain_iter][1],part4[chain_iter][2]])
                flattened_part3.append([-(chain_iter+1234),part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([-(chain_iter+1234),part2[chain_iter][1],part2[chain_iter][2]])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                items.append(part1[chain_iter][2])
                users.append(part1[chain_iter][0])


            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3
            part4 = flattened_part4
            users = users
            items = items

            user_ids, item_ids, keys = get_keys_and_targets_bpl([part1, part2, part3, part4], users, items, graph_type)

            if not chain_instructions:
                chain_instructions = create_instructions_bpl([part1[0], part2[0], part3[0]], graph_type)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)
            part4 = np.array(part4)
            part4 = torch.tensor(part4.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)
            chain4 = kbc.model.get_full_embeddigns(part4)

            #lhs_norm = 0.0
            #for lhs_emb in chain1[0]:
            #    lhs_norm+=torch.norm(lhs_emb)

            #lhs_norm/= len(chain1[0])

            chains = [chain1,chain2,chain3,chain4]
            parts = [part1, part2, part3, part4]

            # from here on, my code for getting the mean of the head and tail for each part
            part1_heads_emb = torch.zeros(chain1[0].shape, device=device)
            part1_tails_emb = torch.zeros(chain1[0].shape, device=device)

            for i in range(part1.shape[0]):
                # gets the relation id
                rel_1 = int(part1[i][1])
                # we also need the relation of the second part of the chain for possible tails
                rel_2 = int(part2[i][1])
                rel_3 = int(part3[i][1])
                # gets the possible heads and tails of it
                valid_heads_ent = [x for x in valid_heads[rel_1]]
                possible_heads = torch.tensor(np.array(valid_heads_ent).astype('int64') , device=device)
                # possible tails of this relation should also be possible heads of the rel of the second part of the chain
                first_intersect = np.intersect1d(np.array(valid_tails[rel_1]), np.array(valid_heads[rel_2]))
                second_intersect = np.intersect1d(np.array(valid_heads[rel_3]), first_intersect)
                valid_tails_ent = second_intersect
                possible_tails = torch.tensor(second_intersect.astype('int64'), device=device)
                possible_heads_embeddings = kbc.model.entity_embeddings(possible_heads)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                # gets the mean of the heads
                mean_head = torch.mean(possible_heads_embeddings, dim=0)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part1_heads_emb[i] = mean_head
                part1_tails_emb[i] = mean_tail

            #for part 2:
            part2_heads_emb = part1_tails_emb.clone()
            part3_heads_emb = part1_tails_emb.clone()
            part2_tails_emb = torch.zeros(chain1[0].shape, device=device)
            part3_tails_emb = torch.zeros(chain1[0].shape, device=device)

            for i in range(part3.shape[0]):
                rel_3 = int(part3[i][1])
                rel_4 = int(part4[i][1])
                # possible tails must be possible for both relations
                intersect = np.intersect1d(np.array(valid_tails[rel_3]), np.array(valid_heads[rel_4]))
                valid_tails_ent = [x for x in intersect]
                possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)
                # gets the mean of the tails
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part3_tails_emb[i] = mean_tail

            possible_heads_emb = [part1_heads_emb, part2_heads_emb, part3_heads_emb]
            possible_tails_emb = [part1_tails_emb, part2_tails_emb, part3_tails_emb]

        elif QuerDAG.TYPE4_3.value == graph_type:
            # raw_chains = [[[user, likes, item], [item, rel, tail1], [tail1, rel, anchor1], [tail1, rel , anchor2]]]
            raw = dataset.type4_3chain

            type4_3chain = []
            for i in range(len(raw)):
                type4_3chain.append(raw[i].data)


            part1 = [x['raw_chain'][0] for x in type4_3chain]
            part2 = [x['raw_chain'][1] for x in type4_3chain]
            part3 = [x['raw_chain'][2] for x in type4_3chain]
            part4 = [x['raw_chain'][3] for x in type4_3chain]

            intact_part1 = part1.copy()
            intact_part2 = part2.copy()
            intact_part3 = part3.copy()
            intact_part4 = part4.copy()
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4]

            flattened_part1 =[]
            flattened_part2 = []
            flattened_part3 = []
            flattened_part4 = []

            items = []
            users = []
            for chain_iter in range(len(part4)):
                flattened_part4.append([-(chain_iter+1234),part4[chain_iter][1],part4[chain_iter][2]])
                flattened_part3.append([-(chain_iter+1234),part3[chain_iter][1],part3[chain_iter][2]])
                flattened_part2.append([-(chain_iter+1234),part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                items.append(part1[chain_iter][2])
                users.append(part1[chain_iter][0])

            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3
            part4 = flattened_part4

            users = users
            items = items
            user_ids, item_ids, keys = get_keys_and_targets_bpl([part1, part2, part3, part4], users, items, graph_type)

            if not chain_instructions:
                chain_instructions = create_instructions_bpl([part1[0], part2[0], part3[0], part4[0]], graph_type)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)

            part4 = np.array(part4)
            part4 = torch.tensor(part4.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)
            chain4 = kbc.model.get_full_embeddigns(part4)


            #lhs_norm = 0.0
            #for lhs_emb in chain1[0]:
            #    lhs_norm+=torch.norm(lhs_emb)

            #lhs_norm/= len(chain1[0])
            chains = [chain1,chain2,chain3,chain4]
            parts = [part1,part2,part3,part4]
            # from here on, my code for getting the mean of the head and tail for each part
            part1_heads_emb = torch.zeros(chain1[0].shape, device=device)
            part1_tails_emb = torch.zeros(chain1[0].shape, device=device)

            for i in range(part1.shape[0]):
                # gets the relation of the first part of the chain
                rel_1 = int(part1[i][1])
                # we also need the relation of the second part of the chain for the possible tails
                rel_2 = int(part2[i][1])
                # possible tails must be possible for both relations and a possible head for the second relation
                valid_heads_ent = [x for x in valid_heads[rel_1]]
                possible_heads = torch.tensor(np.array(valid_heads_ent).astype('int64') , device=device)
                intersect = np.intersect1d(np.array(valid_tails[rel_1]), np.array(valid_heads[rel_2]))
                valid_tails_ent = [x for x in intersect]
                possible_tails = torch.tensor(np.array(valid_tails_ent).astype('int64'), device=device)
                possible_heads_embeddings = kbc.model.entity_embeddings(possible_heads)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                mean_head = torch.mean(possible_heads_embeddings, dim=0)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part1_heads_emb[i] = mean_head
                part1_tails_emb[i] = mean_tail
                
            #for part 2:
            part2_heads_emb = part1_tails_emb.clone()
            part2_tails_emb = torch.zeros(chain2[1].shape, device=device)
            for i in range(part2.shape[0]):
                rel_2 = int(part2[i][1])
                rel_3 = int(part3[i][1])
                rel_4 = int(part4[i][1])
                first_intersect = np.intersect1d(np.array(valid_tails[rel_2]), np.array(valid_heads[rel_3]))
                second_intersect = np.intersect1d(np.array(valid_heads[rel_4]), first_intersect)
                valid_tails_ent = second_intersect
                possible_tails = torch.tensor(second_intersect.astype('int64'), device=device)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part2_tails_emb[i] = mean_tail
            possible_heads_emb = [part1_heads_emb, part2_heads_emb]
            possible_tails_emb = [part1_tails_emb, part2_tails_emb]


        elif QuerDAG.TYPE4_3_disj.value == graph_type:
            raw = dataset.type4_3_disj_chain

            type4_3chain = []
            for i in range(len(raw)):
                type4_3chain.append(raw[i].data)


            part1 = [x['raw_chain'][0] for x in type4_3chain]
            part2 = [x['raw_chain'][1] for x in type4_3chain]
            part3 = [x['raw_chain'][2] for x in type4_3chain]


            flattened_part1 =[]
            flattened_part2 = []
            flattened_part3 = []

            # [A,r_1,B][C,r_2,B][B, r_3, [D's]]
            targets = []
            for chain_iter in range(len(part3)):
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],part2[chain_iter][2]])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],part1[chain_iter][2]])
                targets.append(part3[chain_iter][2])

            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3
            targets = targets

            target_ids, keys = get_keys_and_targets([part1, part2, part3], targets, graph_type)

            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0]])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)


            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm+=torch.norm(lhs_emb)

            lhs_norm/= len(chain1[0])
            chains = [chain1,chain2,chain3]
            parts = [part1,part2,part3]
            # from here on, my code for getting the mean of the head and tail for each part
            part1_heads_emb = torch.zeros(chain1[0].shape, device=device)
            part1_tails_emb = torch.zeros(chain1[0].shape, device=device)

            for i in range(part1.shape[0]):
                # gets the relation of the first part of the chain
                rel_1 = int(part1[i][1])
                # we also need the relation of the second part of the chain for the possible tails
                rel_2 = int(part2[i][1])
                rel_3 = int(part3[i][1])
                # possible tails must be possible for both relations and a possible head for the second relation
                first_intersect = np.union1d(np.array(valid_tails[rel_1]), np.array(valid_tails[rel_2])).astype('int64')
                possible_tails = torch.tensor(np.intersect1d(first_intersect, np.array(valid_heads[rel_3])).astype('int64'), device=device)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                # gets the mean of the heads
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part1_tails_emb[i] = mean_tail
            
            #for part 2:
            part2_heads_emb = torch.zeros(chain2[0].shape, device=device)
            part2_tails_emb = part1_tails_emb.clone()

            #for part 3:
            part3_heads_emb = part1_tails_emb.clone()
            part3_tails_emb = torch.zeros(chain3[1].shape, device=device)
            for i in range(part3.shape[0]):
                rel_3 = int(part3[i][1])
                possible_tails = torch.tensor(np.array(valid_tails[rel_3]).astype('int64'), device=device)
                possible_tails_embeddings = kbc.model.entity_embeddings(possible_tails)
                # gets the mean of the heads
                mean_tail = torch.mean(possible_tails_embeddings, dim=0)
                part3_tails_emb[i] = mean_tail
            
            possible_tails_emb = [part1_tails_emb, part2_tails_emb, part3_tails_emb]
            possible_heads_emb = [part1_heads_emb, part2_heads_emb, part3_heads_emb]

        elif QuerDAG.TYPE1_1_seq.value == graph_type:
            raw = dataset.type1_1chain
            type1_1chain = []
            for i in range(len(raw)):
                type1_1chain.append(raw[i].data)

            part1 = [x['raw_chain'][0] for x in type1_1chain]
            part2 = [x['raw_chain'][1] for x in type1_1chain]
            part3 = [x['raw_chain'][2] for x in type1_1chain]
            intact_part1 = part1.copy(); intact_part2 = part2.copy(); intact_part3 = part3.copy()
            flattened_part1 = []; flattened_part2 = []; flattened_part3 = []
            targets = []
            for chain_iter in range(len(part3)):
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                targets.append(part3[chain_iter][2])
            part1 = flattened_part1; part2 = flattened_part2; part3 = flattened_part3
            target_ids, keys = get_keys_and_targets([part1,part2,part3], targets, graph_type)
            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0]])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1); part1 = torch.tensor(part1.astype('int64'), device=device)
            part2 = np.array(part2); part2 = torch.tensor(part2.astype('int64'), device=device)
            part3 = np.array(part3); part3 = torch.tensor(part3.astype('int64'), device=device)     
            chain1 = kbc.model.get_full_embeddigns(part1); chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)
            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm += torch.norm(lhs_emb)
            lhs_norm /= len(chain1[0])
            chains = [chain1, chain2, chain3]
            parts = [part1, part2, part3]
            intact_parts = [intact_part1, intact_part2, intact_part3]
            possible_heads_emb = []; possible_tails_emb = []; users=[]; items=[]

        
        elif QuerDAG.TYPE1_2_seq.value == graph_type:
            raw = dataset.type1_2chain
            type1_2chain = []
            for i in range(len(raw)):
                type1_2chain.append(raw[i].data)
            part1 = [x['raw_chain'][0] for x in type1_2chain]
            part2 = [x['raw_chain'][1] for x in type1_2chain]
            part3 = [x['raw_chain'][2] for x in type1_2chain]
            part4 = [x['raw_chain'][3] for x in type1_2chain]
            intact_part1 = part1.copy(); intact_part2 = part2.copy(); intact_part3 = part3.copy(); intact_part4 = part4.copy()
            flattened_part1 = []; flattened_part2 = []; flattened_part3 = []; flattened_part4 = []
            targets = []
            
            for chain_iter in range(len(part4)):
                # check this
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part4.append([-(chain_iter+1234),part4[chain_iter][1],-(chain_iter+1234)])
                targets.append(part4[chain_iter][2])
            part1 = flattened_part1; part2 = flattened_part2; part3 = flattened_part3; part4 = flattened_part4
            target_ids, keys = get_keys_and_targets([part1,part2,part3,part4], targets, graph_type)
            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0], part4[0]])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1); part1 = torch.tensor(part1.astype('int64'), device=device)
            part2 = np.array(part2); part2 = torch.tensor(part2.astype('int64'), device=device)
            part3 = np.array(part3); part3 = torch.tensor(part3.astype('int64'), device=device)
            part4 = np.array(part4); part4 = torch.tensor(part4.astype('int64'), device=device)
            chain1 = kbc.model.get_full_embeddigns(part1); chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3); chain4 = kbc.model.get_full_embeddigns(part4)
            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm += torch.norm(lhs_emb)
            lhs_norm /= len(chain1[0])
            chains = [chain1, chain2, chain3, chain4]
            parts = [part1, part2, part3, part4]
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4]
            possible_heads_emb = []; possible_tails_emb = []; users=[]; items=[]
        elif QuerDAG.TYPE1_3_seq.value == graph_type:
            raw = dataset.type1_3chain
            type1_3chain = []
            for i in range(len(raw)):
                type1_3chain.append(raw[i].data)
            part1 = [x['raw_chain'][0] for x in type1_3chain]
            part2 = [x['raw_chain'][1] for x in type1_3chain]
            part3 = [x['raw_chain'][2] for x in type1_3chain]
            part4 = [x['raw_chain'][3] for x in type1_3chain]
            part5 = [x['raw_chain'][4] for x in type1_3chain]
            intact_part1 = part1.copy(); intact_part2 = part2.copy(); intact_part3 = part3.copy(); intact_part4 = part4.copy(); intact_part5 = part5.copy()
            flattened_part1 = []; flattened_part2 = []; flattened_part3 = []; flattened_part4 = []; flattened_part5 = []
            targets = []
            for chain_iter in range(len(part5)):
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part4.append([-(chain_iter+1234),part4[chain_iter][1],-(chain_iter+1234)])
                flattened_part5.append([-(chain_iter+1234),part5[chain_iter][1],-(chain_iter+1234)])
                targets.append(part5[chain_iter][2])
            part1 = flattened_part1; part2 = flattened_part2; part3 = flattened_part3; part4 = flattened_part4; part5 = flattened_part5
            target_ids, keys = get_keys_and_targets([part1,part2,part3,part4,part5], targets, graph_type)
            # check instructions
            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0], part4[0], part5[0]])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1); part1 = torch.tensor(part1.astype('int64'), device=device)
            part2 = np.array(part2); part2 = torch.tensor(part2.astype('int64'), device=device)
            part3 = np.array(part3); part3 = torch.tensor(part3.astype('int64'), device=device)
            part4 = np.array(part4); part4 = torch.tensor(part4.astype('int64'), device=device)
            part5 = np.array(part5); part5 = torch.tensor(part5.astype('int64'), device=device)
            chain1 = kbc.model.get_full_embeddigns(part1); chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3); chain4 = kbc.model.get_full_embeddigns(part4)
            chain5 = kbc.model.get_full_embeddigns(part5)
            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm += torch.norm(lhs_emb)
            lhs_norm /= len(chain1[0])
            chains = [chain1, chain2, chain3, chain4, chain5]
            parts = [part1, part2, part3, part4, part5]
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4, intact_part5]
            possible_heads_emb = []; possible_tails_emb = []; users=[]; items=[]
        
        elif QuerDAG.TYPE2_2_seq.value == graph_type:
            raw = dataset.type2_2chain
            type2_2chain = []
            for i in range(len(raw)):
                type2_2chain.append(raw[i].data)
            part1 = [x['raw_chain'][0] for x in type2_2chain]
            part2 = [x['raw_chain'][1] for x in type2_2chain]
            part3 = [x['raw_chain'][2] for x in type2_2chain]
            part4 = [x['raw_chain'][3] for x in type2_2chain]
            part5 = [x['raw_chain'][4] for x in type2_2chain]
            part6 = [x['raw_chain'][5] for x in type2_2chain]
            intact_part1 = part1.copy(); intact_part2 = part2.copy(); intact_part3 = part3.copy(); intact_part4 = part4.copy()
            intact_part5 = part5.copy(); intact_part6 = part6.copy()
            flattened_part1 = []; flattened_part2 = []; flattened_part3 = []; flattened_part4 = []
            flattened_part5 = []; flattened_part6 = []
            targets = []
            for chain_iter in range(len(part2)):
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part4.append([part4[chain_iter][0],part4[chain_iter][1],-(chain_iter+1234)])
                flattened_part5.append([part5[chain_iter][0],part5[chain_iter][1],-(chain_iter+1234)])
                flattened_part6.append([part6[chain_iter][0],part6[chain_iter][1],-(chain_iter+1234)])
                targets.append(part2[chain_iter][2])
            part1 = flattened_part1; part2 = flattened_part2; part3 = flattened_part3; part4 = flattened_part4
            part5 = flattened_part5; part6 = flattened_part6

            target_ids, keys = get_keys_and_targets([part1,part2, part3, part4, part5, part6], targets, graph_type)  
            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0], part4[0], part5[0], part6[0]])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1); part1 = torch.tensor(part1.astype('int64'), device=device)
            part2 = np.array(part2); part2 = torch.tensor(part2.astype('int64'), device=device)
            part3 = np.array(part3); part3 = torch.tensor(part3.astype('int64'), device=device)
            part4 = np.array(part4); part4 = torch.tensor(part4.astype('int64'), device=device)
            part5 = np.array(part5); part5 = torch.tensor(part5.astype('int64'), device=device)
            part6 = np.array(part6); part6 = torch.tensor(part6.astype('int64'), device=device)
            chain1 = kbc.model.get_full_embeddigns(part1); chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3); chain4 = kbc.model.get_full_embeddigns(part4)
            chain5 = kbc.model.get_full_embeddigns(part5); chain6 = kbc.model.get_full_embeddigns(part6)
            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm += torch.norm(lhs_emb)
            lhs_norm /= len(chain1[0])
            chains = [chain1, chain2, chain3, chain4, chain5, chain6]
            parts = [part1, part2, part3, part4, part5, part6]
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4, intact_part5, intact_part6]
            possible_heads_emb = []; possible_tails_emb = []; users=[]; items=[]
        
        elif QuerDAG.TYPE2_2_disj_seq.value == graph_type:
            raw = dataset.type2_2chain_u
            type2_2chain = []
            for i in range(len(raw)):
                type2_2chain.append(raw[i].data)
            part1 = [x['raw_chain'][0] for x in type2_2chain]
            part2 = [x['raw_chain'][1] for x in type2_2chain]
            part3 = [x['raw_chain'][2] for x in type2_2chain]
            part4 = [x['raw_chain'][3] for x in type2_2chain]
            part5 = [x['raw_chain'][4] for x in type2_2chain]
            part6 = [x['raw_chain'][5] for x in type2_2chain]
            intact_part1 = part1.copy(); intact_part2 = part2.copy(); intact_part3 = part3.copy(); intact_part4 = part4.copy()
            intact_part5 = part5.copy(); intact_part6 = part6.copy()
            flattened_part1 = []; flattened_part2 = []; flattened_part3 = []; flattened_part4 = []
            flattened_part5 = []; flattened_part6 = []
            targets = []
            for chain_iter in range(len(part2)):
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part4.append([part4[chain_iter][0],part4[chain_iter][1],-(chain_iter+1234)])
                flattened_part5.append([part5[chain_iter][0],part5[chain_iter][1],-(chain_iter+1234)])
                flattened_part6.append([part6[chain_iter][0],part6[chain_iter][1],-(chain_iter+1234)])
                targets.append(part2[chain_iter][2])
            part1 = flattened_part1; part2 = flattened_part2; part3 = flattened_part3; part4 = flattened_part4
            part5 = flattened_part5; part6 = flattened_part6
            target_ids, keys = get_keys_and_targets([part1,part2, part3, part4, part5, part6], targets, graph_type)  
            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0], part4[0], part5[0], part6[0]])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1); part1 = torch.tensor(part1.astype('int64'), device=device)
            part2 = np.array(part2); part2 = torch.tensor(part2.astype('int64'), device=device)
            part3 = np.array(part3); part3 = torch.tensor(part3.astype('int64'), device=device)
            part4 = np.array(part4); part4 = torch.tensor(part4.astype('int64'), device=device)
            part5 = np.array(part5); part5 = torch.tensor(part5.astype('int64'), device=device)
            part6 = np.array(part6); part6 = torch.tensor(part6.astype('int64'), device=device)
            chain1 = kbc.model.get_full_embeddigns(part1); chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3); chain4 = kbc.model.get_full_embeddigns(part4)
            chain5 = kbc.model.get_full_embeddigns(part5); chain6 = kbc.model.get_full_embeddigns(part6)
            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm += torch.norm(lhs_emb)
            lhs_norm /= len(chain1[0])
            chains = [chain1, chain2, chain3, chain4, chain5, chain6]
            parts = [part1, part2, part3, part4, part5, part6]
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4, intact_part5, intact_part6]
            possible_heads_emb = []; possible_tails_emb = []; users=[]; items=[]
        

        elif QuerDAG.TYPE2_3_seq.value == graph_type:
            raw = dataset.type2_3chain
            type2_3chain = []
            for i in range(len(raw)):
                type2_3chain.append(raw[i].data)
            part1 = [x['raw_chain'][0] for x in type2_3chain]
            part2 = [x['raw_chain'][1] for x in type2_3chain]
            part3 = [x['raw_chain'][2] for x in type2_3chain]
            part4 = [x['raw_chain'][3] for x in type2_3chain]
            part5 = [x['raw_chain'][4] for x in type2_3chain]
            part6 = [x['raw_chain'][5] for x in type2_3chain]
            part7 = [x['raw_chain'][6] for x in type2_3chain]
            part8 = [x['raw_chain'][7] for x in type2_3chain]
            part9 = [x['raw_chain'][8] for x in type2_3chain]
            intact_part1 = part1.copy(); intact_part2 = part2.copy(); intact_part3 = part3.copy(); intact_part4 = part4.copy(); intact_part5 = part5.copy(); intact_part6 = part6.copy()
            intact_part7 = part7.copy(); intact_part8 = part8.copy(); intact_part9 = part9.copy()
            flattened_part1 = []; flattened_part2 = []; flattened_part3 = []; flattened_part4 = []; flattened_part5 = []; flattened_part6 = []
            flattened_part7 = []; flattened_part8 = []; flattened_part9 = []
            targets = []
            for chain_iter in range(len(part2)):
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part4.append([part4[chain_iter][0],part4[chain_iter][1],-(chain_iter+1234)])
                flattened_part5.append([part5[chain_iter][0],part5[chain_iter][1],-(chain_iter+1234)])
                flattened_part6.append([part6[chain_iter][0],part6[chain_iter][1],-(chain_iter+1234)])
                flattened_part7.append([part7[chain_iter][0],part7[chain_iter][1],-(chain_iter+1234)])
                flattened_part8.append([part8[chain_iter][0],part8[chain_iter][1],-(chain_iter+1234)])
                flattened_part9.append([part9[chain_iter][0],part9[chain_iter][1],-(chain_iter+1234)])
                targets.append(part2[chain_iter][2])
            part1 = flattened_part1; part2 = flattened_part2; part3 = flattened_part3; part4 = flattened_part4; part5 = flattened_part5; part6 = flattened_part6
            part7 = flattened_part7; part8 = flattened_part8; part9 = flattened_part9
            target_ids, keys = get_keys_and_targets([part1,part2, part3, part4, part5, part6, part7, part8, part9], targets, graph_type)
            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0], part4[0], part5[0], part6[0], part7[0], part8[0], part9[0]])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1); part1 = torch.tensor(part1.astype('int64'), device=device)
            part2 = np.array(part2); part2 = torch.tensor(part2.astype('int64'), device=device)
            part3 = np.array(part3); part3 = torch.tensor(part3.astype('int64'), device=device)
            part4 = np.array(part4); part4 = torch.tensor(part4.astype('int64'), device=device)
            part5 = np.array(part5); part5 = torch.tensor(part5.astype('int64'), device=device)
            part6 = np.array(part6); part6 = torch.tensor(part6.astype('int64'), device=device)
            part7 = np.array(part7); part7 = torch.tensor(part7.astype('int64'), device=device)
            part8 = np.array(part8); part8 = torch.tensor(part8.astype('int64'), device=device)
            part9 = np.array(part9); part9 = torch.tensor(part9.astype('int64'), device=device)
            chain1 = kbc.model.get_full_embeddigns(part1); chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3); chain4 = kbc.model.get_full_embeddigns(part4)
            chain5 = kbc.model.get_full_embeddigns(part5); chain6 = kbc.model.get_full_embeddigns(part6)
            chain7 = kbc.model.get_full_embeddigns(part7); chain8 = kbc.model.get_full_embeddigns(part8)
            chain9 = kbc.model.get_full_embeddigns(part9)
            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm += torch.norm(lhs_emb)
            lhs_norm /= len(chain1[0])
            chains = [chain1, chain2, chain3, chain4, chain5, chain6, chain7, chain8, chain9]
            parts = [part1, part2, part3, part4, part5, part6, part7, part8, part9]
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4, intact_part5, intact_part6, intact_part7, intact_part8, intact_part9]
            possible_heads_emb = []; possible_tails_emb = []; users=[]; items=[]
            
        elif QuerDAG.TYPE3_3_seq.value == graph_type:
            raw = dataset.type3_3chain
            type3_3chain = []
            for i in range(len(raw)):
                type3_3chain.append(raw[i].data)
            part1 = [x['raw_chain'][0] for x in type3_3chain]
            part2 = [x['raw_chain'][1] for x in type3_3chain]
            part3 = [x['raw_chain'][2] for x in type3_3chain]
            part4 = [x['raw_chain'][3] for x in type3_3chain]
            part5 = [x['raw_chain'][4] for x in type3_3chain]
            part6 = [x['raw_chain'][5] for x in type3_3chain]
            part7 = [x['raw_chain'][6] for x in type3_3chain]
            part8 = [x['raw_chain'][7] for x in type3_3chain]
            part9 = [x['raw_chain'][8] for x in type3_3chain]
            intact_part1 = part1.copy(); intact_part2 = part2.copy(); intact_part3 = part3.copy(); intact_part4 = part4.copy(); intact_part5 = part5.copy(); intact_part6 = part6.copy()
            intact_part7 = part7.copy(); intact_part8 = part8.copy(); intact_part9 = part9.copy()
            flattened_part1 = []; flattened_part2 = []; flattened_part3 = []; flattened_part4 = []; flattened_part5 = []; flattened_part6 = []
            flattened_part7 = []; flattened_part8 = []; flattened_part9 = []
            targets = []

            for chain_iter in range(len(part2)):
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],part1[chain_iter][2]])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part4.append([part4[chain_iter][0],part4[chain_iter][1],part4[chain_iter][2]])
                flattened_part5.append([part5[chain_iter][0],part5[chain_iter][1],-(chain_iter+1234)])
                flattened_part6.append([part6[chain_iter][0],part6[chain_iter][1],-(chain_iter+1234)])
                flattened_part7.append([part7[chain_iter][0],part7[chain_iter][1],part7[chain_iter][2]])
                flattened_part8.append([part8[chain_iter][0],part8[chain_iter][1],-(chain_iter+1234)])
                flattened_part9.append([part9[chain_iter][0],part9[chain_iter][1],-(chain_iter+1234)])
            
                targets.append(part2[chain_iter][2])
            part1 = flattened_part1; part2 = flattened_part2; part3 = flattened_part3; part4 = flattened_part4; part5 = flattened_part5; part6 = flattened_part6
            part7 = flattened_part7; part8 = flattened_part8; part9 = flattened_part9
            target_ids, keys = get_keys_and_targets([part1,part2, part3, part4, part5, part6, part7, part8, part9], targets, graph_type)
            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0], part4[0], part5[0], part6[0], part7[0], part8[0], part9[0]])
       
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1); part1 = torch.tensor(part1.astype('int64'), device=device)
            part2 = np.array(part2); part2 = torch.tensor(part2.astype('int64'), device=device)
            part3 = np.array(part3); part3 = torch.tensor(part3.astype('int64'), device=device)
            part4 = np.array(part4); part4 = torch.tensor(part4.astype('int64'), device=device)
            part5 = np.array(part5); part5 = torch.tensor(part5.astype('int64'), device=device)
            part6 = np.array(part6); part6 = torch.tensor(part6.astype('int64'), device=device)
            part7 = np.array(part7); part7 = torch.tensor(part7.astype('int64'), device=device)
            part8 = np.array(part8); part8 = torch.tensor(part8.astype('int64'), device=device)
            part9 = np.array(part9); part9 = torch.tensor(part9.astype('int64'), device=device)
            chain1 = kbc.model.get_full_embeddigns(part1); chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3); chain4 = kbc.model.get_full_embeddigns(part4)
            chain5 = kbc.model.get_full_embeddigns(part5); chain6 = kbc.model.get_full_embeddigns(part6)
            chain7 = kbc.model.get_full_embeddigns(part7); chain8 = kbc.model.get_full_embeddigns(part8)
            chain9 = kbc.model.get_full_embeddigns(part9)
            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm += torch.norm(lhs_emb)
            lhs_norm /= len(chain1[0])
            chains = [chain1, chain2, chain3, chain4, chain5, chain6, chain7, chain8, chain9]
            parts = [part1, part2, part3, part4, part5, part6, part7, part8, part9]
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4, intact_part5, intact_part6, intact_part7, intact_part8, intact_part9]
            possible_heads_emb = []; possible_tails_emb = []; users=[]; items=[]
        
        elif QuerDAG.TYPE4_3_seq.value == graph_type:
            raw = dataset.type4_3chain
            type4_3chain = []
            for i in range(len(raw)):
                type4_3chain.append(raw[i].data)
            part1 = [x['raw_chain'][0] for x in type4_3chain]
            part2 = [x['raw_chain'][1] for x in type4_3chain]
            part3 = [x['raw_chain'][2] for x in type4_3chain]
            part4 = [x['raw_chain'][3] for x in type4_3chain]
            part5 = [x['raw_chain'][4] for x in type4_3chain]
            part6 = [x['raw_chain'][5] for x in type4_3chain]
            part7 = [x['raw_chain'][6] for x in type4_3chain]
            part8 = [x['raw_chain'][7] for x in type4_3chain]
            part9 = [x['raw_chain'][8] for x in type4_3chain]
            intact_part1 = part1.copy(); intact_part2 = part2.copy(); intact_part3 = part3.copy(); intact_part4 = part4.copy(); intact_part5 = part5.copy(); intact_part6 = part6.copy()
            intact_part7 = part7.copy(); intact_part8 = part8.copy(); intact_part9 = part9.copy()
            flattened_part1 = []; flattened_part2 = []; flattened_part3 = []; flattened_part4 = []; flattened_part5 = []; flattened_part6 = []
            flattened_part7 = []; flattened_part8 = []; flattened_part9 = []
            targets = []
            for chain_iter in range(len(part2)):
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],part1[chain_iter][2]])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],part2[chain_iter][2]])
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part4.append([part4[chain_iter][0],part4[chain_iter][1],part4[chain_iter][2]])
                flattened_part5.append([part5[chain_iter][0],part5[chain_iter][1],part5[chain_iter][2]])
                flattened_part6.append([part6[chain_iter][0],part6[chain_iter][1],-(chain_iter+1234)])
                flattened_part7.append([part7[chain_iter][0],part7[chain_iter][1],part7[chain_iter][2]])
                flattened_part8.append([part8[chain_iter][0],part8[chain_iter][1],part8[chain_iter][2]])
                flattened_part9.append([part9[chain_iter][0],part9[chain_iter][1],-(chain_iter+1234)])
                targets.append(part3[chain_iter][2])
            part1 = flattened_part1; part2 = flattened_part2; part3 = flattened_part3; part4 = flattened_part4; part5 = flattened_part5; part6 = flattened_part6
            part7 = flattened_part7; part8 = flattened_part8; part9 = flattened_part9
            target_ids, keys = get_keys_and_targets([part1,part2, part3, part4, part5, part6, part7, part8, part9], targets, graph_type)
            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0], part4[0], part5[0], part6[0], part7[0], part8[0], part9[0]])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1); part1 = torch.tensor(part1.astype('int64'), device=device)
            part2 = np.array(part2); part2 = torch.tensor(part2.astype('int64'), device=device)
            part3 = np.array(part3); part3 = torch.tensor(part3.astype('int64'), device=device)
            part4 = np.array(part4); part4 = torch.tensor(part4.astype('int64'), device=device)
            part5 = np.array(part5); part5 = torch.tensor(part5.astype('int64'), device=device)
            part6 = np.array(part6); part6 = torch.tensor(part6.astype('int64'), device=device)
            part7 = np.array(part7); part7 = torch.tensor(part7.astype('int64'), device=device)
            part8 = np.array(part8); part8 = torch.tensor(part8.astype('int64'), device=device)
            part9 = np.array(part9); part9 = torch.tensor(part9.astype('int64'), device=device)
            chain1 = kbc.model.get_full_embeddigns(part1); chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3); chain4 = kbc.model.get_full_embeddigns(part4)
            chain5 = kbc.model.get_full_embeddigns(part5); chain6 = kbc.model.get_full_embeddigns(part6)
            chain7 = kbc.model.get_full_embeddigns(part7); chain8 = kbc.model.get_full_embeddigns(part8)
            chain9 = kbc.model.get_full_embeddigns(part9)
            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm += torch.norm(lhs_emb)
            lhs_norm /= len(chain1[0])
            chains = [chain1, chain2, chain3, chain4, chain5, chain6, chain7, chain8, chain9]
            parts = [part1, part2, part3, part4, part5, part6, part7, part8, part9]
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4, intact_part5, intact_part6, intact_part7, intact_part8, intact_part9]
            possible_heads_emb = []; possible_tails_emb = []; users=[]; items=[]
                
        elif QuerDAG.TYPE4_3_disj_seq.value == graph_type:
            raw = dataset.type4_3chain_u
            type4_3chain = []
            for i in range(len(raw)):
                type4_3chain.append(raw[i].data)
            part1 = [x['raw_chain'][0] for x in type4_3chain]
            part2 = [x['raw_chain'][1] for x in type4_3chain]
            part3 = [x['raw_chain'][2] for x in type4_3chain]
            part4 = [x['raw_chain'][3] for x in type4_3chain]
            part5 = [x['raw_chain'][4] for x in type4_3chain]
            part6 = [x['raw_chain'][5] for x in type4_3chain]
            part7 = [x['raw_chain'][6] for x in type4_3chain]
            part8 = [x['raw_chain'][7] for x in type4_3chain]
            part9 = [x['raw_chain'][8] for x in type4_3chain]
            intact_part1 = part1.copy(); intact_part2 = part2.copy(); intact_part3 = part3.copy(); intact_part4 = part4.copy(); intact_part5 = part5.copy(); intact_part6 = part6.copy()
            intact_part7 = part7.copy(); intact_part8 = part8.copy(); intact_part9 = part9.copy()
            flattened_part1 = []; flattened_part2 = []; flattened_part3 = []; flattened_part4 = []; flattened_part5 = []; flattened_part6 = []
            flattened_part7 = []; flattened_part8 = []; flattened_part9 = []
            targets = []
            for chain_iter in range(len(part2)):
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],part1[chain_iter][2]])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],part2[chain_iter][2]])
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part4.append([part4[chain_iter][0],part4[chain_iter][1],part4[chain_iter][2]])
                flattened_part5.append([part5[chain_iter][0],part5[chain_iter][1],part5[chain_iter][2]])
                flattened_part6.append([part6[chain_iter][0],part6[chain_iter][1],-(chain_iter+1234)])
                flattened_part7.append([part7[chain_iter][0],part7[chain_iter][1],part7[chain_iter][2]])
                flattened_part8.append([part8[chain_iter][0],part8[chain_iter][1],part8[chain_iter][2]])
                flattened_part9.append([part9[chain_iter][0],part9[chain_iter][1],-(chain_iter+1234)])
                targets.append(part3[chain_iter][2])
            part1 = flattened_part1; part2 = flattened_part2; part3 = flattened_part3; part4 = flattened_part4; part5 = flattened_part5; part6 = flattened_part6
            part7 = flattened_part7; part8 = flattened_part8; part9 = flattened_part9
            target_ids, keys = get_keys_and_targets([part1,part2, part3, part4, part5, part6, part7, part8, part9], targets, graph_type)
            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0], part4[0], part5[0], part6[0], part7[0], part8[0], part9[0]])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part1 = np.array(part1); part1 = torch.tensor(part1.astype('int64'), device=device)
            part2 = np.array(part2); part2 = torch.tensor(part2.astype('int64'), device=device)
            part3 = np.array(part3); part3 = torch.tensor(part3.astype('int64'), device=device)
            part4 = np.array(part4); part4 = torch.tensor(part4.astype('int64'), device=device)
            part5 = np.array(part5); part5 = torch.tensor(part5.astype('int64'), device=device)
            part6 = np.array(part6); part6 = torch.tensor(part6.astype('int64'), device=device)
            part7 = np.array(part7); part7 = torch.tensor(part7.astype('int64'), device=device)
            part8 = np.array(part8); part8 = torch.tensor(part8.astype('int64'), device=device)
            part9 = np.array(part9); part9 = torch.tensor(part9.astype('int64'), device=device)
            chain1 = kbc.model.get_full_embeddigns(part1); chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3); chain4 = kbc.model.get_full_embeddigns(part4)
            chain5 = kbc.model.get_full_embeddigns(part5); chain6 = kbc.model.get_full_embeddigns(part6)
            chain7 = kbc.model.get_full_embeddigns(part7); chain8 = kbc.model.get_full_embeddigns(part8)
            chain9 = kbc.model.get_full_embeddigns(part9)
            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm += torch.norm(lhs_emb)
            lhs_norm /= len(chain1[0])
            chains = [chain1, chain2, chain3, chain4, chain5, chain6, chain7, chain8, chain9]
            parts = [part1, part2, part3, part4, part5, part6, part7, part8, part9]
            intact_parts = [intact_part1, intact_part2, intact_part3, intact_part4, intact_part5, intact_part6, intact_part7, intact_part8, intact_part9]
            possible_heads_emb = []; possible_tails_emb = []; users=[]; items=[]
                

        else:
            chains = dataset['chains']
            parts = dataset['parts']
            target_ids = dataset['target_ids']
            chain_instructions = create_instructions([parts[0][0], parts[1][0], parts[2][0]])

        if mode == 'hard':
            if kg_path is not None and explain:
                ent_id2fb = pickle.load(open(osp.join(kg_path, 'ind2ent.pkl'), 'rb'))
                rel_id2fb = pickle.load(open(osp.join(kg_path, 'ind2rel.pkl'), 'rb'))
                fb2name = defaultdict(lambda: '[missing]')
                with open(osp.join(kg_path, 'entity2text.txt')) as f:
                    for line in f:
                        fb_id, name = line.strip().split('\t')
                        fb2name[fb_id] = name
            else:
                ent_id2fb, rel_id2fb, fb2name = None, None, None

            lhs_norm = 0.0
 
            env.set_attr(raw, kbc, chains, parts, intact_parts, target_ids, keys, None, None, chain_instructions, graph_type, lhs_norm, False, ent_id2fb, rel_id2fb, fb2name,
            possible_heads_emb, possible_tails_emb, users, items, ent_id)

            # env.set_attr(kbc,chains,parts,target_ids, keys, chain_instructions , graph_type, lhs_norm)
            # def set_attr(kbc, chains, parts, target_ids_hard, keys_hard, target_ids_complete, keys_complete, chain_instructions, graph_type, lhs_norm, cuda ):
        else:
            env.set_eval_complete(target_ids,keys)

    except RuntimeError as e:
        print("Cannot preload environment with error: ", e)
        return env

    return env
