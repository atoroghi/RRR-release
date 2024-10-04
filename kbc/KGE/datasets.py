# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import pickle
from typing import Dict, Tuple, List

import torch, sys
from kbc.models import KBCModel


class Dataset(object):
    def __init__(self, path):
        self.root = Path(path)

        self.data = {}

        #loads train, test, and valid txt files from data/datasetname/kbc_data
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.txt.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        with open(str(self.root / 'ent_id.pickle'), 'rb') as f:
            self.n_entities = len(pickle.load(f))
        with open(str(self.root / 'rel_id.pickle'), 'rb') as f:
            rel_id = pickle.load(f)
        self.n_predicates = len(rel_id)
        try:
            self.likes_rel = rel_id[(self.n_predicates - 1)]
        except:
            pass
        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        #self.to_skip keys are "lhs" and "rhs"
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

        try:
            head_file = open(str(self.root / ('valid_heads.pickle')), 'rb')
            self.valid_heads = pickle.load(head_file)
            head_file.close()
            tail_file = open(str(self.root / ('valid_tails.pickle')), 'rb')
            self.valid_tails = pickle.load(tail_file)
            tail_file.close()
        except:
            pass


    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        return self.data['train']

    def eval(self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',at: Tuple[int] = (1, 3, 10)):
        
        #gets the split of interest from the dataset
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        # whether we want to evaluate missing lhs or rhs or both
        missing = [missing_eval]
        #missing eval is 'both' by default (for cqd)
        #NOTE: difference between cqd and my simple
        # if missing_eval == 'both':
        #     missing = ['rhs', 'lhs']
        missing = ['rhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        # if we're dealing with the rhs missing, we keep the examples as they are, otherwise we swap the first and second column
        # next, for the lhs missing, we increase even relations by 1 and odd relations by -1
        for m in missing:
            q = examples.clone()

            # this happens only for train split
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            # this was for the case that to_skip only contained direct queries
            # now, both the datasets and the to_skips have both directions, i.e.,
            # for a triple [10,32,179], we also have [179,33,10] in the dataset
            # and for 'rhs' and 'lhs', we have both (10,32):179 and (179,33):10
            #so we can basically just consider the 'rhs' missing and get rid of this
            # if m == 'lhs':
            #     # swap the first and second column
            #     tmp = torch.clone(q[:, 0])
            #     q[:, 0] = q[:, 2]
            #     q[:, 2] = tmp
            #NOTE
            # Comment these lines if rev relations do not have their numbers (my Simple)
            # OW, uncomment them
                # # Note: in q2b relations are labeled as
                # # [rel1, rel1inv, rel2, rel2inv, ...]
                # # In contrast, KBC uses
                # # [rel1, rel2, ..., rel1inv, rel2inv, ...]
                # # That's the reason for this:
                # rels = q[:, 1].clone()
                # q[:, 1][rels % 2 == 0] += 1
                # q[:, 1][rels % 2 != 0] -= 1
                # # Instead of:
                # # q[:, 1] += self.n_predicates // 2

            # get the ranks of the correct answers (while skipping the to_skip) and compute the mean reciprocal rank and hits@k

            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500, side= m)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(lambda x: torch.mean((ranks <= x).float()).item(),at))))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities

# returns the specificed split of the dataset. just if the split is train, it returns a permuted subset of the data
    def dataset_to_queries(self,split: str):
        try:
            # getting the data for that split from the dataset
            test = self.get_examples(split)
            examples = torch.from_numpy(test.astype('int64')).cuda()
            missing = ['rhs']

            for m in missing:
                q = examples.clone()
                if 'train' in split.lower():
                    permutation = torch.randperm(len(examples))[:5000]
                    q = examples[permutation]
                # if m == 'lhs':
                #     tmp = torch.clone(q[:, 0])
                #     q[:, 0] = q[:, 2]
                #     q[:, 2] = tmp
                #     q[:, 1] += self.n_predicates // 2

        except Exception as e:
            print("Unable to segment queries from dataset with error {}".format(str(e)))
            return None

        return q
