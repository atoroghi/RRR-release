# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Callable
import math
import sys, os
import logging
import pickle
import copy
import torch
from torch import nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from kbc.regularizers import Regularizer
import tqdm

import traceback

from kbc.utils import QuerDAG
from kbc.utils import DynKBCSingleton
from kbc.utils import make_batches
from kbc.utils import Device


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def get_queries_separated(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def score_emb(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor):
        pass

    @abstractmethod
    def candidates_score(self, rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor], *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        pass

    @abstractmethod
    def model_type(self):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1, side: str = 'k'
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                # rhs is the embedding of all entities (not just tail part)
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]

                    # side: rhs
                    # q = return torch.cat([0.5 * lhs[1] * rel[1],0.5 * lhs[0] * rel[0]], 1)
                    # side: lhs
                    # q = return torch.cat([0.5 * lhs[1] * rel[0], 0.5 * lhs[0] * rel[1]], 1)
                    q = self.get_queries(these_queries, side)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        try:
                            filter_out = filters[(
                                query[0].item(), query[1].item())]
                        except:
                            print(query)
                            print(side)
                            sys.exit()
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out if c_begin <= x < c_begin + chunk_size]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6

                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1).cpu()

                    b_begin += batch_size

                c_begin += chunk_size

        return ranks

    @staticmethod
    def __get_chains__(chains: List, graph_type: str = QuerDAG.TYPE1_2.value):
        if graph_type == QuerDAG.TYPE1_1.value:
            chain1 = chains[0]
        elif '2' in graph_type[-1]:
            chain1, chain2 = chains
        elif '3' in graph_type[-1]:
            chain1, chain2, chain3 = chains

        if QuerDAG.TYPE1_1.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            raw_chain = [lhs_1, rel_1]

        elif QuerDAG.TYPE1_2.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            rel_2 = chain2[1]

            raw_chain = [lhs_1, rel_1, rel_2]

        elif QuerDAG.TYPE2_2.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            lhs_2 = chain2[0]
            rel_2 = chain2[1]

            raw_chain = [lhs_1, rel_1, lhs_2, rel_2]

        elif QuerDAG.TYPE1_3.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            rel_2 = chain2[1]

            rhs_3 = chain3[1]

            raw_chain = [lhs_1, rel_1, rel_2, rhs_3]

        elif QuerDAG.TYPE2_3.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            lhs_2 = chain2[0]
            rel_2 = chain2[1]

            lhs_3 = chain3[0]
            rel_3 = chain3[1]

            raw_chain = [lhs_1, rel_1, lhs_2, rel_2, lhs_3, rel_3]

        elif QuerDAG.TYPE3_3.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            rel_2 = chain2[1]

            lhs_2 = chain3[0]
            rel_3 = chain3[1]

            raw_chain = [lhs_1, rel_1, rel_2, lhs_2, rel_3]

        elif QuerDAG.TYPE4_3.value in graph_type:
            lhs_1 = chain1[0]
            rel_1 = chain1[1]

            lhs_2 = chain2[0]
            rel_2 = chain2[1]

            rel_3 = chain3[1]

            raw_chain = [lhs_1, rel_1, lhs_2, rel_2, rel_3]

        return raw_chain

    @staticmethod
    def _optimize_variables(scoring_fn: Callable, params: list, optimizer: str,
                            lr: float, max_steps: int):
        if optimizer == 'adam':
            optimizer = optim.Adam(params, lr=lr)
        elif optimizer == 'adagrad':
            optimizer = optim.Adagrad(params, lr=lr)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(params, lr=lr)
        else:
            raise ValueError(f'Unknown optimizer {optimizer}')

        prev_loss_value = 1000
        loss_value = 999
        losses = []

        with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:
            i = 0
            while i < max_steps and math.fabs(prev_loss_value - loss_value) > 1e-9:
                prev_loss_value = loss_value

                norm, regularizer, _ = scoring_fn()
                loss = -norm.mean() + regularizer

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i += 1
                bar.update(1)
                bar.set_postfix(loss=f'{loss.item():.6f}')

                loss_value = loss.item()
                losses.append(loss_value)

            if i != max_steps:
                bar.update(max_steps - i + 1)
                bar.close()
                print("Search converged early after {} iterations".format(i))

        with torch.no_grad():
            *_, scores = scoring_fn(score_all=True)

        return scores

    @staticmethod
    def batch_t_norm(atoms: Tensor, norm_type: str = 'min') -> Tensor:
        if norm_type == 'min':
            scores = torch.min(atoms, dim=-1)[0]
        elif norm_type == 'prod':
            scores = torch.prod(atoms, dim=-1)
        else:
            raise ValueError(
                f't_norm must be "min" or "prod", got {norm_type}')

        return scores

    @staticmethod
    def batch_t_conorm(atoms: Tensor, norm_type: str = 'max') -> Tensor:
        if norm_type == 'min':
            scores = torch.max(atoms, dim=-1)[0]
        elif norm_type == 'prod':
            scores = torch.sum(atoms, dim=-1) - torch.prod(atoms, dim=-1)
        else:
            raise ValueError(
                f't_conorm must be "min" or "prod", got {norm_type}')

        return scores

    def link_prediction(self, chains: List):
        lhs_1, rel_1 = self.__get_chains__(
            chains, graph_type=QuerDAG.TYPE1_1.value)
        # self.forward_emb returns the score of each node to be the rhs of the given lhs and rel
        return self.forward_emb(lhs_1, rel_1)




    def get_best_candidates(self,
                            rel: Tensor,
                            arg1: Optional[Tensor],
                            arg2: Optional[Tensor],
                            candidates: int = 5,
                            last_step=False, env: DynKBCSingleton = None, side: str= 'lhs') -> Tuple[Tensor, Tensor]:

        z_scores, z_emb, z_indices = None, None, None

        assert (arg1 is None) ^ (arg2 is None)

        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        # [B, N]
        # scores_sp = (s, p, ?)
        # scores_sp, scores_po = self.candidates_score(rel, arg1, arg2)
        # scores = scores_sp if arg2 is None else scores_po
        if side == 'lhs':
            scores = self.forward_emb(arg1, rel)
        elif side == 'rhs':
            scores = self.backward_emb(arg1, rel)

        if not last_step:
            # [B, K], [B, K]
            k = min(candidates, scores.shape[1])
            z_scores, z_indices = torch.topk(scores, k=k, dim=1)
            # [B, K, E]
            z_emb = self.entity_embeddings(z_indices)
            assert z_emb.shape[0] == batch_size
            assert z_emb.shape[2] == embedding_size
        else:
            z_scores = scores

            z_indices = torch.arange(z_scores.shape[1]).view(
                1, -1).repeat(z_scores.shape[0], 1).to(Device)
            z_emb = self.entity_embeddings(z_indices)

        return z_scores, z_emb

    def t_norm(self, tens_1: Tensor, tens_2: Tensor, t_norm: str = 'min') -> Tensor:
        if 'min' in t_norm:
            return torch.min(tens_1, tens_2)
        elif 'prod' in t_norm:
            return tens_1 * tens_2
        elif 'luk' in t_norm:
            return torch.max(tens_1 + tens_2 - 1, torch.zeros_like(tens_1))

    def t_conorm(self, tens_1: Tensor, tens_2: Tensor, t_conorm: str = 'max') -> Tensor:
        if 'min' in t_conorm:
            return torch.max(tens_1, tens_2)
        elif 'prod' in t_conorm:
            return (tens_1+tens_2) - (tens_1 * tens_2)
        elif 'luk' in t_conorm:
            return torch.min(tens_1 + tens_2, torch.ones_like(tens_1))

    def min_max_rescale(self, x):
        return (x-torch.min(x))/(torch.max(x) - torch.min(x))

    def query_answering_BF_Marginal_UI(self, env: DynKBCSingleton, candidates: int = 5, t_norm: str = 'min', 
    batch_size=1, scores_normalize=0, explain='no', cov_anchor=1e-2, cov_var=1e-2, cov_target=1e-2):

        if 'disj' in env.graph_type:
            objective = self.t_conorm
        else: 
            objective = self.t_norm
        parts = env.parts
        chains, chain_instructions = env.chains, env.chain_instructions
        intact_parts = env.intact_parts
        possible_heads_emb = env.possible_heads_emb; possible_tails_emb = env.possible_tails_emb
        nb_queries, emb_dim = chains[0][0].shape[0], chains[0][0].shape[1]
        user_embs = torch.empty((nb_queries, emb_dim), device=Device)
        # scores_m = self.backward_emb(rhs_2.unsqueeze(dim=0),rel_2.unsqueeze(dim=0))
        # rcount = (scores_m[0] > scores_m[0][gt]).sum().item() + 1
        # z_scores, z_indices = torch.topk(scores_m, k=10, dim=1)

        if env.graph_type == '1_2':
            part1 , part2 = parts[0], parts[1]
            intact_part1, intact_part2 = intact_parts[0], intact_parts[1]

            chain1, chain2 = chains[0], chains[1]

            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb = chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2]
            if not 'SimplE' in str(self.model_type):
                raise NotImplementedError
            else:
                for i in tqdm.tqdm(range(nb_queries // 5)):

                    for j in range(5):
                        # lhs_1 is the user belief. rhs_2 is the evidence embedding
                        lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                        lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], rhs_2_emb[i*5+j]
                        gt = torch.tensor((intact_part1[i*5+j][2].astype(np.int32)))
                        ### sanity check
                        #print("gt:", intact_part1[i*5+j][2].astype(np.int32))
                        #mu_gt = self.entity_embeddings(torch.tensor((intact_part1[i*5+j][2].astype(np.int32))))
                        #print("mu_gt:", mu_gt)
                        #h_gt_for = (1/cov_anchor) * mu_gt[:emb_dim//2] * rel_1[emb_dim//2:]
                        #h_gt_inv = (1/cov_anchor) * mu_gt[emb_dim//2:] * rel_1[:emb_dim//2]
                        #print("h_gt_inv:", h_gt_inv)
                        ##print(h_gt_for)
                        ##sys.exit()
                        #if j == 0:
                        #    mu_u_for, mu_u_inv = lhs_1[:emb_dim//2], lhs_1[emb_dim//2:]
                        #    print("mu_u_for before:", mu_u_for)
                        #    h_u_for, h_u_inv = (1/cov_target) * mu_u_for, (1/cov_target) * mu_u_inv
                        #    J_u_for, J_u_inv = (1/cov_target), (1/cov_target)
                        #print("h_u_for before:", h_u_for)
                        #h_u_for = h_u_for + h_gt_inv
                        #print("h_u_for after:",h_u_for)
                        #h_u_inv = h_u_inv + h_gt_for
                        #J_u_for = J_u_for + (1/cov_anchor)
                        #J_u_inv = J_u_inv + (1/cov_anchor)
                        #mu_u_for, mu_u_inv = h_u_for / J_u_for, h_u_inv / J_u_inv
                        #print("mu_u_for after:", mu_u_for)
                        ##print(lhs_1)
                        ##print(h_gt_inv)
                        ##print(mu_u_for)
                        ##sys.exit()
                        #user_embs[i*5+j, :emb_dim//2] = mu_u_for
                        #user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                        #user_embs_test = torch.cat([mu_u_for, mu_u_inv], dim=0).unsqueeze(dim=0)
                        #scores_test = self.forward_emb(user_embs_test, rel_1.unsqueeze(dim=0))
                        ##print((scores_test[0][11080]))
                        #print(scores_test[0][1])
                        ## count number of elements in scores_test that are greater than scores_test[1]
                        #rcount = (scores_test[0] > scores_test[0][1]).sum().item() + 1
                        #print(rcount)
                        #sys.exit()

                        #print(lhs_1)
                        mu_m_for = possible_tails_emb[0][i*5+j, :emb_dim//2] 
                        #mu_m_for = torch.zeros_like(mu_m_for)
                        h_m_for = (1/cov_var) * mu_m_for
                        mu_m_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                        #mu_m_inv = torch.zeros_like(mu_m_inv)
                        h_m_inv = (1/cov_var) * mu_m_inv
                        mu_d_for = rhs_2[:emb_dim//2] * rel_2[emb_dim//2:]
                        h_d_for = (1/cov_anchor) * mu_d_for
                        mu_d_inv = rhs_2[emb_dim//2:] * rel_2[:emb_dim//2]
                        h_d_inv = (1/cov_anchor) * mu_d_inv
                        h_m_for = h_m_for + h_d_inv
                        h_m_inv = h_m_inv + h_d_for
                        ### #print(h_m_for)   
                        J_m_for = (1/cov_anchor) + (1/cov_var)
                        ## # mu_m will be useful if you want to do explanation
                        mu_m_for = h_m_for / J_m_for
                        J_m_inv = (1/cov_anchor) + (1/cov_var)
                        mu_m_inv = h_m_inv / J_m_inv

                        # scores_m = self.backward_emb(rhs_2.unsqueeze(dim=0),rel_2.unsqueeze(dim=0))
                        # print(torch.argmax(scores_m))
                        # rcount = (scores_m[0] > scores_m[0][gt]).sum().item() + 1
                        # print(rcount)
                        # z_scores, z_indices = torch.topk(scores_m, k=10, dim=1)
                        # print(z_indices)
                        # sys.exit()
                        ### update the precision and information of the target node given the variable
                        if j == 0:
                           #mu_u = torch.unsqueeze(lhs_1, dim=0)
                           mu_u = lhs_1
                           mu_u_for = mu_u[:emb_dim//2]
                           mu_u_inv = mu_u[emb_dim//2:]
                           h_u_for = (1/cov_target) * mu_u_for
                           h_u_inv = (1/cov_target) * mu_u_inv
                           J_u_for = (1/cov_target)
                           J_u_inv = (1/cov_target)
                           #user_embs_test = torch.cat([mu_u_for, mu_u_inv], dim=0).unsqueeze(dim=0)
                           #scores_test_pre = self.forward_emb(user_embs_test, rel_1.unsqueeze(dim=0))
                           #rcount_pre = (scores_test_pre[0] > scores_test_pre[0][1]).sum().item() + 1
                           #print("rcount_pre:", rcount_pre)
                        #print(h_u_for)

                        # marginal

                        h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * h_m_inv
                        J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * rel_1[:emb_dim//2]
                        h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * h_m_for
                        J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * rel_1[emb_dim//2:]
                        # conditional
                        # h_u_for = h_u_for + h_m_inv * rel_1[:emb_dim//2]
                        # h_u_inv = h_u_inv + h_m_for * rel_1[emb_dim//2:]
                        # J_u_for = J_u_for + J_m_inv
                        # J_u_inv = J_u_inv + J_m_for
                        
                        mu_u_for = h_u_for / J_u_for
                        mu_u_inv = h_u_inv / J_u_inv
                     
                        user_embs[i*5+j, :emb_dim//2] = mu_u_for
                        user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                        #user_embs_test = torch.cat([mu_u_for, mu_u_inv], dim=0).unsqueeze(dim=0)
                        #scores_test = self.forward_emb(user_embs_test, rel_1.unsqueeze(dim=0))
                        #print(scores_test[0][1])
                        #print(torch.max(scores_test[0]))
                        #rcount = (scores_test[0] > scores_test[0][1]).sum().item() + 1
                        #print(rcount)
                        #sys.exit()

                        # from here on, like user belief updating without beam search
                        # for instantiated, we have to first find the top items as existential, then update user belief

                # once we have the user embeddings for each query, we can calculate the recommendation scores
                scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))

        elif env.graph_type == '1_3':
            part1 , part2, part3 = parts[0], parts[1], parts[2]
            #print(part1[0])
            
            intact_part1, intact_part2, intact_part3 = intact_parts[0], intact_parts[1], intact_parts[2]
            #print(intact_part1[0])
            chain1, chain2, chain3 = chains[0], chains[1], chains[2]
            #print(chain1[2])

            #print(self.entity_embeddings(torch.tensor(part1[0][0])))
            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb, lhs_3_emb, rel_3_emb, rhs_3_emb = chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2], chain3[0], chain3[1], chain3[2]
            
            
            #anchor_emb = (self.entity_embeddings(torch.tensor(intact_part3[0][2])))

            #gt_emb = (self.entity_embeddings(torch.tensor(intact_part1[0][2].astype(int))))

            if not 'SimplE' in str(self.model_type):
                raise NotImplementedError

            else:
                for i in tqdm.tqdm(range(nb_queries// 5)):
                    for j in range(5):
                        lhs_3, rel_3, rhs_3 = None, rel_3_emb[i*5+j], rhs_3_emb[i*5+j]
                        lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], None
                        lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                        if j == 0:
                            mu_u = lhs_1
                            mu_u_for = mu_u[:emb_dim//2]
                            mu_u_inv = mu_u[emb_dim//2:]
                            h_u_for = (1/cov_target) * mu_u_for
                            h_u_inv = (1/cov_target) * mu_u_inv
                            J_u_for = 1/cov_target
                            J_u_inv = 1/cov_target
                        top_vars2_scores, top_vars2_embs = self.get_best_candidates(rel_3.view(1,-1), rhs_3.view(1,-1), None, candidates=candidates, last_step=False, side='rhs')
                        top_vars2_mean = torch.mean(top_vars2_embs.squeeze(dim=0), dim=0).view(emb_dim)
                        mu_d_for = top_vars2_mean[:emb_dim//2]* rel_2[emb_dim//2:]; mu_d_inv = top_vars2_mean[emb_dim//2:]* rel_2[:emb_dim//2]
                        h_d_for = (1/cov_anchor) * mu_d_for; h_d_inv = (1/cov_anchor) * mu_d_inv
                        mu_m_for = possible_tails_emb[0][i*5+j, :emb_dim//2] 
                        mu_m_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                        h_m_for = (1/cov_var) * mu_m_for; h_m_inv = (1/cov_var) * mu_m_inv
                        J_m_for = 1/cov_var + 1/cov_anchor; J_m_inv = 1/cov_var + 1/cov_anchor
                        h_m_for = h_m_for + h_d_inv; h_m_inv = h_m_inv + h_d_for

                        h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * h_m_inv
                        J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * rel_1[:emb_dim//2]
                        h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * h_m_for
                        J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * rel_1[emb_dim//2:]
                        mu_u_for = h_u_for / J_u_for
                        mu_u_inv = h_u_inv / J_u_inv
                        user_embs[i*5+j, :emb_dim//2] = mu_u_for
                        user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))

            # else:
            #     for i in tqdm.tqdm(range(nb_queries // 5)):
            #         for j in range(5):
            #             lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
            #             lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], None
            #             lhs_3, rel_3, rhs_3 = None, rel_3_emb[i*5+j], rhs_3_emb[i*5+j]
            #             # here
            #             mu_m2_for = possible_tails_emb[1][i*5+j, :emb_dim//2]
            #             h_m2_for = (1/cov_var) * mu_m2_for

            #             mu_m2_inv = possible_tails_emb[1][i*5+j, emb_dim//2:]
            #             h_m2_inv = (1/cov_var) * mu_m2_inv
            #             mu_d_for = rhs_3[:emb_dim//2] * rel_3[emb_dim//2:]
            #             h_d_for = (1/cov_anchor) * mu_d_for
            #             mu_d_inv = rhs_3[emb_dim//2:] * rel_3[:emb_dim//2]
            #             h_d_inv = (1/cov_anchor) * mu_d_inv
            #             h_m2_for = h_m2_for + h_d_inv; h_m2_inv = h_m2_inv + h_d_for
            #             J_m2_for = (1/cov_anchor) + (1/cov_var); J_m2_inv = (1/cov_anchor) + (1/cov_var)

            # here we should multiply by rel_2[emb_dim//2:] and rel_2[:emb_dim//2] respectively
            #             mu_m2_for = h_m2_for / J_m2_for; mu_m2_inv = h_m2_inv / J_m2_inv

            #             mu_m1_for = possible_tails_emb[0][i*5+j, :emb_dim//2]
            #             h_m1_for = (1/cov_var) * mu_m1_for
                        
            #             mu_m1_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
            #             h_m1_inv = (1/cov_var) * mu_m1_inv
            #             # marginal
            #             # h_m1_for = h_m1_for + rel_2[:emb_dim//2] * (1 / J_m2_inv) * h_m2_inv
            #             # J_m1_for = (1/cov_var) + rel_2[:emb_dim//2] * (1 / J_m2_inv) * rel_2[:emb_dim//2]
            #             # h_m1_inv = h_m1_inv + rel_2[emb_dim//2:] * (1 / J_m2_for) * h_m2_for
            #             # J_m1_inv = (1/cov_var) + rel_2[emb_dim//2:] * (1 / J_m2_for) * rel_2[emb_dim//2:]
            #             # conditional
            #             h_m1_for = h_m1_for + h_m2_inv
            #             h_m1_inv = h_m1_inv + h_m2_for
            #             J_m1_for =  (1/cov_var) + J_m2_inv
            #             J_m1_inv =  (1/cov_var) + J_m2_for

            #             mu_m1_for = h_m1_for / J_m1_for; mu_m1_inv = h_m1_inv / J_m1_inv

            #             if j == 0:
            #                 mu_u = lhs_1
            #                 mu_u_for = mu_u[:emb_dim//2]
            #                 mu_u_inv = mu_u[emb_dim//2:]
            #                 h_u_for = (1/cov_target) * mu_u_for
            #                 h_u_inv = (1/cov_target) * mu_u_inv
            #                 J_u_for = (1/cov_target)
            #                 J_u_inv = (1/cov_target)
            #             h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m1_inv) * h_m1_inv
            #             J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m1_inv) * rel_1[:emb_dim//2]
            #             h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m1_for) * h_m1_for
            #             J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m1_for) * rel_1[emb_dim//2:]


            #             # conditional
            #             # h_u_for = h_u_for + h_m1_inv
            #             # h_u_inv = h_u_inv + h_m1_for

            #             mu_u_for = h_u_for / J_u_for
            #             mu_u_inv = h_u_inv / J_u_inv

            #             user_embs[i*5+j, :emb_dim//2] = mu_u_for
            #             user_embs[i*5+j, emb_dim//2:] = mu_u_inv
            #     scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))
        elif env.graph_type == '1_4':
            part1 , part2, part3, part4 = parts[0], parts[1], parts[2], parts[3]
            intact_part1, intact_part2, intact_part3, intact_part4 = intact_parts[0], intact_parts[1], intact_parts[2], intact_parts[3]
            chain1, chain2, chain3, chain4 = chains[0], chains[1], chains[2], chains[3]
            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb, lhs_3_emb, rel_3_emb, rhs_3_emb, lhs_4_emb, rel_4_emb, rhs_4_emb = chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2], chain3[0], chain3[1], chain3[2], chain4[0], chain4[1], chain4[2]
            new_env = copy.deepcopy(env)
            new_env.chain_instructions = ['hop_0_1']

            if not 'SimplE' in str(self.model_type):
                raise NotImplementedError
            else:
                for i in tqdm.tqdm(range(nb_queries// 5)):
                    for j in range(5):
                        lhs_4, rel_4, rhs_4 = None, rel_4_emb[i*5+j], rhs_4_emb[i*5+j]
                        lhs_3, rel_3, rhs_4 = None, rel_3_emb[i*5+j], None
                        lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], None
                        lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                        if j == 0:
                            mu_u = lhs_1
                            mu_u_for = mu_u[:emb_dim//2]
                            mu_u_inv = mu_u[emb_dim//2:]
                            h_u_for = (1/cov_target) * mu_u_for
                            h_u_inv = (1/cov_target) * mu_u_inv
                            J_u_for = (1/cov_target)
                            J_u_inv = (1/cov_target)
                        new_env.chains = [(chain1[0],chain1[1], chain1[2]),(chain3[0], chain3[1][i].view(1,-1), chain3[2]), (chain4[0],chain4[1][i].view(1,-1),chain4[2][i].view(1,-1))]
                        top_vars2_scores = self.query_answering_BF_Exist(new_env, candidates=candidates, t_norm=t_norm, batch_size=batch_size, scores_normalize=scores_normalize, explain=explain)
                        _, top_vars2_indices = torch.topk(top_vars2_scores, candidates, dim=1)
                        top_vars2_embs = self.entity_embeddings(top_vars2_indices)
                        top_vars2_mean = torch.mean(top_vars2_embs.squeeze(dim=0), dim=0).view(emb_dim)
                        mu_d_for = top_vars2_mean[:emb_dim//2]* rel_2[emb_dim//2:]; mu_d_inv = top_vars2_mean[emb_dim//2:]* rel_2[:emb_dim//2]
                        h_d_for = (1/cov_anchor) * mu_d_for; h_d_inv = (1/cov_anchor) * mu_d_inv
                        mu_m_for = possible_tails_emb[0][i*5+j, :emb_dim//2] 
                        mu_m_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                        h_m_for = (1/cov_var) * mu_m_for; h_m_inv = (1/cov_var) * mu_m_inv
                        J_m_for = 1/cov_var + 1/cov_anchor; J_m_inv = 1/cov_var + 1/cov_anchor
                        h_m_for = h_m_for + h_d_inv; h_m_inv = h_m_inv + h_d_for

                        h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * h_m_inv
                        J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * rel_1[:emb_dim//2]
                        h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * h_m_for
                        J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * rel_1[emb_dim//2:]
                        mu_u_for = h_u_for / J_u_for
                        mu_u_inv = h_u_inv / J_u_inv
                        user_embs[i*5+j, :emb_dim//2] = mu_u_for
                        user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))

                # cond_marginal
                # for i in tqdm.tqdm(range(nb_queries // 5)):
                #     for j in range(5):
                #         lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                #         lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], None
                #         lhs_3, rel_3, rhs_3 = None, rel_3_emb[i*5+j], None
                #         lhs_4, rel_4, rhs_4 = None, rel_4_emb[i*5+j], rhs_4_emb[i*5+j]
                #         mu_m3_for = possible_tails_emb[2][i*5+j, :emb_dim//2]
                #         h_m3_for = (1/cov_var) * mu_m3_for
                #         mu_m3_inv = possible_tails_emb[2][i*5+j, emb_dim//2:]
                #         h_m3_inv = (1/cov_var) * mu_m3_inv
                #         mu_d_for = rhs_4[:emb_dim//2] * rel_4[emb_dim//2:]
                #         h_d_for = (1/cov_anchor) * mu_d_for
                #         mu_d_inv = rhs_4[emb_dim//2:] * rel_4[:emb_dim//2]
                #         h_d_inv = (1/cov_anchor) * mu_d_inv
                #         h_m3_for = h_m3_for + h_d_inv; h_m3_inv = h_m3_inv + h_d_for
                #         J_m3_for = (1/cov_anchor) + (1/cov_var); J_m3_inv = (1/cov_anchor) + (1/cov_var)
                #         mu_m3_for = h_m3_for / J_m3_for; mu_m3_inv = h_m3_inv / J_m3_inv

                #         mu_m2_for = possible_tails_emb[1][i*5+j, :emb_dim//2]
                #         h_m2_for = (1/cov_var) * mu_m2_for
                #         mu_m2_inv = possible_tails_emb[1][i*5+j, emb_dim//2:]
                #         h_m2_inv = (1/cov_var) * mu_m2_inv

                #         #Marginal

                #         # h_m2_for = h_m2_for + rel_3[:emb_dim//2] * (1 / J_m3_inv) * h_m3_inv
                #         # J_m2_for = (1/cov_var) + rel_3[:emb_dim//2] * (1 / J_m3_inv) * rel_3[:emb_dim//2]
                #         # h_m2_inv = h_m2_inv + rel_3[emb_dim//2:] * (1 / J_m3_for) * h_m3_for
                #         # J_m2_inv = (1/cov_var) + rel_3[emb_dim//2:] * (1 / J_m3_for) * rel_3[emb_dim//2:]
                #         # mu_m2_for = h_m2_for / J_m2_for; mu_m2_inv = h_m2_inv / J_m2_inv

                #         #Conditional

                #         h_m2_for = h_m2_for + h_m3_inv; h_m2_inv = h_m2_inv + h_m3_for
                #         J_m2_for = (1/cov_var) + J_m3_inv; J_m2_inv = (1/cov_var) + J_m3_for
                #         mu_m2_for = h_m2_for / J_m2_for; mu_m2_inv = h_m2_inv / J_m2_inv

                #         mu_m1_for = possible_tails_emb[0][i*5+j, :emb_dim//2]
                #         h_m1_for = (1/cov_var) * mu_m1_for
                #         mu_m1_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                #         h_m1_inv = (1/cov_var) * mu_m1_inv
                #         # Marginal
                #         h_m1_for = h_m1_for + rel_2[:emb_dim//2] * (1 / J_m2_inv) * h_m2_inv
                #         J_m1_for = (1/cov_var) + rel_2[:emb_dim//2] * (1 / J_m2_inv) * rel_2[:emb_dim//2]
                #         h_m1_inv = h_m1_inv + rel_2[emb_dim//2:] * (1 / J_m2_for) * h_m2_for
                #         J_m1_inv = (1/cov_var) + rel_2[emb_dim//2:] * (1 / J_m2_for) * rel_2[emb_dim//2:]
                #         mu_m1_for = h_m1_for / J_m1_for; mu_m1_inv = h_m1_inv / J_m1_inv

                #         #Conditional

                #         # h_m1_for = h_m1_for + h_m2_inv; h_m1_inv = h_m1_inv + h_m2_for
                #         # J_m1_for = (1/cov_var) + J_m2_inv; J_m1_inv = (1/cov_var) + J_m2_for
                #         # mu_m1_for = h_m1_for / J_m1_for; mu_m1_inv = h_m1_inv / J_m1_inv

                #         if j == 0:
                #             mu_u = lhs_1
                #             mu_u_for = mu_u[:emb_dim//2]
                #             mu_u_inv = mu_u[emb_dim//2:]
                #             h_u_for = (1/cov_target) * mu_u_for
                #             h_u_inv = (1/cov_target) * mu_u_inv
                #             J_u_for = (1/cov_target)
                #             J_u_inv = (1/cov_target)

                #         #marginal
                        
                #         h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m1_inv) * h_m1_inv
                #         J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m1_inv) * rel_1[:emb_dim//2]
                #         h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m1_for) * h_m1_for
                #         J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m1_for) * rel_1[emb_dim//2:]

                #         #Conditional

                #         # h_u_for = h_u_for + h_m1_inv; h_u_inv = h_u_inv + h_m1_for
                #         # J_u_for = J_u_for + J_m1_inv; J_u_inv = J_u_inv + J_m1_for
                #         mu_u_for = h_u_for / J_u_for
                #         mu_u_inv = h_u_inv / J_u_inv
                #         user_embs[i*5+j, :emb_dim//2] = mu_u_for
                #         user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                # scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))



        elif env.graph_type == '2_2':
            part1, part2, part3 = parts[0], parts[1], parts[2]
            intact_part1, intact_part2, intact_part3 = intact_parts[0], intact_parts[1], intact_parts[2]
            chain1, chain2, chain3 = chains[0], chains[1], chains[2]
            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb, lhs_3_emb, rel_3_emb, rhs_3_emb = chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2], chain3[0], chain3[1], chain3[2]

            if not 'SimplE' in str(self.model_type):
                raise NotImplementedError
            else:
                for i in tqdm.tqdm(range(nb_queries // 5)):
                    for j in range(5):
                        # lhs_1 is the user belief. rhs_2 and rhs_3 are the evidence embeddings
                        lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                        lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], rhs_2_emb[i*5+j]
                        lhs_3, rel_3, rhs_3 = None, rel_3_emb[i*5+j], rhs_3_emb[i*5+j]
                        mu_gt = self.entity_embeddings(torch.tensor((intact_part1[i*5+j][2].astype(np.int32))))
                        mu_gt_for = mu_gt[:emb_dim//2]
                        mu_gt_inv = mu_gt[emb_dim//2:]
                        mu_m_for = possible_tails_emb[0][i*5+j, :emb_dim//2]
                        h_m_for = (1/cov_var) * mu_m_for
                        mu_m_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                        h_m_inv = (1/cov_var) * mu_m_inv
                        # print("dist between mu_gt and mu_m")
                        # print(torch.cdist(mu_gt_for.unsqueeze(dim=0), mu_m_for.unsqueeze(dim=0), p=2))
                        # print(torch.cdist(mu_gt_inv.unsqueeze(dim=0), mu_m_inv.unsqueeze(dim=0), p=2))
                        
                        mu_d_for1 = rhs_2[:emb_dim//2] * rel_2[emb_dim//2:]
                        h_d_for1 = (1/cov_anchor) * mu_d_for1
                        mu_d_inv1 = rhs_2[emb_dim//2:] * rel_2[:emb_dim//2]
                        h_d_inv1 = (1/cov_anchor) * mu_d_inv1
                        mu_d_for2 = rhs_3[:emb_dim//2] * rel_3[emb_dim//2:]
                        h_d_for2 = (1/cov_anchor) * mu_d_for2
                        mu_d_inv2 = rhs_3[emb_dim//2:] * rel_3[:emb_dim//2]
                        h_d_inv2 = (1/cov_anchor) * mu_d_inv2
                        # mu_d_for = (mu_d_for1 + mu_d_for2)/2
                        # mu_d_inv = (mu_d_inv1 + mu_d_inv2)/2
                        # h_d_for = (1/cov_anchor) * mu_d_for
                        # h_d_inv = (1/cov_anchor) * mu_d_inv
                        # print("dist between mu_d and mu_m")
                        # print(torch.cdist(mu_m_for.unsqueeze(dim=0), mu_d_inv.unsqueeze(dim=0), p=2))
                        # print(torch.cdist(mu_m_inv.unsqueeze(dim=0), mu_d_for.unsqueeze(dim=0), p=2))
                        

                        h_m_for = h_m_for + (h_d_inv1 + h_d_inv2)/2
                        J_m_for = (1/cov_var) + (1/cov_anchor) + (1/cov_anchor)
                        h_m_inv = h_m_inv + (h_d_for1 + h_d_for2)/2
                        J_m_inv = (1/cov_var) + (1/cov_anchor) + (1/cov_anchor)
                        mu_m_for = h_m_for / J_m_for
                        mu_m_inv = h_m_inv / J_m_inv
                        # print("dist between mu_d and mu_m after update")
                        # print(torch.dist(mu_m_for.unsqueeze(dim=0), mu_d_inv.unsqueeze(dim=0), p=2))
                        # print(torch.dist(mu_m_inv.unsqueeze(dim=0), mu_d_for.unsqueeze(dim=0), p=2))

                        # print("dist between mu_gt and mu_m after update")
                        # print(torch.dist(mu_gt_for.unsqueeze(dim=0), mu_m_for.unsqueeze(dim=0), p=2))
                        # print(torch.dist(mu_gt_inv.unsqueeze(dim=0), mu_m_inv.unsqueeze(dim=0), p=2))
                        # sys.exit()

                        # update user belief
                        if j==0:
                            mu_u = lhs_1
                            mu_u_for = mu_u[:emb_dim//2]
                            mu_u_inv = mu_u[emb_dim//2:]
                            h_u_for = (1/cov_target) * mu_u_for
                            h_u_inv = (1/cov_target) * mu_u_inv
                            J_u_for = (1/cov_target)
                            J_u_inv = (1/cov_target)

                        h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * h_m_inv
                        J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * rel_1[:emb_dim//2]
                        h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * h_m_for
                        J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * rel_1[emb_dim//2:]

                        # h_u_for = h_u_for + h_m_inv; h_u_inv = h_u_inv + h_m_for
                        # J_u_for = J_u_for + (1/cov_var); J_u_inv = J_u_inv + (1/cov_var)
                        # mu_u_for = h_u_for / J_u_for
                        # mu_u_inv = h_u_inv / J_u_inv

                        mu_u_for = h_u_for / J_u_for
                        mu_u_inv = h_u_inv / J_u_inv



                        user_embs[i*5+j, :emb_dim//2] = mu_u_for
                        user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))

        elif env.graph_type == '2_3':
            part1, part2, part3, part4 = parts[0], parts[1], parts[2], parts[3]
            chain1, chain2, chain3, chain4 = chains[0], chains[1], chains[2], chains[3]

            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb, lhs_3_emb, rel_3_emb, rhs_3_emb, lhs_4_emb, rel_4_emb, rhs_4_emb = \
                chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2], chain3[0], chain3[1], chain3[2], chain4[0], chain4[1], chain4[2]
            #print(lhs_1_emb[0])
            if not 'SimplE' in str(self.model_type):
                raise NotImplementedError
            else:
                for i in tqdm.tqdm(range(nb_queries // 5)):
                    for j in range(5):
                        #print(intact_parts[0][i*5+j])
                        #print(intact_parts[1][i*5+j])
                        #print(intact_parts[2][i*5+j])
                        #print(intact_parts[3][i*5+j])
    
                        # lhs_1 is the user belief. rhs_2, rhs_3, and rhs_4 are the evidence embeddings
                        lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                        lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], rhs_2_emb[i*5+j]
                        lhs_3, rel_3, rhs_3 = None, rel_3_emb[i*5+j], rhs_3_emb[i*5+j]
                        lhs_4, rel_4, rhs_4 = None, rel_4_emb[i*5+j], rhs_4_emb[i*5+j]

                        mu_m_for = possible_tails_emb[0][i*5+j, :emb_dim//2]
                        h_m_for = (1/cov_var) * mu_m_for

                        mu_m_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                        h_m_inv = (1/cov_var) * mu_m_inv

                        mu_d_for1 = rhs_2[:emb_dim//2] * rel_2[emb_dim//2:]
                        h_d_for1 = (1/cov_anchor) * mu_d_for1

                        mu_d_inv1 = rhs_2[emb_dim//2:] * rel_2[:emb_dim//2]
                        h_d_inv1 = (1/cov_anchor) * mu_d_inv1

                        mu_d_for2 = rhs_3[:emb_dim//2] * rel_3[emb_dim//2:]
                        h_d_for2 = (1/cov_anchor) * mu_d_for2
                        mu_d_inv2 = rhs_3[emb_dim//2:] * rel_3[:emb_dim//2]
                        h_d_inv2 = (1/cov_anchor) * mu_d_inv2
                        mu_d_for3 = rhs_4[:emb_dim//2] * rel_4[emb_dim//2:]
                        h_d_for3 = (1/cov_anchor) * mu_d_for3
                        mu_d_inv3 = rhs_4[emb_dim//2:] * rel_4[:emb_dim//2]
                        h_d_inv3 = (1/cov_anchor) * mu_d_inv3

                        h_m_for = h_m_for + (h_d_inv1 + h_d_inv2 + h_d_inv3)/3

                        J_m_for = (1/cov_var) + (1/cov_anchor) + (1/cov_anchor) + (1/cov_anchor)
                        h_m_inv = h_m_inv + (h_d_for1 + h_d_for2 + h_d_for3)/3
                        J_m_inv = (1/cov_var) + (1/cov_anchor) + (1/cov_anchor) + (1/cov_anchor)
                        mu_m_for = h_m_for / J_m_for
                        mu_m_inv = h_m_inv / J_m_inv

                        # update user belief
                        if j==0:
                            mu_u = lhs_1
                            mu_u_for = mu_u[:emb_dim//2]
                            mu_u_inv = mu_u[emb_dim//2:]
                            h_u_for = (1/cov_target) * mu_u_for
                            h_u_inv = (1/cov_target) * mu_u_inv
                            J_u_for = (1/cov_target)
                            J_u_inv = (1/cov_target)

                        h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * h_m_inv

                        J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * rel_1[:emb_dim//2]
                        h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * h_m_for
                        J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * rel_1[emb_dim//2:]
                        # h_u_for = h_u_for + h_m_inv; h_u_inv = h_u_inv + h_m_for
                        # J_u_for = J_u_for + (1/cov_var); J_u_inv = J_u_inv + (1/cov_var)
                        mu_u_for = h_u_for / J_u_for
                        mu_u_inv = h_u_inv / J_u_inv


                        user_embs[i*5+j, :emb_dim//2] = mu_u_for
                        user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))
        
        elif env.graph_type == '3_3': # pi
            part1, part2, part3, part4 = parts[0], parts[1], parts[2], parts[3]
            chain1, chain2, chain3, chain4 = chains[0], chains[1], chains[2], chains[3]
            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb, lhs_3_emb, rel_3_emb, rhs_3_emb, lhs_4_emb, rel_4_emb, rhs_4_emb = \
                chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2], chain3[0], chain3[1], chain3[2], chain4[0], chain4[1], chain4[2]
            #print(lhs_1_emb[0])
            
            if not 'SimplE' in str(self.model_type):
                raise NotImplementedError
            else:
                for i in tqdm.tqdm(range(nb_queries // 5)):
                    for j in range(5):
                        lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                        lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], rhs_2_emb[i*5+j]
                        lhs_3, rel_3, rhs_3 = None, rel_3_emb[i*5+j], None
                        lhs_4, rel_4, rhs_4 = None, rel_4_emb[i*5+j], rhs_4_emb[i*5+j]
                        if j == 0:
                            mu_u = lhs_1
                            mu_u_for = mu_u[:emb_dim//2]
                            mu_u_inv = mu_u[emb_dim//2:]
                            h_u_for = (1/cov_target) * mu_u_for
                            h_u_inv = (1/cov_target) * mu_u_inv
                            J_u_for = (1/cov_target)
                            J_u_inv = (1/cov_target)
                        
                        var_scores, var_embs = self.get_best_candidates(rel_4.view(1,-1), rhs_4.view(1,-1), None, candidates=candidates, last_step=False, side='rhs')
                        top_vars_mean = torch.mean(var_embs.squeeze(dim=0), dim=0).view(emb_dim)
                        mu_d_for1 = top_vars_mean[:emb_dim//2]* rel_3[emb_dim//2:]; mu_d_inv1 = top_vars_mean[emb_dim//2:]* rel_3[:emb_dim//2]
                        h_d_for1 = (1/cov_anchor) * mu_d_for1; h_d_inv1 = (1/cov_anchor) * mu_d_inv1

                        mu_d_for2 = rhs_2[:emb_dim//2] * rel_2[emb_dim//2:]
                        h_d_for2 = (1/cov_anchor) * mu_d_for2
                        mu_d_inv2 = rhs_2[emb_dim//2:] * rel_2[:emb_dim//2]
                        h_d_inv2 = (1/cov_anchor) * mu_d_inv2

                        mu_m_for = possible_tails_emb[0][i*5+j, :emb_dim//2] 
                        mu_m_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                        h_m_for = (1/cov_var) * mu_m_for; h_m_inv = (1/cov_var) * mu_m_inv
                        J_m_for = 1/cov_var + 1/cov_anchor; J_m_inv = 1/cov_var + 1/cov_anchor

                        h_m_for = h_m_for + (h_d_inv1 + h_d_inv2)/2; h_m_inv = h_m_inv + (h_d_for1 + h_d_for2)/2
                        J_m_for = (1/cov_var) + (1/cov_var) + (1/cov_anchor)
                        J_m_inv = (1/cov_var) + (1/cov_var) + (1/cov_anchor)
                        mu_m_for = h_m_for / J_m_for
                        mu_m_inv = h_m_inv / J_m_inv

                        h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * h_m_inv
                        J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * rel_1[:emb_dim//2]
                        h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * h_m_for
                        J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * rel_1[emb_dim//2:]
                        mu_u_for = h_u_for / J_u_for
                        mu_u_inv = h_u_inv / J_u_inv
                        user_embs[i*5+j, :emb_dim//2] = mu_u_for
                        user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))
                        




                # for i in tqdm.tqdm(range(nb_queries // 5)):
                #     for j in range(5):
                #         lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                #         lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], rhs_2_emb[i*5+j]
                #         lhs_3, rel_3, rhs_3 = None, rel_3_emb[i*5+j], None
                #         lhs_4, rel_4, rhs_4 = None, rel_4_emb[i*5+j], rhs_4_emb[i*5+j]

                #         mu_m2_for = possible_tails_emb[2][i*5+j, :emb_dim//2]
                #         h_m2_for = (1/cov_var) * mu_m2_for
                #         mu_m2_inv = possible_tails_emb[2][i*5+j, emb_dim//2:]
                #         h_m2_inv = (1/cov_var) * mu_m2_inv
                #         mu_d2_for = rhs_4[:emb_dim//2] * rel_4[emb_dim//2:]
                #         h_d2_for = (1/cov_anchor) * mu_d2_for
                #         mu_d2_inv = rhs_4[emb_dim//2:] * rel_4[:emb_dim//2]
                #         h_d2_inv = (1/cov_anchor) * mu_d2_inv
                #         h_m2_for = h_m2_for + h_d2_inv; h_m2_inv = h_m2_inv + h_d2_for
                #         J_m2_for = (1/cov_var) + (1/cov_anchor); J_m2_inv = (1/cov_var) + (1/cov_anchor)
                #         mu_m2_for = h_m2_for / J_m2_for; mu_m2_inv = h_m2_inv / J_m2_inv

                #         mu_m1_for = possible_tails_emb[0][i*5+j, :emb_dim//2]
                #         h_m1_for = (1/cov_var) * mu_m1_for
                #         mu_m1_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                #         h_m1_inv = (1/cov_var) * mu_m1_inv
                #         h_m1_for = h_m1_for - rel_3[:emb_dim//2] * (1 / J_m2_inv) * h_m2_inv
                #         J_m1_for = (1/cov_var) - rel_3[:emb_dim//2] * (1 / J_m2_inv) * rel_3[:emb_dim//2]
                #         h_m1_inv = h_m1_inv - rel_3[emb_dim//2:] * (1 / J_m2_for) * h_m2_for
                #         J_m1_inv = (1/cov_var) - rel_3[emb_dim//2:] * (1 / J_m2_for) * rel_3[emb_dim//2:]
                #         # h_m1_for = h_m1_for + h_m2_inv; h_m1_inv = h_m1_inv + h_m2_for
                #         # J_m1_for = (1/cov_var) + J_m2_inv; J_m1_inv = (1/cov_var) + J_m2_for

                #         mu_d1_for = rhs_2[:emb_dim//2] * rel_2[emb_dim//2:]
                #         h_d1_for = (1/cov_anchor) * mu_d1_for
                #         mu_d1_inv = rhs_2[emb_dim//2:] * rel_2[:emb_dim//2]
                #         h_d1_inv = (1/cov_anchor) * mu_d1_inv
                #         h_m1_for = h_m1_for + h_d1_inv; h_m1_inv = h_m1_inv + h_d1_for
                #         J_m1_for = J_m1_for + (1/cov_anchor); J_m1_inv = J_m1_inv + (1/cov_anchor)

                #         if j == 0:
                #             mu_u = lhs_1
                #             mu_u_for = mu_u[:emb_dim//2]
                #             mu_u_inv = mu_u[emb_dim//2:]
                #             h_u_for = (1/cov_target) * mu_u_for
                #             h_u_inv = (1/cov_target) * mu_u_inv
                #             J_u_for = (1/cov_target)
                #             J_u_inv = (1/cov_target)

                #         h_u_for = h_u_for - rel_1[:emb_dim//2] * (1 / J_m1_inv) * h_m1_inv
                #         J_u_for = J_u_for - rel_1[:emb_dim//2] * (1 / J_m1_inv) * rel_1[:emb_dim//2]
                #         h_u_inv = h_u_inv - rel_1[emb_dim//2:] * (1 / J_m1_for) * h_m1_for
                #         J_u_inv = J_u_inv - rel_1[emb_dim//2:] * (1 / J_m1_for) * rel_1[emb_dim//2:]
                #         # h_u_for = h_u_for + h_m1_inv; h_u_inv = h_u_inv + h_m1_for
                #         # J_u_for = J_u_for + (1/cov_var); J_u_inv = J_u_inv + (1/cov_var)
                #         mu_u_for = h_u_for / J_u_for
                #         mu_u_inv = h_u_inv / J_u_inv
                #         user_embs[i*5+j, :emb_dim//2] = mu_u_for
                #         user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                # scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))


        elif env.graph_type == '4_3': #ip
            part1, part2, part3, part4 = parts[0], parts[1], parts[2], parts[3]
            chain1, chain2, chain3, chain4 = chains[0], chains[1], chains[2], chains[3]
            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb, lhs_3_emb, rel_3_emb, rhs_3_emb, lhs_4_emb, rel_4_emb, rhs_4_emb = \
                chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2], chain3[0], chain3[1], chain3[2], chain4[0], chain4[1], chain4[2]
            new_env = copy.deepcopy(env)
            new_env.chain_instructions = ['intersect_0_1']
            if not 'SimplE' in str(self.model_type):
                raise NotImplementedError
            else:
                for i in tqdm.tqdm(range(nb_queries // 5)):
                    for j in range(5):
                        lhs_4, rel_4, rhs_4 = None, rel_4_emb[i*5+j], rhs_4_emb[i*5+j]
                        lhs_3, rel_3, rhs_3 = None, rel_3_emb[i*5+j], rhs_3_emb[i*5+j]
                        lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], None
                        lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                        if j == 0:
                            mu_u = lhs_1
                            mu_u_for = mu_u[:emb_dim//2]
                            mu_u_inv = mu_u[emb_dim//2:]
                            h_u_for = (1/cov_target) * mu_u_for
                            h_u_inv = (1/cov_target) * mu_u_inv
                            J_u_for = (1/cov_target)
                            J_u_inv = (1/cov_target)
                        new_env.chains = [(chain1[0],chain1[1], chain1[2]),(chain3[0], chain3[1][i].view(1,-1), chain3[2][i].view(1,-1)), (chain4[0],chain4[1][i].view(1,-1),chain4[2][i].view(1,-1))]
                        top_vars2_scores = self.query_answering_BF_Exist(new_env, candidates=candidates, t_norm=t_norm, batch_size=batch_size, scores_normalize=scores_normalize, explain=explain)
                        _, top_vars2_indices = torch.topk(top_vars2_scores, candidates, dim=1)
                        top_vars2_embs = self.entity_embeddings(top_vars2_indices)
                        top_vars2_mean = torch.mean(top_vars2_embs.squeeze(dim=0), dim=0).view(emb_dim)
                        mu_d_for = top_vars2_mean[:emb_dim//2]* rel_2[emb_dim//2:]; mu_d_inv = top_vars2_mean[emb_dim//2:]* rel_2[:emb_dim//2]
                        h_d_for = (1/cov_var) * mu_d_for; h_d_inv = (1/cov_var) * mu_d_inv
                        mu_m_for = possible_tails_emb[0][i*5+j, :emb_dim//2] 
                        mu_m_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                        h_m_for = (1/cov_var) * mu_m_for; h_m_inv = (1/cov_var) * mu_m_inv
                        J_m_for = 1/cov_var + 1/cov_var; J_m_inv = 1/cov_var + 1/cov_var
                        h_m_for = h_m_for + h_d_inv; h_m_inv = h_m_inv + h_d_for

                        h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * h_m_inv
                        J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * rel_1[:emb_dim//2]
                        h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * h_m_for
                        J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * rel_1[emb_dim//2:]
                        mu_u_for = h_u_for / J_u_for
                        mu_u_inv = h_u_inv / J_u_inv
                        user_embs[i*5+j, :emb_dim//2] = mu_u_for
                        user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))

                # for i in tqdm.tqdm(range(nb_queries // 5)):
                #     for j in range(5):
                #         lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                #         lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], None
                #         lhs_3, rel_3, rhs_3 = None, rel_3_emb[i*5+j], rhs_3_emb[i*5+j]
                #         lhs_4, rel_4, rhs_4 = None, rel_4_emb[i*5+j], rhs_4_emb[i*5+j]

                #         mu_m2_for = possible_tails_emb[1][i*5+j, :emb_dim//2]
                #         h_m2_for = (1/cov_var) * mu_m2_for
                #         mu_m2_inv = possible_tails_emb[1][i*5+j, emb_dim//2:]
                #         h_m2_inv = (1/cov_var) * mu_m2_inv

                #         mu_d_for1 = rhs_3[:emb_dim//2] * rel_3[emb_dim//2:]
                #         h_d_for1 = (1/cov_anchor) * mu_d_for1
                #         mu_d_inv1 = rhs_3[emb_dim//2:] * rel_3[:emb_dim//2]
                #         h_d_inv1 = (1/cov_anchor) * mu_d_inv1
                #         mu_d_for2 = rhs_4[:emb_dim//2] * rel_4[emb_dim//2:]
                #         h_d_for2 = (1/cov_anchor) * mu_d_for2
                #         mu_d_inv2 = rhs_4[emb_dim//2:] * rel_4[:emb_dim//2]
                #         h_d_inv2 = (1/cov_anchor) * mu_d_inv2

                #         h_m2_for = h_m2_for + (h_d_inv1 + h_d_inv2)/2
                #         J_m2_for = (1/cov_var) + (1/cov_anchor) + (1/cov_anchor)
                #         h_m2_inv = h_m2_inv + (h_d_for1 + h_d_for2)/2
                #         J_m2_inv = (1/cov_var) + (1/cov_anchor) + (1/cov_anchor)
                #         mu_m2_for = h_m2_for / J_m2_for
                #         mu_m2_inv = h_m2_inv / J_m2_inv

                #         mu_m1_for = possible_tails_emb[0][i*5+j, :emb_dim//2]
                #         h_m1_for = (1/cov_var) * mu_m1_for
                #         mu_m1_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                #         h_m1_inv = (1/cov_var) * mu_m1_inv

                #         # h_m1_for = h_m1_for - rel_2[:emb_dim//2] * (1 / J_m2_inv) * h_m2_inv
                #         # J_m1_for = (1/cov_var) - rel_2[:emb_dim//2] * (1 / J_m2_inv) * rel_2[:emb_dim//2]
                #         # h_m1_inv = h_m1_inv - rel_2[emb_dim//2:] * (1 / J_m2_for) * h_m2_for
                #         # J_m1_inv = (1/cov_var) - rel_2[emb_dim//2:] * (1 / J_m2_for) * rel_2[emb_dim//2:]
                #         h_m1_for = h_m1_for + h_m2_inv; h_m1_inv = h_m1_inv + h_m2_for
                #         J_m1_for = (1/cov_var) + J_m2_inv; J_m1_inv = (1/cov_var) + J_m2_for
                #         mu_m1_for = h_m1_for / J_m1_for; mu_m1_inv = h_m1_inv / J_m1_inv

                #         if j == 0:
                #             mu_u = lhs_1
                #             mu_u_for = mu_u[:emb_dim//2]
                #             mu_u_inv = mu_u[emb_dim//2:]
                #             h_u_for = (1/cov_target) * mu_u_for
                #             h_u_inv = (1/cov_target) * mu_u_inv
                #             J_u_for = (1/cov_target)
                #             J_u_inv = (1/cov_target)
                        
                #         h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m1_inv) * h_m1_inv
                #         J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m1_inv) * rel_1[:emb_dim//2]
                #         h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m1_for) * h_m1_for
                #         J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m1_for) * rel_1[emb_dim//2:]
                #         # h_u_for = h_u_for + h_m1_inv; h_u_inv = h_u_inv + h_m1_for
                #         # J_u_for = J_u_for + (1/J_m1_inv); J_u_inv = J_u_inv + J_m1_for
                #         mu_u_for = h_u_for / J_u_for
                #         mu_u_inv = h_u_inv / J_u_inv
                #         user_embs[i*5+j, :emb_dim//2] = mu_u_for
                #         user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                # scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))


        return scores

    def query_answering_BF_InstMarginal(self, env: DynKBCSingleton, candidates: int=5, t_norm: str='min', 
    batch_size=1, scores_normalize=0, explain='no', cov_anchor=1e-2, cov_var=1e-2, cov_target=1e-2, instantiations: int=3):       
        parts = env.parts
        chains, chain_instructions = env.chains, env.chain_instructions
        nb_queries, emb_dim = chains[0][0].shape[0], chains[0][0].shape[1]
        possible_heads_emb = env.possible_heads_emb; possible_tails_emb = env.possible_tails_emb
        

        user_embs = torch.zeros((nb_queries, emb_dim), device=Device)
        if env.graph_type == '1_3':
            
            part1, part2, part3 = parts[0], parts[1], parts[2]
            chain1, chain2, chain3 = chains[0], chains[1], chains[2]
            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb, lhs_3_emb, rel_3_emb, rhs_3_emb = chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2], chain3[0], chain3[1], chain3[2]
            for i in tqdm.tqdm(range(nb_queries// 5)):
                for j in range(5):
                    if j == 0:
                        mu_u = torch.zeros(1, emb_dim, device=Device)
                        mu_u_for = mu_u[0,:emb_dim//2]
                        mu_u_inv = mu_u[0,emb_dim//2:]
                        h_u_for = (1/cov_target) * mu_u_for
                        h_u_inv = (1/cov_target) * mu_u_inv
                        J_u_for = 1/cov_target
                        J_u_inv = 1/cov_target
                    lhs_3, rel_3, rhs_3 = None, rel_3_emb[i*5+j], rhs_3_emb[i*5+j]
                    lhs_2, rel_2, rhs_2 = None, rel_2_emb[i*5+j], None
                    lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                    top_vars2_scores, top_vars2_embs = self.get_best_candidates(rel_3.view(1,-1), rhs_3.view(1,-1), None, candidates=instantiations, last_step=False, side='rhs')
                    top_vars2_mean = torch.mean(top_vars2_embs.squeeze(dim=0), dim=0).view(1, emb_dim)
                    mu_d_for = top_vars2_mean[:, :emb_dim//2] * rel_2[:,emb_dim//2:]; mu_d_inv = top_vars2_mean[:, emb_dim//2:]* rel_2[:,:emb_dim//2]
                    h_d_for = (1/cov_anchor) * mu_d_for; h_d_inv = (1/cov_anchor) * mu_d_inv
                    mu_m_for = possible_tails_emb[0][i*5+j, :emb_dim//2] 
                    mu_m_inv = possible_tails_emb[0][i*5+j, emb_dim//2:]
                    h_m_for = (1/cov_var) * mu_m_for; h_m_inv = (1/cov_var) * mu_m_inv
                    J_m_for = 1/cov_var + 1/cov_anchor; J_m_inv = 1/cov_var + 1/cov_anchor
                    h_m_for = h_m_for + h_d_inv; h_m_inv = h_m_inv + h_d_for

                    h_u_for = h_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * h_m_inv
                    J_u_for = J_u_for + rel_1[:emb_dim//2] * (1 / J_m_inv) * rel_1[:emb_dim//2]
                    h_u_inv = h_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * h_m_for
                    J_u_inv = J_u_inv + rel_1[emb_dim//2:] * (1 / J_m_for) * rel_1[emb_dim//2:]
                    mu_u_for = h_u_for / J_u_for
                    mu_u_inv = h_u_inv / J_u_inv
                    user_embs[i*5+j, :emb_dim//2] = mu_u_for
                    user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                    
            scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))
            return scores

    def query_answering_BF_Instantiated(self, env: DynKBCSingleton, candidates: int=5, t_norm: str='min', 
    batch_size=1, scores_normalize=0, explain='no', cov_anchor=1e-2, cov_var=1e-2, cov_target=1e-2, instantiations: int=3):
        # scores will tell us which items are the most probable instantiation of the evidences
        scores = self.query_answering_BF_Exist(env)
        if 'disj' in env.graph_type:
            objective = self.t_conorm
        else: 
            objective = self.t_norm
        parts = env.parts
        chains, chain_instructions = env.chains, env.chain_instructions
        nb_queries, emb_dim = chains[0][0].shape[0], chains[0][0].shape[1]
        possible_heads_emb = env.possible_heads_emb; possible_tails_emb = env.possible_tails_emb
        
        user_embs = torch.empty((nb_queries, emb_dim), device=Device)
        if env.graph_type == '1_2':
            part1 , part2 = parts[0], parts[1]
            chain1, chain2 = chains[0], chains[1]
            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb = chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2]
        
        elif env.graph_type == '2_2' or env.graph_type == '1_3':
            part1, part2, part3 = parts[0], parts[1], parts[2]
            chain1, chain2, chain3 = chains[0], chains[1], chains[2]
            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb, lhs_3_emb, rel_3_emb, rhs_3_emb = chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2], chain3[0], chain3[1], chain3[2]
        elif env.graph_type == '2_3' or env.graph_type == '1_4' or env.graph_type == '3_3' or env.graph_type == '4_3':
            part1, part2, part3, part4 = parts[0], parts[1], parts[2], parts[3]
            chain1, chain2, chain3, chain4 = chains[0], chains[1], chains[2], chains[3]
            lhs_1_emb, rel_1_emb, rhs_1_emb, lhs_2_emb, rel_2_emb, rhs_2_emb, lhs_3_emb, rel_3_emb, rhs_3_emb, lhs_4_emb, rel_4_emb, rhs_4_emb = \
                chain1[0], chain1[1], chain1[2], chain2[0], chain2[1], chain2[2], chain3[0], chain3[1], chain3[2], chain4[0], chain4[1], chain4[2]

        if not 'SimplE' in str(self.model_type):
            raise NotImplementedError
        else:
            for i in tqdm.tqdm(range(nb_queries // 5)):
                for j in range(5):
                    lhs_1, rel_1, rhs_1 = lhs_1_emb[i*5+j], rel_1_emb[i*5+j], None
                    #instantiated_ents = torch.topk(scores[i*5+j], instantiations).indices
                    instantiated_ents = torch.topk(scores[i*5+j], instantiations).indices
                        
                    if j == 0:
                        user_belief = lhs_1
                        J_u_for = 1/cov_target
                        J_u_inv = 1/cov_target
                    rhs = torch.empty((instantiations, emb_dim), device=Device)
                    # for ent in instantiated_ents:
                    #     rhs_1 = self.entity_embeddings(ent)
                    #     h_m_for = (1/cov_var) * rhs_1[:emb_dim//2] * rel_1[emb_dim//2:]
                    #     h_m_inv = (1/cov_var) * rhs_1[emb_dim//2:] * rel_1[:emb_dim//2]
                    #     h_u_for = J_u_for * user_belief[:emb_dim//2] + h_m_inv
                    #     h_u_inv = J_u_inv * user_belief[emb_dim//2:] + h_m_for
                    #     J_u_for = J_u_for + (1/cov_var) 
                    #     J_u_inv = J_u_inv + (1/cov_var)
                    #     mu_u_for = h_u_for / J_u_for
                    #     mu_u_inv = h_u_inv / J_u_inv
                    for k, ent in enumerate(instantiated_ents):
                        rhs_1 = self.entity_embeddings(ent)
                        rhs[k] = rhs_1
                    rhs_1 = rhs.mean(dim=0)
                    h_m_for = (1/cov_var) * rhs_1[:emb_dim//2] * rel_1[emb_dim//2:]
                    h_m_inv = (1/cov_var) * rhs_1[emb_dim//2:] * rel_1[:emb_dim//2]
                    h_u_for = J_u_for * user_belief[:emb_dim//2] + h_m_inv
                    h_u_inv = J_u_inv * user_belief[emb_dim//2:] + h_m_for
                    J_u_for = J_u_for + (1/cov_var)
                    J_u_inv = J_u_inv + (1/cov_var)
                    mu_u_for = h_u_for / J_u_for
                    mu_u_inv = h_u_inv / J_u_inv
                        
                    user_embs[i*5+j, :emb_dim//2] = mu_u_for
                    user_embs[i*5+j, emb_dim//2:] = mu_u_inv
                
            scores = self.forward_emb(user_embs, rel_1_emb[0].unsqueeze(dim=0))
        



        return scores

    # def query_answering_BF_Exist(self, env: DynKBCSingleton, candidates: int = 5, t_norm: str = 'min', 
    # batch_size=1, scores_normalize=0, explain=False, user_belief=None):

    #     res = None
    #     if 'disj' in env.graph_type:
    #         objective = self.t_conorm
    #     else: 
    #         objective = self.t_norm
    #     chains, chain_instructions = env.chains, env.chain_instructions
    #     nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]
    #     scores = None
    #     batches = make_batches(nb_queries, batch_size)
    #     # batches = [(0, 1), (1, 2), ...)]
    #     for i, batch in enumerate(tqdm.tqdm(batches)):
    #         nb_branches = 1
    #         nb_ent = 0
    #         batch_scores = None
    #         candidate_cache = {}
    #         batch_size = batch[1] - batch[0]
    #         dnf_flag = False
    #         if 'disj' in env.graph_type:
    #             dnf_flag = True


    #         for inst_ind, inst in enumerate(chain_instructions):

    #             # inst = "hop_0_1"
    #             # inst_ind = 0
    #             with torch.no_grad():
    #                 # in fact our projection is like an intersection for the cqd (item has a fact and is 
    #                 # liked by the user)
    #                 if 'hop' in inst or 'inter' in inst:

    #                     ind_1 = int(inst.split("_")[-2])
    #                     ind_2 = int(inst.split("_")[-1])
    #                     # indices = [0,1]
    #                     indices = [ind_1, ind_2]

    #                     if objective == self.t_norm and dnf_flag:
    #                         objective = self.t_conorm
    #                     if 'inter' in inst:

    #                         if len(inst.split("_")) == 4:
    #                             ind_1 = 0
    #                             ind_2 = int(inst.split("_")[-2])
    #                             ind_3 = int(inst.split("_")[-1])
    #                             indices = [ind_1, ind_2, ind_3]
    #                         elif len(inst.split("_")) == 5:
    #                             ind_1 = 0
    #                             ind_2 = int(inst.split("_")[-3])
    #                             ind_3 = int(inst.split("_")[-2])
    #                             ind_4 = int(inst.split("_")[-1])
    #                             indices = [ind_1, ind_2, ind_3, ind_4]

    #                     for intersection_num, ind in enumerate(indices):

    #                         # ind = 0 - 1 
    #                         # intersection_num = 0 - 1 
    #                         last_step = (inst_ind == len(chain_instructions)-1)
    #                         # last_step = True
                            
    #                         lhs, rel, rhs = chains[ind]
                            
    #                         # this "if" only happens for the first part of the chain ([user, likes, ?])
    #                         if lhs is not None:
    #                             if user_belief is not None:
    #                                 lhs = user_belief
    #                             lhs = lhs[batch[0]:batch[1]]
    #                             lhs = lhs.view(-1, 1,
    #                                            embedding_size).repeat(1, nb_branches, 1)

    #                             lhs = lhs.view(-1, embedding_size)
    #                             # lhs becomes [1, emb_size]
    #                             rel = rel[batch[0]:batch[1]]
    #                             rel = rel.view(-1, 1,
    #                                        embedding_size).repeat(1, nb_branches, 1)
    #                             rel = rel.view(-1, embedding_size)
    #                             # rel becomes [1, emb_size]

    #                             if intersection_num > 0 and 'disj' in env.graph_type:
    #                                 raise NotImplementedError

    #                             if f"rhs_{ind}" not in candidate_cache or last_step:
    #                                 # z_scores is the scores of all entities to be the 
    #                                 # rhs and rhs_3d is their embeddings ([1,no_entity, emb_size])
    #                                 z_scores, rhs_3d = self.get_best_candidates(
    #                                 rel, lhs, None, candidates, last_step, None)
    #                                 z_scores_1d = z_scores.view(-1)

    #                                 if 'disj' in env.graph_type or scores_normalize:
    #                                     z_scores_1d = torch.sigmoid(z_scores_1d)

    #                                 if not last_step:
    #                                     nb_sources = rhs_3d.shape[0] * \
    #                                         rhs_3d.shape[1]
    #                                     nb_branches = nb_sources // batch_size
    #                                 else:
    #                                     if ind == indices[0]:
    #                                         nb_ent = rhs_3d.shape[1]
    #                                     else:
    #                                         nb_ent = 1
 
    #                                     # first time the batch scores is None and the z_scores we make it equal to z_scores_1d
    #                                     batch_scores = z_scores_1d if batch_scores is None else objective(
    #                                         z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)
    #                                     nb_ent = rhs_3d.shape[1]

    #                                 candidate_cache[f"rhs_{ind}"] = (batch_scores, rhs_3d)

    #                                 if ind == indices[0] and 'disj' in env.graph_type:
    #                                     raise NotImplementedError

    #                                 #if ind == indices[-1]:
    #                                 #    candidate_cache[f"lhs_{ind+1}"] = (batch_scores, rhs_3d)
    #                             else:
    #                                 raise NotImplementedError
    #                             del lhs, rel, rhs, rhs_3d, z_scores_1d, z_scores
                                    
    #                         # this is for the second part of the chain ([item, rel, tail])
    #                         elif ind>0:
    #                             rhs = rhs[batch[0]:batch[1]]
    #                             rhs = rhs.view(-1, 1,
    #                                            embedding_size).repeat(1, nb_branches, 1)
    #                             rhs = rhs.view(-1, embedding_size)
    #                             rel = rel[batch[0]:batch[1]]
    #                             rel = rel.view(-1, 1,
    #                                        embedding_size).repeat(1, nb_branches, 1)
    #                             rel = rel.view(-1, embedding_size)

    #                             if intersection_num > 0 and 'disj' in env.graph_type:
    #                                 raise NotImplementedError
    #                             if f"lhs_{ind}" not in candidate_cache or last_step:
    #                                 z_scores, lhs_3d = self.get_best_candidates(
    #                                     rel, rhs, None, candidates, last_step, None, 'rhs')
    #                                 z_scores_1d = z_scores.view(-1)

    #                                 if 'disj' in env.graph_type or scores_normalize:
    #                                     z_scores_1d = torch.sigmoid(z_scores_1d)
    #                                 # TODO: check this
    #                                 if not last_step:
    #                                     nb_sources = lhs_3d.shape[0] * \
    #                                         lhs_3d.shape[1]
    #                                     nb_branches = nb_sources // batch_size
    #                                 if not last_step:
    #                                     batch_scores = z_scores_1d if batch_scores is None else objective(
    #                                         z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
    #                                 else:

    #                                     if ind == indices[0]:
    #                                         nb_ent = lhs_3d.shape[1]
    #                                     else:
    #                                         nb_ent = 1
                                        
    #                                     batch_scores = z_scores_1d if batch_scores is None else objective(
    #                                         z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)
    #                                     nb_ent = lhs_3d.shape[1]
    #                                 candidate_cache[f"lhs_{ind}"] = (batch_scores, lhs_3d)

    #                             else:
    #                                 raise NotImplementedError
    #                             del lhs, rel, rhs, lhs_3d, z_scores_1d, z_scores
                                
    #         if batch_scores is not None:

    #             scores_2d = batch_scores.view(batch_size, -1, nb_ent)
    #             res, _ = torch.max(scores_2d, dim=1)
    #             scores = res if scores is None else torch.cat([scores, res])
    #             del batch_scores, scores_2d, res, candidate_cache

    #         else:
    #             assert False, "Batch Scores are empty: an error went uncaught."
    #         res = scores

    #     return res
    def query_answering_BF_Exist(self, env: DynKBCSingleton, candidates: int = 5, t_norm: str = 'min', 
    batch_size=1, scores_normalize=0, explain=False, user_belief=None):

        res = None
        # for disjunction, we need to use the t-conorm
        if 'disj' in env.graph_type:
            objective = self.t_conorm
        else:
            objective = self.t_norm

        chains, chain_instructions = env.chains, env.chain_instructions
        # chain_instructions = ['hop_0_1']
        # chains = [part1, part2]
        # part1 = [lhs_1, rels_1, rhs_1]
        # len(lhs_2) = 8000


        # in our lists, the order is from target to anchor, but this code assumes the opposite, so we reverse chains
        chains = chains[::-1]
        # for now, we're neglecting the 'user' part of the chain evidence
        chains = chains[:-1]

        nb_queries, embedding_size = chains[0][2].shape[0], chains[0][2].shape[1]

        scores = None

        # data_loader = DataLoader(dataset=chains, batch_size=16, shuffle=False)

        batches = make_batches(nb_queries, batch_size)
        # batches = [(0,1), (1,2), (2,3), ...]

        #for i, batch in enumerate(tqdm.tqdm(batches)):
        for i , batch in enumerate(batches):
            nb_branches = 1
            nb_ent = 0
            batch_scores = None
            candidate_cache = {}

            batch_size = batch[1] - batch[0]
            # here, batch_size is 1
            # torch.cuda.empty_cache()
            dnf_flag = False
            if 'disj' in env.graph_type:
                dnf_flag = True

            for inst_ind, inst in enumerate(chain_instructions):
                with torch.no_grad():
                    # this if for the case of projection
                    if 'hop' in inst:
                        if len(inst.split("_")) == 2:
                            # this is 2p where we only have one hop
                            ind_1 = int(inst.split("_")[-1])
                            indices = [ind_1]

                            #  ATTENTION! this is last hop only for the case of neglecting the user
                            last_hop = True
                        elif len(inst.split("_")) == 3:

                            ind_1 = int(inst.split("_")[-2])
                            ind_2 = int(inst.split("_")[-1])

                            indices = [ind_1, ind_2]
                            last_hop = False
                        elif len(inst.split("_")) == 4:
                            ind_1 = int(inst.split("_")[-3])
                            ind_2 = int(inst.split("_")[-2])
                            ind_3 = int(inst.split("_")[-1])
                            indices = [ind_1, ind_2, ind_3]
                            last_hop = False
                        # indices = [0, 1]
                        # each index is one hop

                        if objective == self.t_conorm and dnf_flag:
                            objective = self.t_norm

                        
                        for hop_num, ind in enumerate(indices):
                            
                            # print("HOP")
                            # print(candidate_cache.keys())
                            last_step = (inst_ind == len(
                                chain_instructions)-1) and last_hop


                            lhs, rel, rhs = chains[ind]

                            # [a, p, X], [X, p, Y][Y, p, Z]

                            # takes one of the lhs, rel (their embeddings)
                            if rhs is not None:
                                rhs = rhs[batch[0]:batch[1]]

                            else:
                                # print("MTA BRAT")
                                batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                                rhs = rhs_3d.view(-1, embedding_size)
                            rel = rel[batch[0]:batch[1]]
                            rel = rel.view(-1, 1,
                                           embedding_size).repeat(1, nb_branches, 1)
                            rel = rel.view(-1, embedding_size)
                            if f"lhs_{ind}" not in candidate_cache:
                                # gets best candidates for the rhs of this hop and the scores
                                z_scores, lhs_3d = self.get_best_candidates(
                                    rel, rhs, None, candidates, last_step, None, 'rhs')

                                # z_scores : tensor of shape [Num_queries * Candidates^K]
                                # rhs_3d : tensor of shape [Num_queries, Candidates^K, Embedding_size]

                                # [Num_queries * Candidates^K]
                                z_scores_1d = z_scores.view(-1)
                                if 'disj' in env.graph_type or scores_normalize:
                                    z_scores_1d = torch.sigmoid(z_scores_1d)

                                # B * S
                                nb_sources = lhs_3d.shape[0]*lhs_3d.shape[1]
                                nb_branches = nb_sources // batch_size
                                # if the batch_score is None, we initialize it with the candidates scores (since there's just one hop). otherwise, the t-norm is applied
                                if not last_step:
                                    batch_scores = z_scores_1d if batch_scores is None else objective(
                                        z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
                                else:
                                    nb_ent = lhs_3d.shape[1]
                                    batch_scores = z_scores_1d if batch_scores is None else objective(
                                        z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)
                                # candidate_cache stores the scores and the candidate embeddings for each rhs
                                candidate_cache[f"lhs_{ind}"] = (
                                    batch_scores, lhs_3d)
                                if not last_hop:
                                    # candidate_cache of the lhs of this hop is the rhs of the next hop
                                    # remember that since we've reverted the chains, this should be hop_num +1 not -1
                                    candidate_cache[f"rhs_{indices[hop_num+1]}"] = (
                                        batch_scores, lhs_3d)

                            else:
                                # if we already have the lhs of this hop, we are in the last hop (so no more rhs)
                                batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                                candidate_cache[f"rhs_{ind+1}"] = (
                                    batch_scores, lhs_3d)
                                last_hop = True
                                del lhs, rel
                                # #torch.cuda.empty_cache() 
                                continue
                            if not last_hop:
                                if hop_num == indices[-2]:
                                    last_hop = True
                            del lhs, rel, rhs, lhs_3d, z_scores_1d, z_scores
                            # #torch.cuda.empty_cache()

                    elif 'inter' in inst:
                        ind_1 = int(inst.split("_")[-2])
                        ind_2 = int(inst.split("_")[-1])

                        indices = [ind_1, ind_2]

                        if objective == self.t_norm and dnf_flag:
                            objective = self.t_conorm

                        if len(inst.split("_")) > 3:
                            ind_1 = int(inst.split("_")[-3])
                            ind_2 = int(inst.split("_")[-2])
                            ind_3 = int(inst.split("_")[-1])

                            indices = [ind_1, ind_2, ind_3]

                        for intersection_num, ind in enumerate(indices):
                            # print("intersection")
                            # print(candidate_cache.keys())

                            # and ind == indices[0]
                            last_step = (inst_ind == len(chain_instructions)-1)
                            last_seq_step = (inst_ind % 2 == 1)

                            lhs, rel, rhs = chains[ind]

                            if rhs is not None:
                                rhs = rhs[batch[0]:batch[1]]
                                rhs = rhs.view(-1, 1,
                                               embedding_size).repeat(1, nb_branches, 1)
                                rhs = rhs.view(-1, embedding_size)

                            else:
                                batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                                rhs = rhs_3d.view(-1, embedding_size)
                                nb_sources = rhs_3d.shape[0]*rhs_3d.shape[1]
                                nb_branches = nb_sources // batch_size

                            rel = rel[batch[0]:batch[1]]
                            rel = rel.view(-1, 1,
                                           embedding_size).repeat(1, nb_branches, 1)
                            rel = rel.view(-1, embedding_size)

                            # not implemented for now
                            if intersection_num > 0 and 'disj' in env.graph_type:
                                batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                                lhs = lhs_3d.view(-1, embedding_size)
                                z_scores = self.score_fixed(
                                    rel, lhs, rhs, candidates)

                                z_scores_1d = z_scores.view(-1)
                                if 'disj' in env.graph_type or scores_normalize:
                                    z_scores_1d = torch.sigmoid(z_scores_1d)

                                batch_scores = z_scores_1d if batch_scores is None else objective(
                                    z_scores_1d, batch_scores, t_norm)

                                continue

                            if f"lhs_{ind}" not in candidate_cache or last_step:
                                z_scores, lhs_3d = self.get_best_candidates(
                                    rel, rhs, None, candidates, last_step, None, 'rhs')

                                # [B * Candidates^K] or [B, S-1, N]
                                z_scores_1d = z_scores.view(-1)
                                # print(z_scores_1d)
                                if 'disj' in env.graph_type or scores_normalize:
                                    z_scores_1d = torch.sigmoid(z_scores_1d)

                                if not last_step:
                                    nb_sources = lhs_3d.shape[0] * \
                                        lhs_3d.shape[1]
                                    nb_branches = nb_sources // batch_size

                                if not last_step:
                                    batch_scores = z_scores_1d if batch_scores is None else objective(
                                        z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
                                else:
                                    if ind == indices[0]:
                                        nb_ent = lhs_3d.shape[1]
                                    else:
                                        nb_ent = 1

                                    batch_scores = z_scores_1d if batch_scores is None else objective(
                                        z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)
                                    nb_ent = lhs_3d.shape[1]

                                candidate_cache[f"lhs_{ind}"] = (
                                    batch_scores, lhs_3d)

                                if ind == indices[0] and 'disj' in env.graph_type:
                                    count = len(indices)-1
                                    iterator = 1
                                    while count > 0:
                                        candidate_cache[f"lhs_{indices[intersection_num+iterator]}"] = (
                                            batch_scores, lhs_3d)
                                        iterator += 1
                                        count -= 1

                                if ind == indices[-1]:
                                    candidate_cache[f"rhs_{ind+1}"] = (
                                        batch_scores, lhs_3d)
                            else:
                                batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                                candidate_cache[f"rhs_{ind+1}"] = (
                                    batch_scores, lhs_3d)

                                last_hop = True
                                del rhs, rel
                                continue

                            del lhs, rel, rhs, lhs_3d, z_scores_1d, z_scores

            if batch_scores is not None:
                # [B * entites * S ]
                # S ==  K**(V-1)

                scores_2d = batch_scores.view(batch_size, -1, nb_ent)

                # [1,candidates, nb_ent]
                # res is the max score for each entity among the candidates
                res, _ = torch.max(scores_2d, dim=1)
                scores = res if scores is None else torch.cat([scores, res])

                del batch_scores, scores_2d, res, candidate_cache

            else:
                assert False, "Batch Scores are empty: an error went uncaught."
            res = scores

        # res has the score of each entity for each query
        return res

    def query_answering_Bayesian1(self, env: DynKBCSingleton, candidates: int = 5, t_norm: str = 'min', batch_size=1, scores_normalize=0, explain=False,
    cov_anchor=None, cov_var=None, cov_target=None):
        if env.graph_type == '1_1_seq':
            last_step = False
            gt_targets = env.target_ids_hard
            chains = env.chains
            chain1, chain2, chain3 = chains
            seq_chains = [chain1, chain2, chain3]
            nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]
            scores = torch.empty((3, nb_queries, self.sizes[0])).to(chains[0][0].device)
            for i in tqdm.tqdm(range(nb_queries)):
                gts = list(gt_targets.values())[i]  
                for seq in range(3):
                    if seq == 0:
                        target_emb = torch.zeros((1, embedding_size)).to(chains[0][0].device)
                    chain = seq_chains[seq]
                    lhs, rel, rhs = chain
                    lhs = lhs[i].view(-1, embedding_size)
                    rel = rel[i].view(-1, embedding_size)
                    evidence_mean = lhs * rel
                    target_emb = (1 /(cov_anchor + 1 * cov_target)) * ((cov_anchor)*target_emb + (1 * cov_target)*evidence_mean)
                    cov_target = (cov_target * cov_anchor) /(cov_anchor + 1 * cov_target)
                    rel_virtual = torch.ones_like(target_emb)
                    ent_scores = self.forward_emb(target_emb, rel_virtual)
                    scores[seq][i] = ent_scores.view(-1)

        if env.graph_type == '1_2_seq':
            last_step = False
            gt_targets = env.target_ids_hard

            chains = env.chains
            chain1 , chain2, chain3, chain4 = chains
            seq_chains = [chain1, chain2, chain3]
            nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]
            # one dimension for each seq. each seq has no_queries * entities dims
            scores = torch.empty((3, nb_queries, self.sizes[0])).to(chains[0][0].device)

            for i in tqdm.tqdm(range(nb_queries)):
                gts = list(gt_targets.values())[i]               
                for seq in range(3):
                    if seq==0:
                        target_emb = torch.zeros((1, embedding_size)).to(chains[0][0].device)
                    # chain is (anchor_emb, rel1_emb, None) in all cases
                    chain = seq_chains[seq]
                    lhs, rel, rhs = chain
                    lhs = lhs[i].view(-1, embedding_size)
                    rel = rel[i].view(-1, embedding_size)
                    # z_scores is the scores of top candidates and rhs_3d is the embeddings of top candidates
                    z_scores, rhs_3d = self.get_best_candidates(
                        rel, lhs, None, candidates, last_step, None)
                    # this is the mean of most likely variable candidates used as evidence
                    rhs_2d_mean = torch.mean(rhs_3d[0], dim=0).view(1, embedding_size)
                    rel_2 = (chain4[1][i]).view(1, embedding_size)
                    evidence_mean = rhs_2d_mean * rel_2

                    target_emb = (1 /(cov_anchor + candidates * cov_target)) * ((cov_anchor)*target_emb + (candidates * cov_target)*evidence_mean)
                    #print(torch.argmax(rhs_2d_mean))
                    cov_target = (cov_target * cov_anchor) /(cov_anchor + candidates * cov_target)
                    rel_virtual = torch.ones_like(target_emb)
                    ent_scores = self.forward_emb(target_emb, rel_virtual)
                    scores[seq][i] = ent_scores.view(-1)

                    #gt_ranks = []
                    #for gt in gts:
                    #    rank_gt = (ent_scores[0] > ent_scores[0][gt]).sum().item() + 1
                    #    gt_ranks.append(rank_gt)
                    #print(np.mean(gt_ranks))

        
        elif env.graph_type == '1_3_seq':
            gt_targets = env.target_ids_hard
            chains = env.chains
            new_env = copy.deepcopy(env)
            new_env.chain_instructions = ['hop_0_1']
            # # (all embeddings) chain1, chain2, and chain3 are the 3 anchors related to the var and chain4 is the first hop
            chain1 , chain2, chain3, chain4, chain5 = chains
            seq_chains = [chain1, chain2, chain3]
            nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]
            # one dimension for each seq. each seq has no_queries * entities dims
            scores = torch.empty((3, nb_queries, self.sizes[0])).to(chains[0][0].device)
            
            for i in tqdm.tqdm(range(nb_queries)):
                gts = list(gt_targets.values())[i]  
                for seq in range(3):
                    if seq==0:
                        target_emb = torch.zeros((1, embedding_size)).to(chains[0][0].device)
                    chain1 = seq_chains[seq]
                    new_env.chains = [(chain1[0][i].view(1,-1),chain1[1][i].view(1,-1),chain1[2]), (chain4[0], chain4[1][i].view(1,-1), chain4[2])]
                    scores_var = self.query_answering_BF(new_env, candidates=candidates, t_norm=t_norm, batch_size=batch_size, scores_normalize=scores_normalize, explain=explain)
                    _, top_var_indices = torch.topk(scores_var, candidates, dim=1)
                    top_var_embeddings = self.entity_embeddings(top_var_indices[0])
                    evidence_mean = (torch.mean(top_var_embeddings, dim=0).view(1, embedding_size)) * (chain5[1][i].view(1, embedding_size))
                    target_emb = (1 /(cov_anchor + candidates * cov_target)) * ((cov_anchor)*target_emb + (candidates * cov_target)*evidence_mean)
                    cov_target = (cov_target * cov_anchor) /(cov_anchor + candidates * cov_target)
                    rel_virtual = torch.ones_like(target_emb)
                    ent_scores = self.forward_emb(target_emb, rel_virtual)
                    scores[seq][i] = ent_scores.view(-1)

        elif env.graph_type == '2_2_seq' or env.graph_type == '2_2_disj_seq':
            
            gt_targets = env.target_ids_hard
            chains = env.chains
            chain1 , chain2, chain3, chain4, chain5, chain6 = chains
            nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]
            seq_chains = [[chain1, chain2], [chain3, chain4], [chain5, chain6]]
            scores = torch.zeros((3, nb_queries, self.sizes[0])).to(chains[0][0].device)
            for i in tqdm.tqdm(range(nb_queries)):
                gts = list(gt_targets.values())[i]
                for seq in range(3):
                    if seq==0:
                        target_emb = torch.zeros((1, embedding_size)).to(chains[0][0].device)
                    chain1_seq, chain2_seq = seq_chains[seq]
                    evidence1 = chain1_seq[0][i].view(1,-1) * chain1_seq[1][i].view(1,-1)
                    evidence2 = chain2_seq[0][i].view(1,-1) * chain2_seq[1][i].view(1,-1)
                    evidence_mean = (evidence1 + evidence2) / 2
                    target_emb = (1 /(cov_anchor + 2 * cov_target)) * ((cov_anchor)*target_emb + (2 * cov_target)*evidence_mean)
                    cov_target = (cov_target * cov_anchor) /(cov_anchor + 2 * cov_target)
                    rel_virtual = torch.ones_like(target_emb)
                    ent_scores = self.forward_emb(target_emb, rel_virtual)
                    scores[seq][i] = ent_scores.view(-1)
        # elif env.graph_type == '2_2_disj_seq':
        #     gt_targets = env.target_ids_hard
        #     chains = env.chains
        #     chain1 , chain2, chain3, chain4, chain5, chain6 = chains
        #     nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]
        #     seq_chains = [[chain1, chain2], [chain3, chain4], [chain5, chain6]]
        #     scores = torch.zeros((3, nb_queries, self.sizes[0])).to(chains[0][0].device)
        #     for i in tqdm.tqdm(range(nb_queries)):
        #         gts = list(gt_targets.values())[i]
        #         for seq in range(3):
        #             if seq==0:
        #                 target_emb = torch.zeros((1, embedding_size)).to(chains[0][0].device)
        #             chain1_seq, chain2_seq = seq_chains[seq]
        #             evidence1 = chain1_seq[0][i].view(1,-1) * chain1_seq[1][i].view(1,-1)
        #             evidence2 = chain2_seq[0][i].view(1,-1) * chain2_seq[1][i].view(1,-1)
        #             target_emb1 = (1 /(cov_anchor + cov_target)) * ((cov_anchor)*target_emb + (cov_target)*evidence1)
        #             target_emb2 = (1 /(cov_anchor + cov_target)) * ((cov_anchor)*target_emb + (cov_target)*evidence2)
        #             cov_target = (cov_target * cov_anchor) /(cov_anchor + 2* cov_target)
        #             rel_virtual = torch.ones_like(target_emb)
        #             ent_scores1 = self.forward_emb(target_emb1, rel_virtual)
        #             ent_scores2 = self.forward_emb(target_emb2, rel_virtual)
        #             scores[seq][i] = (0.5 * ent_scores1 + 0.5*ent_scores2)



        elif env.graph_type == '2_3_seq':
            gt_targets = env.target_ids_hard
            chains = env.chains
            chain1 , chain2, chain3, chain4, chain5, chain6, chain7, chain8, chain9 = chains
            nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]

            seq_chains = [[chain1, chain2, chain3], [chain4, chain5, chain6], [chain7, chain8, chain9]]
            scores = torch.zeros((3, nb_queries, self.sizes[0])).to(chains[0][0].device)
            for i in tqdm.tqdm(range(nb_queries)):
                gts = list(gt_targets.values())[i]
                for seq in range(3):
                    if seq==0:
                        target_emb = torch.zeros((1, embedding_size)).to(chains[0][0].device)
                    chain1_seq, chain2_seq, chain3_seq = seq_chains[seq]
                    evidence1 = chain1_seq[0][i].view(1,-1) * chain1_seq[1][i].view(1,-1)
                    evidence2 = chain2_seq[0][i].view(1,-1) * chain2_seq[1][i].view(1,-1)
                    evidence3 = chain3_seq[0][i].view(1,-1) * chain3_seq[1][i].view(1,-1)
                    evidence_mean = (evidence1 + evidence2 + evidence3) / 3
                    target_emb = (1 /(cov_anchor + 3 * cov_target)) * ((cov_anchor)*target_emb + (3 * cov_target)*evidence_mean)
                    cov_target = (cov_target * cov_anchor) /(cov_anchor + 3 * cov_target)
                    rel_virtual = torch.ones_like(target_emb)
                    ent_scores = self.forward_emb(target_emb, rel_virtual)
                    scores[seq][i] = ent_scores.view(-1)

        elif env.graph_type == '3_3_seq':
            last_step = False
            gt_targets = env.target_ids_hard
            chains = env.chains
            chain1 , chain2, chain3, chain4, chain5, chain6, chain7, chain8, chain9 = chains
            nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]

            seq_chains = [[chain1, chain2, chain3], [chain4, chain5, chain6], [chain7, chain8, chain9]]
            scores = torch.zeros((3, nb_queries, self.sizes[0])).to(chains[0][0].device)
            for i in tqdm.tqdm(range(nb_queries)):
                gts = list(gt_targets.values())[i]
                for seq in range(3):
                    if seq==0:
                        target_emb = torch.zeros((1, embedding_size)).to(chains[0][0].device)
                    chain1_seq, chain2_seq, chain3_seq = seq_chains[seq]
                    lhs_1, rel_1, rhs_1 = chain1_seq
                    lhs_1, rel_1 = lhs_1[i].view(1,-1), rel_1[i].view(1,-1)
                    # z_scores is the scores of top candidates and rhs_3d is the embeddings of top candidates
                    z_scores, rhs_3d = self.get_best_candidates(rel_1, lhs_1, None, candidates, last_step, None)
                    rhs_2d_mean = torch.mean(rhs_3d[0], dim=0).view(1, embedding_size)
                    rel_2 = (chain2_seq[1][i]).view(1, embedding_size)
                    evidence1_mean = rhs_2d_mean * rel_2

                    # for the second anchor
                    lhs_3, rel_3, rhs_3 = chain3_seq
                    lhs_3, rel_3 = lhs_3[i].view(1,-1), rel_3[i].view(1,-1)
                    z_scores_3, rhs_3d_3 = self.get_best_candidates(rel_3, lhs_3, None, candidates, last_step, None)
                    rhs_2d_mean_3 = torch.mean(rhs_3d_3[0], dim=0).view(1, embedding_size)
                    evidence3_mean = rhs_2d_mean_3

                    evidence_mean = (evidence1_mean + evidence3_mean) / 2
                    target_emb = (1 /(cov_anchor + candidates * cov_target)) * ((cov_anchor)*target_emb + (2*candidates * cov_target)*evidence_mean)
                    cov_target = (cov_target * cov_anchor) /(cov_anchor + 2* candidates * cov_target)
                    rel_virtual = torch.ones_like(target_emb)
                    ent_scores = self.forward_emb(target_emb, rel_virtual)
                    scores[seq][i] = ent_scores.view(-1)

        elif env.graph_type == '4_3_seq':
            gt_targets = env.target_ids_hard
            chains = env.chains
            new_env = copy.deepcopy(env)
            new_env.chain_instructions = ['intersect_0_1']
            chain1 , chain2, chain3, chain4, chain5, chain6, chain7, chain8, chain9 = chains
            nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]

            seq_chains = [[chain1, chain2, chain3], [chain4, chain5, chain6], [chain7, chain8, chain9]]
            scores = torch.zeros((3, nb_queries, self.sizes[0])).to(chains[0][0].device)
            for i in tqdm.tqdm(range(nb_queries)):
                gts = list(gt_targets.values())[i]
                for seq in range(3):
                    if seq==0:
                        target_emb = torch.zeros((1, embedding_size)).to(chains[0][0].device)
                    chain1_seq, chain2_seq, chain3_seq = seq_chains[seq]
                    new_env.chains = [(chain1_seq[0][i].view(1,-1), chain1_seq[1][i].view(1,-1), chain1_seq[2]), (chain2_seq[0][i].view(1,-1), chain2_seq[1][i].view(1,-1), chain2_seq[2])]
                    scores_var = self.query_answering_BF(new_env, candidates=candidates, t_norm=t_norm, batch_size=batch_size, scores_normalize=scores_normalize, explain=explain)
                    _, top_var_indices = torch.topk(scores_var, candidates, dim=1)
                    top_var_embeddings = self.entity_embeddings(top_var_indices[0])
                    evidence_mean = (torch.mean(top_var_embeddings, dim=0).view(1, embedding_size)) * (chain3_seq[1][i].view(1, embedding_size))
                    target_emb = (1 /(cov_anchor + candidates * cov_target)) * ((cov_anchor)*target_emb + (candidates * cov_target)*evidence_mean)
                    cov_target = (cov_target * cov_anchor) /(cov_anchor + candidates * cov_target)
                    rel_virtual = torch.ones_like(target_emb)
                    ent_scores = self.forward_emb(target_emb, rel_virtual)
                    scores[seq][i] = ent_scores.view(-1)
        elif env.graph_type == '4_3_disj_seq':
            gt_targets = env.target_ids_hard
            chains = env.chains
            new_env = copy.deepcopy(env)
            new_env.chain_instructions = ['intersect_0_1']
            new_env.graph_type = '2_2_disj'
            chain1 , chain2, chain3, chain4, chain5, chain6, chain7, chain8, chain9 = chains
            nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]
            seq_chains = [[chain1, chain2, chain3], [chain4, chain5, chain6], [chain7, chain8, chain9]]
            scores = torch.zeros((3, nb_queries, self.sizes[0])).to(chains[0][0].device)
            for i in tqdm.tqdm(range(nb_queries)):
                gts = list(gt_targets.values())[i]
                for seq in range(3):
                    if seq == 0:
                        target_emb = torch.zeros((1, embedding_size)).to(chains[0][0].device)
                    chain1_seq, chain2_seq, chain3_seq = seq_chains[seq]
                    new_env.chains = [(chain1_seq[0][i].view(1,-1), chain1_seq[1][i].view(1,-1), chain1_seq[2]), (chain2_seq[0][i].view(1,-1), chain2_seq[1][i].view(1,-1), chain2_seq[2])]
                    scores_var = self.query_answering_BF(new_env, candidates=candidates, t_norm=t_norm, batch_size=batch_size, scores_normalize=scores_normalize, explain=explain)
                    _, top_var_indices = torch.topk(scores_var, candidates, dim=1)
                    top_var_embeddings = self.entity_embeddings(top_var_indices[0])
                    evidence_mean = (torch.mean(top_var_embeddings, dim=0).view(1, embedding_size)) * (chain3_seq[1][i].view(1, embedding_size))
                    target_emb = (1 /(cov_anchor + candidates * cov_target)) * ((cov_anchor)*target_emb + (candidates * cov_target)*evidence_mean)
                    cov_target = (cov_target * cov_anchor) /(cov_anchor + candidates * cov_target)
                    rel_virtual = torch.ones_like(target_emb)
                    ent_scores = self.forward_emb(target_emb, rel_virtual)
                    scores[seq][i] = ent_scores.view(-1)



                    


        return scores

    def query_answering_BF(self, env: DynKBCSingleton, candidates: int = 5, t_norm: str = 'min', batch_size=1, scores_normalize=0, explain=False):

        res = None
        # for disjunction, we need to use the t-conorm
        if 'disj' in env.graph_type:
            objective = self.t_conorm
        else:
            objective = self.t_norm

        chains, chain_instructions = env.chains, env.chain_instructions
        chain_instructions = ['intersect_0_1','hop_1_3','hop_3_4']
        #chain_instructions = ['intersect_0_1_2_3_4_5']
        #chain_instructions = ['intersect_0_1','hop_1_3']
        #chain_instructions = ['hop_0_3', 'hop_3_4']
        # this is for 3_3
        if len(chain_instructions) == 6 and chain_instructions[0] == 'hop_0_1':
            chain_instructions = ['hop_0_1', 'intersect_1_2', 'hop_3_4', 'intersect_4_5', 'hop_6_7', 'intersect_7_8']
        # this is for both 4_3 and 4_3_seq
        elif len(chain_instructions) == 7 and chain_instructions[0] == 'intersect_0_1':
            chain_instructions = ['intersect_0_1', 'hop_1_2', 'intersect_3_4', 'hop_4_5', 'intersect_6_7', 'hop_7_8']
            #chain_instructions = ['intersect_0_1', 'hop_1_2', 'intersect_3_4', 'hop_4_5']
            #chain_instructions = ['intersect_0_1', 'hop_1_2']
        if env.graph_type == '2_2_disj_seq':
            chain_instructions = ['intersect_0_1', 'intersect_2_3', 'intersect_4_5']
            #chain_instructions = ['intersect_0_1', 'intersect_2_3']
            #chain_instructions = ['intersect_0_1']

        
        # chain_instructions = ['hop_0_1']
        # chains = [part1, part2]
        # part1 = [lhs_1, rels_1, rhs_1]
        # len(lhs_2) = 8000

        nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]
        scores = None

        # data_loader = DataLoader(dataset=chains, batch_size=16, shuffle=False)

        batches = make_batches(nb_queries, batch_size)

        seq_scores = []
        seq_rhs_3d = []

        # batches = [(0,1), (1,2), (2,3), ...]

        for i, batch in enumerate(tqdm.tqdm(batches)):
        #for i, batch in enumerate(batches):
            nb_branches = 1
            nb_ent = 0
            batch_scores = None
            candidate_cache = {}

            batch_size = batch[1] - batch[0]
            # here, batch_size is 1
            # torch.cuda.empty_cache()
            dnf_flag = False
            if 'disj' in env.graph_type:
                dnf_flag = True
            for inst_ind, inst in enumerate(chain_instructions):
                #print("inst", inst)
                with torch.no_grad():
                    # this if for the case of projection
                    if 'hop' in inst:

                        ind_1 = int(inst.split("_")[-2])
                        ind_2 = int(inst.split("_")[-1])

                        indices = [ind_1, ind_2]
                        # indices = [0, 1]
                        # each index is one hop

                        if objective == self.t_conorm and dnf_flag:
                            objective = self.t_norm

                        last_hop = False
                        for hop_num, ind in enumerate(indices):
                            # if inst_ind ==2  and ind ==3:
                            #     print(nb_branches)   # 25
                            #     sys.exit()
                            # we need to reset batch scores and nb_branches for beginning of hop in sequential setting
                            if len(chain_instructions) ==6 and (inst_ind == 2 or inst_ind == 4):
                                if ind ==3 or ind == 6:
                                    nb_branches = 1
                                batch_scores = None

                            last_step = (inst_ind == len(
                                chain_instructions)-1) and last_hop
                            if len(chain_instructions) == 6 and chain_instructions[0] == 'intersect_0_1':
                                last_step = (ind == indices[-1]) and last_hop

                            lhs, rel, rhs = chains[ind]

                            

                            # [a, p, X], [X, p, Y][Y, p, Z]

                            # takes one of the lhs, rel (their embeddings)
                            if lhs is not None:
                                lhs = lhs[batch[0]:batch[1]]

                            else:
                                # TODO: remove this
                                candidate_cache['lhs_3'] = candidate_cache['lhs_2']
                                batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                                lhs = lhs_3d.view(-1, embedding_size)

                            rel = rel[batch[0]:batch[1]]
                            rel = rel.view(-1, 1,
                                           embedding_size).repeat(1, nb_branches, 1)
                            rel = rel.view(-1, embedding_size)

                            
                            if f"rhs_{ind}" not in candidate_cache:
                                
                                z_scores, rhs_3d = self.get_best_candidates(
                                    rel, lhs, None, candidates, last_step, env if explain else None)
                                # z_scores : tensor of shape [Num_queries * Candidates^K]
                                # rhs_3d : tensor of shape [Num_queries, Candidates^K, Embedding_size]

                                # [Num_queries * Candidates^K]
                                z_scores_1d = z_scores.view(-1)
                                if 'disj' in env.graph_type or scores_normalize:
                                    z_scores_1d = torch.sigmoid(z_scores_1d)

                                # B * S
                                nb_sources = rhs_3d.shape[0]*rhs_3d.shape[1]
                                nb_branches = nb_sources // batch_size

                                # if the batch_score is None, we initialize it with the candidates scores (since there's just one hop). otherwise, the t-norm is applied
                                if not last_step:
                                    batch_scores = z_scores_1d if batch_scores is None else objective(
                                        z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)

                                else:
                                    nb_ent = rhs_3d.shape[1]

                                    batch_scores = z_scores_1d if batch_scores is None else objective(
                                        z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)
                                # candidate_cache stores the scores and the candidate embeddings for each rhs
                                candidate_cache[f"rhs_{ind}"] = (
                                    batch_scores, rhs_3d)
                                
                                if not last_hop:
                                    # candidate_cache of the rhs of this hop is the lhs of the next hop
                                    candidate_cache[f"lhs_{indices[hop_num+1]}"] = (
                                        batch_scores, rhs_3d)

                            else:
                                # if we already have the rhs of this hop, we are in the last hop (so no more lhs)
                                batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                                candidate_cache[f"lhs_{ind+1}"] = (
                                    batch_scores, rhs_3d)
                                last_hop = True
                                del lhs, rel
                                # #torch.cuda.empty_cache() 
                                continue

                            last_hop = True
                            # chain type is 4_3
                            if len(chain_instructions) == 6 and chain_instructions[0] == 'intersect_0_1' and last_step:
                                #seq_scores.append(batch_scores)
                                #seq_rhs_3d.append(rhs_3d)
                                if inst_ind == 1:
                                    seq_scores = 1 * batch_scores
                                else:
                                    #seq_scores = seq_scores * batch_scores
                                    seq_scores = self.t_norm(seq_scores, batch_scores, t_norm)
                                # TODO: resetting these params. Maybe do the same for 3_3

                                if ind != 8:
                                    nb_branches = 1
                                    nb_ent = 0
                                    batch_scores = None

                            del lhs, rel, rhs, rhs_3d, z_scores_1d, z_scores
                            # #torch.cuda.empty_cache()

                    elif 'inter' in inst:
                        if len(inst.split("_")) < 3:
                            ind_1 = int(inst.split("_")[-1])
                            indices = [ind_1]
                        else:
                            ind_1 = int(inst.split("_")[-2])
                            ind_2 = int(inst.split("_")[-1])

                            indices = [ind_1, ind_2]

                        if objective == self.t_norm and dnf_flag:
                            objective = self.t_conorm

                        if len(inst.split("_")) == 4:
                            ind_1 = int(inst.split("_")[-3])
                            ind_2 = int(inst.split("_")[-2])
                            ind_3 = int(inst.split("_")[-1])

                            indices = [ind_1, ind_2, ind_3]
                        elif len(inst.split("_")) == 5:
                            ind_1 = int(inst.split("_")[-4])
                            ind_2 = int(inst.split("_")[-3])
                            ind_3 = int(inst.split("_")[-2])
                            ind_4 = int(inst.split("_")[-1])
                            indices = [ind_1, ind_2, ind_3, ind_4]
                        elif len(inst.split("_")) == 7:
                            ind_1 = int(inst.split("_")[-6])
                            ind_2 = int(inst.split("_")[-5])
                            ind_3 = int(inst.split("_")[-4])
                            ind_4 = int(inst.split("_")[-3])
                            ind_5 = int(inst.split("_")[-2])
                            ind_6 = int(inst.split("_")[-1])
                            indices = [ind_1, ind_2, ind_3, ind_4, ind_5, ind_6]                        
                        last_step_seq = False

                        for intersection_num, ind in enumerate(indices):

                            if len(chain_instructions) ==6:
                                last_step = (inst_ind == len(chain_instructions)-1) or inst_ind % 2 == 1
                            else:
                                last_step = (inst_ind == len(chain_instructions)-1)
                            if env.graph_type == '2_2_disj_seq':
                                last_step = True

                            
                            lhs, rel, rhs = chains[ind]

                            if lhs is not None:
                                lhs = lhs[batch[0]:batch[1]]
                                lhs = lhs.view(-1, 1,
                                               embedding_size).repeat(1, nb_branches, 1)
                                lhs = lhs.view(-1, embedding_size)

                            else:
                                batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                                lhs = lhs_3d.view(-1, embedding_size)
                                nb_sources = lhs_3d.shape[0]*lhs_3d.shape[1]
                                nb_branches = nb_sources // batch_size

                            rel = rel[batch[0]:batch[1]]
                            rel = rel.view(-1, 1,
                                           embedding_size).repeat(1, nb_branches, 1)
                            rel = rel.view(-1, embedding_size)

                            if intersection_num > 0 and 'disj' in env.graph_type:
                                batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                                rhs = rhs_3d.view(-1, embedding_size)
                                z_scores = self.score_fixed(
                                    rel, lhs, rhs, candidates)

                                z_scores_1d = z_scores.view(-1)
                                if 'disj' in env.graph_type or scores_normalize:
                                    z_scores_1d = torch.sigmoid(z_scores_1d)

                                batch_scores = z_scores_1d if batch_scores is None else objective(
                                    z_scores_1d, batch_scores, t_norm)
                                if ind == indices[-1] and env.graph_type == '2_2_disj_seq':
                                    if inst_ind == 0:
                                        seq_scores = 1 * batch_scores
                                    else:
                                        seq_scores = self.t_norm(seq_scores, batch_scores, t_norm)
                                        #seq_scores = seq_scores * batch_scores
                                    if ind == 5:
                                        batch_scores = seq_scores
                                    else:
                                        batch_scores = None
                                continue

                            if f"rhs_{ind}" not in candidate_cache or last_step or last_step_seq:

                                z_scores, rhs_3d = self.get_best_candidates(
                                    rel, lhs, None, candidates, last_step, env if explain else None)


                                # [B * Candidates^K] or [B, S-1, N]
                                z_scores_1d = z_scores.view(-1)

                                if 'disj' in env.graph_type or scores_normalize:
                                    z_scores_1d = torch.sigmoid(z_scores_1d)
                            

                                if not last_step:
                                    # 25
                                    nb_sources = rhs_3d.shape[0] * \
                                        rhs_3d.shape[1]
                                    # 25
                                    nb_branches = nb_sources // batch_size

                                if not last_step:
                                    # if len(chain_instructions) == 6 and (ind == indices[-1]):
                                    #     batch_scores = z_scores_1d if batch_scores is None else objective(
                                    #         z_scores_1d.view(-1, 1).repeat(1, candidates).view(-1), batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
                                        
                                        # if ind == 5:
                                        #     batch_scores = objective(batch_scores, candidate_cache['rhs_2'][0], t_norm)
                                        # if ind == 8:
                                        #     batch_scores = objective(batch_scores, candidate_cache['rhs_5'][0], t_norm)

                                    # else:
                                        batch_scores = z_scores_1d if batch_scores is None else objective(
                                            z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
                                else:
                                    if ind == indices[0]:
                                        nb_ent = rhs_3d.shape[1]
                                    else:
                                        nb_ent = 1

                                    batch_scores = z_scores_1d if batch_scores is None else objective(
                                        z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)
                                    nb_ent = rhs_3d.shape[1]                        

                                candidate_cache[f"rhs_{ind}"] = (
                                    batch_scores, rhs_3d)

                                if ind == indices[0] and 'disj' in env.graph_type:
                                    count = len(indices)-1
                                    iterator = 1
                                    while count > 0:
                                        candidate_cache[f"rhs_{indices[intersection_num+iterator]}"] = (
                                            batch_scores, rhs_3d)
                                        iterator += 1
                                        count -= 1

                                if ind == indices[-1] and len(chain_instructions) != 6:
                                    # in the seq mode, we don't want rhs candidates to be next lhss
                                    candidate_cache[f"lhs_{ind+1}"] = (
                                        batch_scores, rhs_3d)
                                if ind == indices[-1] and len(chain_instructions) == 6 and chain_instructions[0] == 'intersect_0_1':
                                    candidate_cache[f"lhs_{ind+1}"] = (batch_scores, rhs_3d)
                                
                                if ind == indices[-1] and len(chain_instructions) == 6 and chain_instructions[0] == 'hop_0_1':

                                    if inst_ind == 1:
                                        seq_scores = 1 * batch_scores
                                    else:
                                        #seq_scores = seq_scores * batch_scores
                                        seq_scores = self.t_norm(seq_scores, batch_scores, t_norm)
                                    if ind == 8:
                                        batch_scores = seq_scores
                            else:
                                

                                batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]

                                if f"rhs_{ind+1}" not in candidate_cache.keys():
                                    candidate_cache[f"rhs_{ind+1}"] = (batch_scores, rhs_3d)
                                        
                                last_hop = True

                                if len(chain_instructions) == 6 and (ind == indices[-2]):
                                    last_step_seq = True

                                del lhs, rel

                                continue

                            # if ind == 2:
                            #     print(candidate_cache['rhs_2'][0].shape)
                            #     sys.exit()
                            del lhs, rel, rhs, rhs_3d, z_scores_1d, z_scores

            # TODO: remove this
            #batch_scores = seq_scores
            if batch_scores is not None:
                # [B * entites * S ]
                # S ==  K**(V-1)
                if len(chain_instructions) == 6 or env.graph_type == '2_2_disj_seq':
                    #batch_scores = seq_scores[0] * seq_scores[1] * seq_scores[2]
                    batch_scores = seq_scores

                scores_2d = batch_scores.view(batch_size, -1, nb_ent)
                # print(scores_2d.shape)
                # [1,candidates, nb_ent]
                # res is the max score for each entity among the candidates
                res, _ = torch.max(scores_2d, dim=1)
                scores = res if scores is None else torch.cat([scores, res])

                del batch_scores, scores_2d, res, candidate_cache

            else:
                assert False, "Batch Scores are empty: an error went uncaught."
            res = scores


        # res has the score of each entity for each query
        return res


class SimplE(KBCModel):
    def __init__(
        self, sizes: Tuple[int, int, int], rank: int,
        init_size: float = 1e-3
    ):
        super(SimplE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.init_size = init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        for_prod = torch.sum(lhs[0] * rel[0] * rhs[1], 1, keepdim=True)
        inv_prod = torch.sum(lhs[1] * rel[1] * rhs[0], 1, keepdim=True)
        return torch.clamp((for_prod + inv_prod)/2, min=-20, max=20)

    def entity_embeddings(self, indices: Tensor):
        return self.embeddings[0](indices)
    def rel_embeddings(self, indices: Tensor):
        return self.embeddings[1](indices)

    def score_fixed(self, rel: Tensor, arg1: Tensor, arg2: Tensor,
                    *args, **kwargs) -> Tensor:
        rel_f, rel_inv = rel[:, :self.rank], rel[:, self.rank:]
        arg1_f, arg1_inv = arg1[:, :self.rank], arg1[:, self.rank:]
        arg2_f, arg2_inv = arg2[:, :self.rank], arg2[:, self.rank:]

        score_f = torch.sum(arg1_f * rel_f * arg2_inv, 1, keepdim=True)
        score_inv = torch.sum(arg1_inv * rel_inv * arg2_f, 1, keepdim=True)
        res = torch.clamp((score_f + score_inv)/2, min=-20, max=20)
        del rel_f, rel_inv, arg1_f, arg1_inv, arg2_f, arg2_inv, score_f, score_inv
        return res


# TODO: Check if this is correct


    def candidates_score(self,
                         rel: Tensor,
                         arg1: Optional[Tensor],
                         arg2: Optional[Tensor],
                         *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        emb = self.embeddings[0].weight
        rel_f, rel_inv = rel[:, :self.rank], rel[:, self.rank:]
        emb_f, emb_inv = emb[:, :self.rank], emb[:, self.rank:]

        score_sp = score_po = None
        # calculates score of <arg1, rel, ent> triples for all entities
        if arg1 is not None:
            arg1_f, arg1_inv = arg1[:, :self.rank], arg1[:, self.rank:]
            score_f_sp = (rel_f * arg1_f) @ emb_inv.t()
            score_inv_sp = (rel_inv * arg1_inv) @ emb_f.t()
            score_sp = torch.clamp(
                (score_f_sp + score_inv_sp)/2, min=-20, max=20)
        # calculates scores of <ent, rel, arg2> triples for all entities
        if arg2 is not None:
            arg2_f, arg2_inv = arg2[:, :self.rank], arg2[:, self.rank:]
            score_f_po = (rel_f * arg2_f) @ emb_inv.t()
            score_inv_po = (rel_inv * arg2_inv) @ emb_f.t()
            score_po = torch.clamp(
                (score_f_po + score_inv_po)/2, min=-20, max=20)

        return score_sp, score_po
# returns the score of the given triples and regularization losses for head, rel, and tail

    def score_emb(self, lhs_emb, rel_emb, rhs_emb):
        lhs = lhs_emb[:, :self.rank], lhs_emb[:, self.rank:]
        rel = rel_emb[:, :self.rank], rel_emb[:, self.rank:]
        rhs = rhs_emb[:, :self.rank], rhs_emb[:, self.rank:]
        for_prod = torch.sum(lhs[0] * rel[0] * rhs[1], 1, keepdim=True)
        inv_prod = torch.sum(lhs[1] * rel[1] * rhs[0], 1, keepdim=True)
        score = torch.clamp((for_prod + inv_prod)/2, min=-20, max=20)

        return score, (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        # to_score is all entities in tail
        for_prod = (lhs[0] * rel[0]) @ to_score[1].transpose(0, 1)
        inv_prod = (lhs[1] * rel[1]) @ to_score[0].transpose(0, 1)
        return torch.clamp((for_prod + inv_prod)/2, min=-20, max=20), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
    
    def get_rank_distance(self, dist_mean_f, dist_mean_inv, candidate_ent):
        #candidate_emb = self.entity_embeddings(torch.tensor(candidate_ent))
        #candidate = candidate_emb[:self.rank], candidate_emb[self.rank:]

        to_score = self.embeddings[0].weight
        for_score = dist_mean_f @ to_score[0].transpose(0,1)
        inv_score = dist_mean_inv @ to_score[1].transpose(0,1)
        score = for_score + inv_score
        score_candidate = score[0][candidate_ent]
        rcount = (score[0] > score_candidate).sum().item() + 1
        return rcount


    def forward_emb(self, lhs, rel):

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        #print(rel[0]* self.embeddings[0].weight[1][self.rank:])
        #sys.exit()
        
        to_score_initial = self.embeddings[0].weight

        to_score_for = rel[1] * to_score_initial[:, :self.rank]
        to_score_inv = rel[0] * to_score_initial[:, self.rank:]
        to_score = to_score_for , to_score_inv
        #to_score = F.normalize(to_score_for, p=2.0, dim = 1), F.normalize(to_score_inv, p=2.0, dim = 1)
        # to_score is all entities in tail
        #for_prod = (lhs[0] * rel[0]) @ to_score[1].transpose(0, 1)
        #inv_prod = (lhs[1] * rel[1]) @ to_score[0].transpose(0, 1)
        for_prod = lhs[0] @ to_score[1].transpose(0,1)
        inv_prod = lhs[1] @ to_score[0].transpose(0,1)
        #return torch.clamp((for_prod + inv_prod)/2, min=-20, max=20)
        return (for_prod + inv_prod)/2
    
    # def forward_emb_norel(self, lhs):
    #     lhs = lhs[:, :self.rank], lhs[:, self.rank:]
    #     to_score = self.embeddings[0].weight
    #     to_score = to_score[:, :self.rank], to_score[:, self.rank:]
    #     for_prod = (lhs[0]) @ to_score[1].transpose(0, 1)
    #     inv_prod = (lhs[1]) @ to_score[0].transpose(0, 1)
    #     return torch.clamp((for_prod + inv_prod)/2, min=-20, max=20)
    
    def backward_emb(self, rhs, rel):
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        to_score = self.embeddings[0].weight
        #to_score_initial= self.embeddings[0].weight
        #to_score_for = rel[0] * to_score_initial[:, :self.rank]
        #to_score_inv = rel[1] * to_score_initial[:, self.rank:]
        #to_score = F.normalize(to_score_for, p=2.0, dim = 1), F.normalize(to_score_inv, p=2.0, dim = 1)
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        # to_score is all entities in head
        for_prod = (rhs[1] * rel[0]) @ to_score[0].transpose(0, 1)
        inv_prod = (rhs[0] * rel[1]) @ to_score[1].transpose(0, 1)
        #for_prod = (rhs[1]) @ to_score[0].transpose(0, 1)
        #inv_prod = (rhs[0]) @ to_score[1].transpose(0, 1)
        #return torch.clamp((for_prod + inv_prod)/2, min=-20, max=20)
        return (for_prod + inv_prod)/2

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries_separated(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        

        return (lhs, rel)

    def get_full_embeddigns(self, queries: torch.Tensor):

        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.embeddings[0](queries[:, 0])
        else:
            lhs = None

        if torch.sum(queries[:, 1]).item() > 0:

            rel = self.embeddings[1](queries[:, 1])
        else:
            rel = None

        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.embeddings[0](queries[:, 2])
        else:
            rhs = None

        return (lhs, rel, rhs)

    def get_queries(self, queries: torch.Tensor, side: str = 'rhs'):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        if side == 'rhs':

            return torch.cat([
                0.5 * lhs[1] * rel[1],
                0.5 * lhs[0] * rel[0]
            ], 1)
        elif side == 'lhs':
            return torch.cat([0.5 * lhs[1] * rel[0], 0.5 * lhs[0] * rel[1]], 1)

    def model_type(self):
        return "SimplE"


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.init_size = init_size

    def entity_embeddings(self, indices: Tensor):
        return self.embeddings[0](indices)

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        prod = torch.sum(lhs * rel * rhs, 1, keepdim=True)

        return torch.clamp(prod, min=-20, max=20)

    def score_fixed(self, rel: Tensor, arg1: Tensor, arg2: Tensor,
                    *args, **kwargs) -> Tensor:
        score = torch.sum(arg1 * rel * arg2, 1, keepdim=True)
        res = torch.clamp(score, min=-20, max=20)
        del rel, arg1, arg2, score
        return res

    def candidates_score(self,
                         rel: Tensor,
                         arg1: Optional[Tensor],
                         arg2: Optional[Tensor],
                         *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        emb = self.embeddings[0].weight
        score_sp = score_po = None
        # calculates score of <arg1, rel, ent> triples for all entities
        if arg1 is not None:

            score_f_sp = (rel * arg1) @ emb.t()

            score_sp = torch.clamp((score_f_sp), min=-20, max=20)
        # calculates scores of <ent, rel, arg2> triples for all entities
        if arg2 is not None:

            score_f_po = (rel * arg2) @ emb.t()

            score_po = torch.clamp((score_f_po), min=-20, max=20)

        return score_sp, score_po

    def score_emb(self, lhs_emb, rel_emb, rhs_emb):

        prod = torch.sum(lhs_emb * rel_emb * rhs_emb, 1, keepdim=True)
        score = torch.clamp(prod, min=-20, max=20)

        return score, (
            torch.sqrt(lhs_emb ** 2),
            torch.sqrt(rel_emb ** 2),
            torch.sqrt(rhs_emb ** 2)
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        to_score = self.embeddings[0].weight

        # to_score is all entities in tail
        prod = (lhs * rel) @ to_score.transpose(0, 1)

        return torch.clamp(prod, min=-20, max=20), (
            torch.sqrt(lhs ** 2),
            torch.sqrt(rel ** 2),
            torch.sqrt(rhs ** 2)
        )

    def forward_emb(self, lhs, rel):

        to_score = self.embeddings[0].weight

        # to_score is all entities in tail
        prod = (lhs * rel) @ to_score.transpose(0, 1)
        return torch.clamp(prod, min=-20, max=20)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries_separated(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return (lhs, rel)

    def get_full_embeddigns(self, queries: torch.Tensor):

        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.embeddings[0](queries[:, 0])
        else:
            lhs = None

        if torch.sum(queries[:, 1]).item() > 0:

            rel = self.embeddings[1](queries[:, 1])
        else:
            rel = None

        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.embeddings[0](queries[:, 2])
        else:
            rhs = None

        return (lhs, rel, rhs)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return torch.cat([
            lhs * rel,
            lhs * rel
        ], 1)

    def model_type(self):
        return "CP"


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()

        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.init_size = init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def entity_embeddings(self, indices: Tensor):
        return self.embeddings[0](indices)

    def score_fixed(self, rel: Tensor, arg1: Tensor, arg2: Tensor,
                    *args, **kwargs) -> Tensor:
        # [B, E]
        rel_real, rel_img = rel[:, :self.rank], rel[:, self.rank:]
        arg1_real, arg1_img = arg1[:, :self.rank], arg1[:, self.rank:]
        arg2_real, arg2_img = arg2[:, :self.rank], arg2[:, self.rank:]

        # [B] Tensor
        score1 = torch.sum(rel_real * arg1_real * arg2_real, 1)
        score2 = torch.sum(rel_real * arg1_img * arg2_img, 1)
        score3 = torch.sum(rel_img * arg1_real * arg2_img, 1)
        score4 = torch.sum(rel_img * arg1_img * arg2_real, 1)

        res = score1 + score2 + score3 - score4

        del score1, score2, score3, score4, rel_real, rel_img, arg1_real, arg1_img, arg2_real, arg2_img

        return res

    def candidates_score(self,
                         rel: Tensor,
                         arg1: Optional[Tensor],
                         arg2: Optional[Tensor],
                         *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:

        emb = self.embeddings[0].weight

        rel_real, rel_img = rel[:, :self.rank], rel[:, self.rank:]
        emb_real, emb_img = emb[:, :self.rank], emb[:, self.rank:]

        # [B] Tensor

        score_sp = score_po = None

        if arg1 is not None:
            arg1_real, arg1_img = arg1[:, :self.rank], arg1[:, self.rank:]

            score1_sp = (rel_real * arg1_real) @ emb_real.t()
            score2_sp = (rel_real * arg1_img) @ emb_img.t()
            score3_sp = (rel_img * arg1_real) @ emb_img.t()
            score4_sp = (rel_img * arg1_img) @ emb_real.t()

            score_sp = score1_sp + score2_sp + score3_sp - score4_sp

        if arg2 is not None:
            arg2_real, arg2_img = arg2[:, :self.rank], arg2[:, self.rank:]

            score1_po = (rel_real * arg2_real) @ emb_real.t()
            score2_po = (rel_real * arg2_img) @ emb_img.t()
            score3_po = (rel_img * arg2_img) @ emb_real.t()
            score4_po = (rel_img * arg2_real) @ emb_img.t()

            score_po = score1_po + score2_po + score3_po - score4_po

        return score_sp, score_po

    def score_emb(self, lhs_emb, rel_emb, rhs_emb):
        lhs = lhs_emb[:, :self.rank], lhs_emb[:, self.rank:]
        rel = rel_emb[:, :self.rank], rel_emb[:, self.rank:]
        rhs = rhs_emb[:, :self.rank], rhs_emb[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def forward_emb(self, lhs, rel):
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries_separated(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return (lhs, rel)

    def get_full_embeddigns(self, queries: torch.Tensor):

        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.embeddings[0](queries[:, 0])
        else:
            lhs = None

        if torch.sum(queries[:, 1]).item() > 0:

            rel = self.embeddings[1](queries[:, 1])
        else:
            rel = None

        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.embeddings[0](queries[:, 2])
        else:
            rhs = None

        return (lhs, rel, rhs)

    def get_queries(self, queries: torch.Tensor, side: str = 'rhs'):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def model_type(self):
        return "ComplEx"


class DistMult(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(DistMult, self).__init__()

        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList(
            [nn.Embedding(s, rank, sparse=True) for s in sizes[:2]])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.init_size = init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def entity_embeddings(self, indices: Tensor):
        return self.embeddings[0](indices)

    def score_fixed(self, rel: Tensor, arg1: Tensor, arg2: Tensor,
                    *args, **kwargs) -> Tensor:
        return torch.sum(rel * arg1 * arg2, 1)

    def candidates_score(self,
                         rel: Tensor,
                         arg1: Optional[Tensor],
                         arg2: Optional[Tensor],
                         *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:

        emb = self.embeddings[0].weight

        # [B] Tensor

        score_sp = score_po = None

        if arg1 is not None:
            score_sp = (rel * arg1) @ emb.t()

        if arg2 is not None:
            score_po = (rel * arg2) @ emb.t()

        return score_sp, score_po

    def score_emb(self, lhs_emb, rel_emb, rhs_emb):
        return (torch.sum(lhs_emb * rel_emb * rhs_emb[0], 1, keepdim=True),
                (lhs_emb, rel_emb, rhs_emb))

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        to_score = self.embeddings[0].weight
        return ((lhs * rel) @ to_score.transpose(0, 1),
                (lhs, rel, rhs))

    def forward_emb(self, lhs, rel):
        to_score = self.embeddings[0].weight
        return (lhs * rel) @ to_score.transpose(0, 1)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries_separated(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return (lhs, rel)

    def get_full_embeddigns(self, queries: torch.Tensor):

        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.embeddings[0](queries[:, 0])
        else:
            lhs = None

        if torch.sum(queries[:, 1]).item() > 0:

            rel = self.embeddings[1](queries[:, 1])
        else:
            rel = None

        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.embeddings[0](queries[:, 2])
        else:
            rhs = None

        return (lhs, rel, rhs)

    def get_queries(self, queries: torch.Tensor, side: str = 'rhs'):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return lhs * rel

    def model_type(self):
        return "DistMult"
