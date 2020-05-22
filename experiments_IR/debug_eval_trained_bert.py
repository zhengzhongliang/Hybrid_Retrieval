import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
datasets_folder_path = parent_folder_path+"/datasets/"
sys.path+=[parent_folder_path, datasets_folder_path]

import os
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader


from pytorch_pretrained_bert import BertTokenizer, BertModel

import time

import squad_retrieval, openbook_retrieval
import random
import datetime
import os

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

class BertRepRanker(nn.Module):
    def __init__(self, device, scorer="dot"):
        super(BertRepRanker, self).__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.bert_q = BertModel.from_pretrained('bert-base-uncased')
        self.bert_d = BertModel.from_pretrained('bert-base-uncased')

        self.criterion = nn.CrossEntropyLoss(size_average=False)

        self.scorer = scorer
        print('Initializing model: using scorer ', self.scorer)
        if self.scorer == "linear":
            self.linear = nn.Linear(768 * 4, 1)
        elif self.scorer == "MLP":
            self.linear_1 = nn.Linear(768 * 4, 100)
            self.linear_2 = nn.Linear(100, 1)

    def get_loss(self, outputs, target):
        loss = self.criterion(outputs.view(1, len(outputs)), target.view(1))
        _, pred = torch.max(outputs, 0)
        return loss, pred.detach().cpu().numpy()

        # This includes the forward_all_first and forward_al_second. It returns index of top 2 retrieved facts and the loss of the whole process.

    def forward(self, query, documents):
        # This training method is borrowed from this paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf
        if self.scorer == "linear":
            input_concat = torch.cat([query.expand_as(documents), documents, query.expand_as(documents) * documents,
                                      torch.abs(query.expand_as(documents) - documents)], dim=1)
            scores = self.linear(input_concat).squeeze()
        elif self.scorer == "MLP":
            input_concat = torch.cat([query.expand_as(documents), documents, query.expand_as(documents) * documents,
                                      torch.abs(query.expand_as(documents) - documents)], dim=1)
            scores = self.linear_2(F.tanh(self.linear_1(input_concat))).squeeze()
        else:
            scores = torch.sum(query.expand_as(documents) * documents.squeeze(), dim=1)

        return scores

class BertRepRanker(BertRepRanker):
    def __init__(self, device, knowledge_base, scorer="dot", context_shuffle=False):
        super(BertRepRanker, self).__init__(device, scorer)
        self.sci_facts_tensor_list = self.facts_to_tensor(knowledge_base)
        self.context_shuffle = context_shuffle

        # TODO: maybe we should add deep interaction later.
        # elif self.scorer == "deep":

    def query_to_tensor(self, query_list: list) -> torch.tensor:
        # Step 1: add CLS token and SEP token to each sentence
        # question_text = question_text.lower()
        query_text = list(['[CLS]'])

        if len(query_list) > 1:
            context_list = query_list[:-1]

            for single_query in context_list:
                query_text.extend(single_query.split() + ['[SEP]'])

            # Step 2: tokens -> ids, generate segment ids
            tokens_context = self.tokenizer.tokenize(" ".join(query_text))
            ids_context = self.tokenizer.convert_tokens_to_ids(tokens_context)
            seg_ids_context = [0] * len(ids_context)

            tokens_question = self.tokenizer.tokenize(query_list[-1] + " [SEP]")
            ids_question = self.tokenizer.convert_tokens_to_ids(tokens_question)
            seg_ids_question = [1] * len(ids_question)

            # Step 3: Convert to tensor
            CLS_id = ids_context[0]
            all_token_ids = ids_context + ids_question
            all_seg_ids = seg_ids_context + seg_ids_question

        else:
            query_text.extend(query_list[0].split() + ['[SEP]'])
            all_tokens = self.tokenizer.tokenize(" ".join(query_text))
            all_token_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
            all_seg_ids = [0] * len(all_token_ids)

            CLS_id = all_token_ids[0]

        if len(all_token_ids) > 512:
            all_token_ids = all_token_ids[-512:]
            all_seg_ids = all_seg_ids[-512:]
            all_token_ids[0] = CLS_id

        tokens_tensor = torch.tensor([all_token_ids]).to(self.device)
        segments_tensor = torch.tensor([all_seg_ids]).to(self.device)

        return tokens_tensor, segments_tensor

    # This function is to convert all science facts to Glove embedding list, later fed to GRU
    def facts_to_tensor(self, sci_facts) -> list:

        sci_facts_input_list = list([])
        for sci_fact in sci_facts:
            sci_fact = sci_fact  # Convert to lowercase and remove quotations.
            input_dict = {}
            # Step 1: add CLS token and SEP token to each sentence
            fact_text = ["[CLS]"] + sci_fact.split() + [" . [SEP]"]

            # Step 2: tokens -> ids, generate segment ids
            tokens = self.tokenizer.tokenize(" ".join(fact_text))
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            seg_ids = [0] * len(token_ids)

            # Step 3: Convert to tensor
            tokens_tensor = torch.tensor([token_ids]).to(self.device)
            segments_tensor = torch.tensor([seg_ids]).to(self.device)

            input_dict["tokens_tensor"] = tokens_tensor
            input_dict["segments_tensor"] = segments_tensor

            sci_facts_input_list.append(input_dict)

        return sci_facts_input_list

    # Take the Glove embedding tensor of question-choice as input and produce sentence embedding as the output
    def forward_query(self, question_text: list):
        # print(question_text)
        query_input_tensor_tokens, query_input_tensor_segments = self.query_to_tensor(question_text)
        query_output_tensor, _ = self.bert_q(query_input_tensor_tokens, query_input_tensor_segments)

        return query_output_tensor[-1][0, 0]  # Output size: batch_size * embd_size = 1*400

    # Take the Glove embedding tensor of science facts as input and produce snetence embeddings as the output
    def forward_facts(self, sci_facts_input_list):
        sci_facts_output_list = list([])
        for input_dict in sci_facts_input_list:
            sci_fact_output, _ = self.bert_d(input_dict["tokens_tensor"], input_dict["segments_tensor"])
            sci_facts_output_list.append(sci_fact_output[-1][0, 0])

        sci_facts_output_tensor = torch.stack(sci_facts_output_list)

        return sci_facts_output_tensor.squeeze()  # Output size:  batch_size * embd_size = 1367*400

    def save_model(self, save_folder_path, doc_embedding, train_fact_score, dev_fact_score, test_fact_score, epoch):
        # Create a folder if there is not already one.
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        # Save the trained model:
        torch.save(self, save_folder_path+'/savedBertRepRanker_epoch_' + str(epoch))

        # Save the fact embeddings:
        np.save(save_folder_path+'/fact_embeddings_epoch_'+str(epoch)+'.npy', doc_embedding)

        # Save the train scores:
        with open(save_folder_path+'/train_fact_scores_epoch_'+str(epoch)+".pickle", "wb") as handle:
            pickle.dump(train_fact_score, handle)

        # Save the dev scores:
        with open(save_folder_path + '/dev_fact_scores_epoch_' + str(epoch) + ".pickle", "wb") as handle:
            pickle.dump(dev_fact_score, handle)

        # Save the train scores:
        with open(save_folder_path + '/test_fact_scores_epoch_' + str(epoch) + ".pickle", "wb") as handle:
            pickle.dump(test_fact_score, handle)

        return 0

class BertEvalLoader(nn.Module):
    def __init__(self, n_neg_sample, device, batch_size_train, batch_size_eval, old_bert_directory = ""):
        super(BertEvalLoader, self).__init__()

        old_bert = torch.load(old_bert_directory)

        self.bert_q = old_bert.bert_q
        self.bert_d = old_bert.bert_d

        self.n_neg_sample = n_neg_sample
        self.device = device
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval

    def forward_train(self, query_token_ids, query_seg_ids, query_att_mask_ids, fact_token_ids, fact_seg_ids,
                      fact_att_mask_ids):
        query_output_tensor_, _ = self.bert_q(input_ids=query_token_ids, token_type_ids=query_seg_ids,
                                              attention_mask=query_att_mask_ids)
        fact_output_tensor_, _ = self.bert_d(input_ids=fact_token_ids, token_type_ids=fact_seg_ids,
                                             attention_mask=fact_att_mask_ids)

        batch_size = query_token_ids.size()[0]

        query_output_tensor = query_output_tensor_[-1][:, 0].view(batch_size, 768, 1)
        fact_output_tensor = fact_output_tensor_[-1][:, 0].view(batch_size, self.n_neg_sample + 1,
                                                                768)  # the middle number should be n_neg_sample+1

        return query_output_tensor, fact_output_tensor

    def forward_eval_query(self, query_token_ids, query_seg_ids, query_att_mask_ids):
        query_output_tensor_, _ = self.bert_q(input_ids=query_token_ids, token_type_ids=query_seg_ids,
                                              attention_mask=query_att_mask_ids)

        batch_size = query_token_ids.size()[0]

        query_output_tensor = query_output_tensor_[-1][:, 0].view(batch_size, 768)

        return query_output_tensor

    def forward_eval_fact(self, fact_token_ids, fact_seg_ids, fact_att_mask_ids):
        fact_output_tensor_, _ = self.bert_d(input_ids=fact_token_ids, token_type_ids=fact_seg_ids,
                                             attention_mask=fact_att_mask_ids)

        batch_size = fact_token_ids.size()[0]

        fact_output_tensor = fact_output_tensor_[-1][:, 0].view(batch_size, 768)

        return fact_output_tensor

    def eval_epoch(self, retrieval_dev_dataloader, retrieval_test_dataloader, retrieval_eval_fact_dataloader):

        # ref size: 1,000*768 numpy array is about 3 MB.
        # dev query size: 2,000*768 = 6 MB.
        # test query size: 10,000*768 = 30 MB
        # facts size: 100,000*768 = 300 MB

        # score size without cut:
        # dev score size = 2,000 * 100,000 = 830 MB
        # test score size = 10,000 * 100,000 = 4.2 GB

        # What data we need to save for each question (for the best epoch):
        # gold fact index, gold fact ranking, gold fact score
        # top 64 fact scores.

        # What other things we need to save:
        # the best performed model (model with the best test mrr).
        # training loss, dev mrr, test mrr.

        self.eval()

        with torch.no_grad():
            # First step: compute all fact embeddings
            fact_embds = []
            for i, batch in enumerate(retrieval_eval_fact_dataloader):
                fact_embds_batch = self.forward_eval_fact(batch["fact_token_ids"].to(self.device), batch["fact_seg_ids"].to(self.device), batch["fact_att_mask_ids"].to(self.device))
                fact_embds.append(fact_embds_batch.detach().cpu().numpy())
                if (i+1)%100==0:
                    print("\tget fact "+str(i+1))
            fact_embds = np.transpose(np.concatenate(fact_embds, axis = 0))  # transpose the embedding for better multiplication.
            #fact_embds = np.random.rand(768, 102003)

            # Second step: compute the query embedding for each batch. At the same time return the needed results.
            dev_results_dict = {"mrr": [], "gold_fact_index": [], "gold_fact_ranking": [], "gold_fact_score": [], "top_64_facts":[], "top_64_scores":[]}
            for i, batch in enumerate(retrieval_dev_dataloader):
                query_embds_batch = self.forward_eval_query(batch["query_token_ids"].to(self.device),batch["query_seg_ids"].to(self.device),batch["query_att_mask_ids"].to(self.device))
                query_embds_batch = query_embds_batch.detach().cpu().numpy()

                self._fill_results_dict(batch, query_embds_batch, fact_embds, dev_results_dict)

                if (i+1)%100==0:
                    print("\tget dev query "+str(i+1))

            # Third step: compute the query embedding for each batch, then store the result to a dict.
            test_results_dict = {"mrr": [], "gold_fact_index": [], "gold_fact_ranking": [], "gold_fact_score": [], "top_64_facts":[], "top_64_scores":[]}
            for i, batch in enumerate(retrieval_test_dataloader):
                query_embds_batch = self.forward_eval_query(batch["query_token_ids"].to(self.device), batch["query_seg_ids"].to(self.device),batch["query_att_mask_ids"].to(self.device))
                query_embds_batch = query_embds_batch.detach().cpu().numpy()

                self._fill_results_dict(batch, query_embds_batch, fact_embds, test_results_dict)

                if (i+1)%100==0:
                    print("\tget test query "+str(i+1))

        return dev_results_dict, test_results_dict

    def _fill_results_dict(self, batch, query_embds_batch, fact_embds, result_dict):
        # Things to return:
        batch_size = len(query_embds_batch)

        gold_facts_indices = batch["response"].numpy().reshape((batch_size, 1))  # size: n_query * 1

        batch_scores = softmax(np.matmul(query_embds_batch, fact_embds))   # size: n_query * n_facts
        sorted_scores = np.flip(np.sort(batch_scores, axis=1), axis = 1)
        sorted_facts = np.flip(np.argsort(batch_scores, axis=1), axis= 1)

        gold_fact_rankings_indices_row,  gold_fact_rankings= np.where(sorted_facts==gold_facts_indices)

        result_dict["gold_fact_index"].extend(gold_facts_indices.flatten().tolist())
        result_dict["gold_fact_ranking"].extend(gold_fact_rankings.tolist())   # get the gold fact ranking of each query
        result_dict["gold_fact_score"].extend(sorted_scores[ gold_fact_rankings_indices_row,  gold_fact_rankings].tolist())
        result_dict["mrr"].extend((1/(1+gold_fact_rankings)).tolist())

        result_dict["top_64_facts"].extend(sorted_facts[:,:64].tolist())
        result_dict["top_64_scores"].extend(sorted_scores[:,:64].tolist())

        return 0

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    BATCH_SIZE_EVAL = 4
    NUM_WORKERS = 3

    train_list, dev_list, test_list, kb = openbook_retrieval.construct_retrieval_dataset_openbook(
        num_neg_sample=4, random_seed=0)

    openbook_retrieval_dev_dataset = openbook_retrieval.OpenbookRetrievalDatasetEvalQuery(
        instance_list=dev_list,
        tokenizer=tokenizer)

    retrieval_dev_dataloader = DataLoader(openbook_retrieval_dev_dataset, batch_size=BATCH_SIZE_EVAL,
                                          shuffle=False, num_workers=NUM_WORKERS,
                                          collate_fn=openbook_retrieval.PadCollateOpenbookEvalQuery())

    openbook_retrieval_test_dataset = openbook_retrieval.OpenbookRetrievalDatasetEvalQuery(
        instance_list=test_list,
        tokenizer=tokenizer)

    retrieval_test_dataloader = DataLoader(openbook_retrieval_test_dataset, batch_size=BATCH_SIZE_EVAL,
                                           shuffle=False, num_workers=NUM_WORKERS,
                                           collate_fn=openbook_retrieval.PadCollateOpenbookEvalQuery())

    openbook_retrieval_eval_fact_dataset = openbook_retrieval.OpenbookRetrievalDatasetEvalFact(
        kb=kb,
        tokenizer=tokenizer)

    retrieval_eval_fact_dataloader = DataLoader(openbook_retrieval_eval_fact_dataset, batch_size=BATCH_SIZE_EVAL,
                                                shuffle=False, num_workers=NUM_WORKERS,
                                                collate_fn=openbook_retrieval.PadCollateOpenbookEvalFact())

    device = torch.device("cuda:0")

    bertEvalLoader = BertEvalLoader(n_neg_sample=10,
                                    device = device,
                                    batch_size_train=1,
                                    batch_size_eval=BATCH_SIZE_EVAL,
                                    old_bert_directory="/home/zhengzhongliang/CLU_Projects/2019_QA/Learn_for_association/experiments_acl2020_trial2/saved_models/bert_openbook_retrieval_seed_0_2019-12-03_0547/savedBertRepRanker_epoch_0")
    bertEvalLoader.to(device)

    dev_result_dict, test_result_dict = bertEvalLoader.eval_epoch(retrieval_dev_dataloader, retrieval_test_dataloader,
                                                                  retrieval_eval_fact_dataloader)

    dev_mrr = sum(dev_result_dict["mrr"]) / len(dev_result_dict["mrr"])
    test_mrr = sum(test_result_dict["mrr"]) / len(test_result_dict["mrr"])

    print("dev_mrr:", dev_mrr)
    print("test_mrr:", test_mrr)

main()