import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
datasets_folder_path = parent_folder_path+"/datasets/"
sys.path+=[parent_folder_path, datasets_folder_path]

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch.optim as optim

import pickle
import numpy as np

import squad_retrieval
import random

# TODO: print essential information.


class BertSQuADRetriever(nn.Module):
    def __init__(self, n_neg_sample, device):
        super(BertSQuADRetriever).__init__()

        self.bert_q = BertModel.from_pretrained('bert-base-uncased')
        self.bert_d = BertModel.from_pretrained('bert-base-uncased')

        self.criterion = torch.nn.CrossEntropyLoss()

        self.n_neg_sample = n_neg_sample
        self.device = device

    def forward_train(self, query_token_ids, query_seg_ids, fact_token_ids, fact_seg_ids):
        query_output_tensor_, _ = self.bert_q(query_token_ids, query_seg_ids)
        fact_output_tensor_, _ = self.bert_d(fact_token_ids, fact_seg_ids)

        batch_size = query_token_ids.size()[0]

        query_output_tensor = query_output_tensor_[-1][:, 0].view(batch_size, 768, 1)
        fact_output_tensor = fact_output_tensor_[-1][:, 0].view(batch_size, self.n_neg_sample+1, 768)  # the middle number should be n_neg_sample+1

        return query_output_tensor, fact_output_tensor

    def forward_eval_query(self, query_token_ids, query_seg_ids):
        query_output_tensor_, _ = self.bert_q(query_token_ids, query_seg_ids)

        batch_size = query_token_ids.size()[0]

        query_output_tensor = query_output_tensor_[-1][:, 0].view(batch_size, 768)

        return query_output_tensor

    def forward_eval_fact(self, fact_token_ids, fact_seg_ids):
        fact_output_tensor_, _ = self.bert_d(fact_token_ids, fact_seg_ids)

        batch_size = fact_token_ids.size()[0]

        fact_output_tensor = fact_output_tensor_[-1][:, 0].view(batch_size, 768)

        return fact_output_tensor

    def train_epoch(self, optimizer, squad_retrieval_train_dataloader):

        total_loss = 0
        for i, batch in enumerate(squad_retrieval_train_dataloader):
            query_output_tensor, fact_output_tensor = self.forward_train(batch["query_token_ids"].to(self.device),
                                                                 batch["query_seg_ids"].to(self.device),
                                                                 batch["fact_token_ids"].to(self.device),
                                                                 batch["fact_seg_ids"].to(self.device))

            scores = torch.matmul(fact_output_tensor, query_output_tensor).squeeze(dim=2)

            label = batch["label_in_distractor"].to(self.device)

            loss = self.criterion(scores, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()

        return total_loss/len(squad_retrieval_train_dataloader)


    def eval_epoch(self, squad_retrieval_dev_dataloader, squad_retrieval_test_dataloader, squad_retrieval_eval_fact_dataloader):

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

        # First step: compute all fact embeddings
        fact_embds = []
        for i, batch in enumerate(squad_retrieval_eval_fact_dataloader):
            fact_embds_batch = self.forward_eval_fact(batch["fact_token_ids"].to(self.device), batch["fact_seg_ids"].to(self.device))
            fact_embds.append(fact_embds_batch.detach().cpu().numpy())
        fact_embds = np.transpose(np.concatenate(fact_embds, axis = 0))  # transpose the embedding for better multiplication.

        # Second step: compute the query embedding for each batch. At the same time return the needed results.
        dev_results_dict = {"mrr": [], "gold_fact_index": [], "gold_fact_ranking": [], "gold_fact_score": [], "top_64_facts":[], "top_64_scores":[]}
        for i, batch in enumerate(squad_retrieval_dev_dataloader):
            query_embds_batch = self.forward_eval_query(batch["query_token_ids"].to(self.device),batch["query_seg_ids"].to(self.device))
            query_embds_batch = query_embds_batch.detach().cpu().numpy()

            self._fill_results_dict(batch, query_embds_batch, fact_embds, dev_results_dict)

        # Third step: compute the query embedding for each batch, then store the result to a dict.
        test_results_dict = {"mrr": [], "gold_fact_index": [], "gold_fact_ranking": [], "gold_fact_score": [], "top_64_facts":[], "top_64_scores":[]}
        for i, batch in enumerate(squad_retrieval_test_dataloader):
            query_embds_batch = self.forward_eval_query(batch["query_token_ids"].to(self.device), batch["query_seg_ids"].to(self.device))
            query_embds_batch = query_embds_batch.detach().cpu().numpy()

            self._fill_results_dict(batch, query_embds_batch, fact_embds, test_results_dict)

        return dev_results_dict, test_results_dict

    def _fill_results_dict(self, batch, query_embds_batch, fact_embds, result_dict):
        # Things to return:
        gold_facts_indices = batch["response"].numpy().reshape((len(batch), 1))  # size: n_query * 1

        batch_scores = np.matmul(query_embds_batch, fact_embds)   # size: n_query * n_facts  TODO: this needs to be after softmaxed.
        sorted_scores = np.flip(np.sort(batch_scores, axis=1), axis = 1)
        sorted_facts = np.flip(np.argsort(batch_scores, axis=1), axis= 1)

        gold_fact_rankings_indices_row,  gold_fact_rankings= np.where(sorted_facts==gold_facts_indices)

        result_dict["gold_fact_index"].extend(gold_facts_indices.flatten().tolist())
        result_dict["gold_fact_ranking"].extend(gold_fact_rankings.tolist())   # get the gold fact ranking of each query
        result_dict["gold_fact_score"].extend(sorted_scores[ gold_fact_rankings_indices_row,  gold_fact_rankings].tolist())
        result_dict["mrr"].extend(1/(1+gold_fact_rankings).tolist())

        result_dict["top_64_facts"].extend(sorted_facts[:,:64].tolist())
        result_dict["top_64_scores"].extend(sorted_scores[:,:64].tolist())

        return 0

def train_and_eval_model(args, saved_pickle_path = parent_folder_path + "/data_generated/squad_retrieval_data_seed_0_dev_2000.pickle"):
    N_EPOCH = args.n_epoch
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.n_worker
    N_NEG_FACT = args.n_neg_sample
    DEVICE = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    # Instantiate BERT retriever, optimizer and tokenizer.
    bert_retriever = BertSQuADRetriever(N_NEG_FACT, DEVICE)
    bert_retriever.to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    optimizer = optim.Adam(bert_retriever.parameters(), lr=0.00001)

    # Load SQuAD dataset and dataloader.
    squad_retrieval_data = squad_retrieval.convert_squad_to_retrieval(tokenizer, random_seed = args.seed, num_dev = args.num_dev)

    squad_retrieval_train_dataset = squad_retrieval.SQuADRetrievalDatasetTrain(instance_list=squad_retrieval_data["train_list"],
                                                               sent_list=squad_retrieval_data["sent_list"],
                                                               doc_list=squad_retrieval_data["doc_list"],
                                                               resp_list=squad_retrieval_data["resp_list"],
                                                               tokenizer=tokenizer,
                                                               random_seed=args.seed,
                                                                n_negative_sample = N_NEG_FACT)

    squad_retrieval_train_dataloader = DataLoader(squad_retrieval_train_dataset, batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=NUM_WORKERS, collate_fn=squad_retrieval.PadCollateSQuADTrain())

    squad_retrieval_dev_dataset = squad_retrieval.SQuADRetrievalDatasetEvalQuery(instance_list=squad_retrieval_data["dev_list"],
                                                                 sent_list=squad_retrieval_data["sent_list"],
                                                                 doc_list=squad_retrieval_data["doc_list"],
                                                                 resp_list=squad_retrieval_data["resp_list"],
                                                                 tokenizer=tokenizer)

    squad_retrieval_dev_dataloader = DataLoader(squad_retrieval_dev_dataset, batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=NUM_WORKERS, collate_fn=squad_retrieval.PadCollateSQuADEvalQuery())

    squad_retrieval_test_dataset = squad_retrieval.SQuADRetrievalDatasetEvalQuery(instance_list=squad_retrieval_data["test_list"],
                                                                  sent_list=squad_retrieval_data["sent_list"],
                                                                  doc_list=squad_retrieval_data["doc_list"],
                                                                  resp_list=squad_retrieval_data["resp_list"],
                                                                  tokenizer=tokenizer)

    squad_retrieval_test_dataloader = DataLoader(squad_retrieval_test_dataset, batch_size=BATCH_SIZE,
                                                 shuffle=False, num_workers=NUM_WORKERS, collate_fn=squad_retrieval.PadCollateSQuADEvalQuery())

    squad_retrieval_eval_fact_dataset = squad_retrieval.SQuADRetrievalDatasetEvalFact(instance_list=squad_retrieval_data["resp_list"],
                                                                      sent_list=squad_retrieval_data["sent_list"],
                                                                      doc_list=squad_retrieval_data["doc_list"],
                                                                      resp_list=squad_retrieval_data["resp_list"],
                                                                      tokenizer=tokenizer)

    squad_retrieval_eval_fact_dataloader = DataLoader(squad_retrieval_eval_fact_dataset, batch_size=BATCH_SIZE,
                                                      shuffle=False, num_workers=NUM_WORKERS,
                                                      collate_fn=squad_retrieval.PadCollateSQuADEvalFact())

    # TODO: save foler path. If no folder is found, make directory.

    # Start evaluation.
    best_mrr = 0
    main_result_array = np.zeros((N_EPOCH, 3))
    for epoch in range(N_EPOCH):
        train_loss = bert_retriever.train_epoch(optimizer, squad_retrieval_train_dataloader)
        dev_result_dict, test_result_dict = bert_retriever.eval_epoch(squad_retrieval_dev_dataloader, squad_retrieval_test_dataloader, squad_retrieval_eval_fact_dataloader)

        dev_mrr = sum(dev_result_dict["mrr"])/len(dev_result_dict["mrr"])
        test_mrr = sum(test_result_dict["mrr"])/len(test_result_dict["mrr"])

        main_result_array[epoch,:] = [train_loss, dev_mrr, test_mrr]

        if dev_mrr > best_mrr:

            torch.save(bert_retriever, "data_generated/saved_bert_retriever_seed_"+str(args.seed))  # TODO: fix the folder path, and save the dev and test dict

            with open("data_generated/dev_dict_"+str(args.seed)+".pickle", "wb") as handle:
                pickle.dump(dev_result_dict, handle)

            with open("data_generated/test_dict_"+str(args.seed)+".pickle", "wb") as handle:
                pickle.dump(test_result_dict, handle)

    # TODO: save the main result array
    np.save("data_generated/main_result_"+str(args.seed)+".npy", main_result_array)

    return 0

def main():

    # Things need to be parsed:
    # device, batch_size, n_epoch, num_workers, n_neg_sample,
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--n_epoch", type=int, default=4)
    parser.add_argument("--n_worker", type=int, default=3)
    parser.add_argument("--n_neg_sample", type=int, default=5)
    parser.add_argument("--num_sample", type=int, default=2000)

    # parse the input arguments
    args = parser.parse_args()

    # set the random seeds
    torch.manual_seed(args.seed)  # set pytorch seed
    random.seed(args.seed)     # set python seed.
    # #This python random library is used in two places: one is constructing the raw dataset, the other is when constructing train data.
    np.random.seed(args.seed)   # set numpy seed

    train_and_eval_model(args)


    return 0

main()






