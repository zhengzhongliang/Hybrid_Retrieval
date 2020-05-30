import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
datasets_folder_path = parent_folder_path+"/datasets/"
generated_data_path = parent_folder_path+"/data_generated/"
bm25_folder = str(Path('.').absolute().parent.parent)+"/IR_BM25/"

sys.path+=[parent_folder_path, datasets_folder_path, generated_data_path, bm25_folder]

import numpy as np
import pickle
from debug_dataloader import LoadRawData

def check_with_bootstrap_resampling(baseline_mrrs, hybrid_mrrs):
    print("check with bootstrap resampling ...")
    print("\tresampling size:", len(baseline_mrrs), " baseline shape:", baseline_mrrs.shape, " control shape:", hybrid_mrrs.shape)
    print("\tbaseline mrr:", np.mean(baseline_mrrs), " control mrr:", np.mean(hybrid_mrrs))

    count=0
    for i in range(10000):
        sampled_indx = np.random.choice(len(baseline_mrrs), len(hybrid_mrrs))

        baseline_mrr_all = np.sum(baseline_mrrs[sampled_indx])
        hybrid_mrr_all = np.sum(hybrid_mrrs[sampled_indx])

        if hybrid_mrr_all>baseline_mrr_all:
            count+=1

        # if i%1000==0:
        #     print("resampling ", i, " ... tfidf score:",tfdif_mrr_all, " bert score:", bert_mrr_all, " hybrid score:", hybrid_mrr_all )

    return count/10000

class OpenbookBootstrap():
    def __init__(self):
        dataset_result = LoadRawData("openbook")

        self.bm25_test_mrrs = dataset_result.result_test_bm25["mrr"]
        self.useqa_test_mrrs = dataset_result.result_test_useqa["mrr"]

        with open(generated_data_path+"/hybrid_classifier_result/openbook_hybrid_result.pickle", "rb") as handle:
            hybrid_models_result = pickle.load(handle)

        self.hybrid_threshold_test_mrrs = hybrid_models_result["hybrid_threshold"]["mrr"]
        self.hybrid_lr_test_mrrs = hybrid_models_result["hybrid_lr"]["mrr"]

    def run_bootstrap(self):

        bs_hybridlr_useqa = check_with_bootstrap_resampling(self.hybrid_lr_test_mrrs, self.useqa_test_mrrs) # should be less than 0.95
        bs_useqa_hybridt = check_with_bootstrap_resampling(self.useqa_test_mrrs, self.hybrid_threshold_test_mrrs)  # should be larger than 0.95
        bs_hybridlr_hybridt = check_with_bootstrap_resampling(self.hybrid_lr_test_mrrs, self.hybrid_threshold_test_mrrs)  # should be less than 0.95

        print(bs_hybridlr_useqa, bs_useqa_hybridt, bs_hybridlr_hybridt)
        # result = 0.87, 0.53, 0.89, which means hyrbid lr is as good as threshold and useqa.

class SquadBootstrap():
    def __init__(self):
        dataset_result = LoadRawData("squad")

        self.bm25_test_mrrs = np.concatenate([dataset_result.result_test_bm25["mrr"] for i in range(5)])
        self.useqa_test_mrrs = np.concatenate([dataset_result.result_test_useqa["mrr"] for i in range(5)])

        with open(generated_data_path+"/hybrid_classifier_result/squad_hybrid_result.pickle", "rb") as handle:
            hybrid_models_result = pickle.load(handle)

        self.hybrid_threshold_test_mrrs = np.concatenate([single_seed["hybrid_threshold"]["mrr"] for single_seed in hybrid_models_result])
        self.hybrid_lr_test_mrrs =np.concatenate([single_seed["hybrid_lr"]["mrr"] for single_seed in hybrid_models_result])

    def run_bootstrap(self):
        bs_bm25_hybrid = check_with_bootstrap_resampling(self.bm25_test_mrrs, self.hybrid_lr_test_mrrs)  # should be less than 0.95

        print(bs_bm25_hybrid)

class NqBootstrap():
    def __init__(self):
        dataset_result = LoadRawData("nq")

        self.bm25_test_mrrs_raw = dataset_result.result_test_bm25
        #self.useqa_test_mrrs_raw = np.concatenate([dataset_result.result_test_useqa["mrr"] for i in range(5)])

        with open(generated_data_path+"/hybrid_classifier_result/nq_hybrid_result.pickle", "rb") as handle:
            hybrid_models_result = pickle.load(handle)

        all_test_indices = [single_seed["test_index_in_all_list"] for single_seed in hybrid_models_result]

        self.bm25_test_mrrs = np.concatenate([self.bm25_test_mrrs_raw["mrr"][test_split] for test_split in all_test_indices])

        self.hybrid_threshold_test_mrrs = np.concatenate([single_seed["hybrid_threshold"]["mrr"] for single_seed in hybrid_models_result])
        self.hybrid_lr_test_mrrs =np.concatenate([single_seed["hybrid_lr"]["mrr"] for single_seed in hybrid_models_result])

    def run_bootstrap(self):
        bs_bm25_hybrid = check_with_bootstrap_resampling(self.bm25_test_mrrs, self.hybrid_lr_test_mrrs)  # should be less than 0.95

        print(bs_bm25_hybrid)

# openbookBS = OpenbookBootstrap()
# openbookBS.run_bootstrap()

# squadBS = SquadBootstrap()
# squadBS.run_bootstrap()

nqBS = NqBootstrap()
nqBS.run_bootstrap()