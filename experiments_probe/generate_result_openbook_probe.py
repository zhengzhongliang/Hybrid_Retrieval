import pickle
import numpy as np

def generate_probe_result_openbook():

    root_foler_path = "data_generated/openbook/probe_experiment_2020-05-30_125749/"

    exp_folder_paths = {
        "useqa_embd_gold_label": "query_useqa_embd_gold_result_seed_",
        "rand_embd_gold_label": "query_random_embd_gold_result_seed_",
        "tfidf_embd_gold_label": "query_tfidf_embd_gold_result_seed_",
        "useqa_embd_rand_label": "query_useqa_embd_ques_shuffle_result_seed_"
    }

    for exp_name in exp_folder_paths.keys():
        query_map = []
        query_ppl = []
        target_map = []
        target_ppl = []
        for seed in range(5):
            result_dict_name = root_foler_path+exp_folder_paths[exp_name]+str(seed)+"/best_epoch_result.pickle"

            with open(result_dict_name,"rb") as handle:
                result_dict = pickle.load(handle)

            query_map.extend(result_dict["query map:"].tolist())
            query_ppl.extend(result_dict["query ppl:"].tolist())
            target_map.extend(result_dict["target map:"].tolist())
            target_ppl.extend(result_dict["target ppl:"].tolist())

        print("="*20)
        print(exp_name)
        print("query map\tquery ppl\ttarget map\ttarget ppl")
        print(np.mean(np.array(query_map)), np.mean(np.array(query_ppl)), np.mean(np.array(target_map)), np.mean(np.array(target_ppl)))

generate_probe_result_openbook()

