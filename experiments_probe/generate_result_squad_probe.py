import pickle
import numpy as np

def generate_probe_result_squad():

    root_foler_path = "data_generated/squad/probe_experiment_2020-05-31_002033/"

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

            query_map.extend(result_dict["query map:"])
            query_ppl.extend(result_dict["query ppl:"])
            target_map.extend(result_dict["target map:"])
            target_ppl.extend(result_dict["target ppl:"])

            print(np.std(result_dict["query map:"]))

        print("="*20)
        print(exp_name)
        print("query map\tquery ppl\ttarget map\ttarget ppl")
        print(np.mean(np.array(query_map)), np.mean(np.array(query_ppl)), np.mean(np.array(target_map)), np.mean(np.array(target_ppl)))


generate_probe_result_squad()

