import utils_dataset_openbook
import utils_probe_openbook

import utils_dataset_squad
import utils_probe_squad

import time


def check_openbook_data():
    useqa_embds_paths = {
        "train":"data_generated/openbook/openbook_ques_train_embds.npy",
        "dev":"data_generated/openbook/openbook_ques_dev_embds.npy"
    }
    saved_data_folder = ""

    train_list, dev_list, test_list, sci_kb = utils_dataset_openbook.construct_retrieval_dataset_openbook()
    vocab_dict, tfidf_vectorizer = utils_probe_openbook.get_vocabulary(train_list, sci_kb, "openbook_vocab_dict.pickle", "openbook_tfidf_vectorizer.pickle")
    #instances_all_seeds = utils_probe_openbook.get_probe_dataset(train_list, dev_list, sci_kb, useqa_embds_paths, vocab_dict, tfidf_vectorizer, saved_data_folder, "openbook_probe.pickle")

    print(vocab_dict)

def check_squad_data():
    useqa_embds_paths = {
        "train":"data_generated/squad/squad_ques_train_embds.npy",
        "dev":"data_generated/squad/squad_ques_dev_embds.npy"
    }
    saved_data_folder = ""

    train_list, dev_list, kb = utils_dataset_squad.load_squad_probe_raw_data()

    start_time = time.time()
    vocab_dict, tfidf_vectorizer = utils_probe_squad.get_vocabulary(train_list, kb, "squad_vocab_dict.pickle", "squad_tfidf_vectorizer.pickle")
    end_time = time.time()

    print(list(vocab_dict.items()))
    input("AAA")
    print(list(tfidf_vectorizer.vocabulary_.items()))

    print("vocab build time:", end_time-start_time)
    print("vocab size:", len(vocab_dict))
    print("tfidf vocab size:", len(tfidf_vectorizer.vocabulary_))

    print("water" in vocab_dict , "water" in tfidf_vectorizer.vocabulary_)
    print("sun" in vocab_dict , "sun" in tfidf_vectorizer.vocabulary_)
    #instances_all_seeds = utils_probe_squad.get_probe_dataset(train_list, dev_list, sci_kb, useqa_embds_paths, vocab_dict, tfidf_vectorizer, saved_data_folder, "openbook_probe.pickle")


check_squad_data()