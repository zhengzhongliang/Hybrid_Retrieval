import utils_dataset_openbook
import utils_probe_openbook


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

