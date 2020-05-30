import pickle
import numpy as np

def load_raw_squad_data(saved_squad_file_pcikle = "data_raw/squad_retrieval_data.pickle"):
    with (saved_squad_file_pcikle, "rb") as handle:
        squad_raw_data = pickle.load(handle)

    return squad_raw_data

def load_squad_query_embeddings(saved_squad_embds_train = "data_raw/ques_train_embds.npy",
                                saved_squad_embds_dev = "data_raw/ques_dev_embds.npy"):
    train_embds = np.load(saved_squad_embds_train)

    dev_embds = np.load(saved_squad_embds_dev)

    return train_embds, dev_embds

