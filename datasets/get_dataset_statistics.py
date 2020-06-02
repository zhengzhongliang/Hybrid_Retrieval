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


def get_squad_statistics():
    with open(generated_data_path+"squad_useqa/squad_retrieval_data.pickle", "rb") as handle:
        squad_pickle = pickle.load(handle)

    train_list = squad_pickle["train_list"]
    dev_list = squad_pickle["dev_list"]

    sent_list=  squad_pickle["sent_list"]
    doc_list = squad_pickle["doc_list"]
    resp_list =squad_pickle["resp_list"]

    print("squad n train:", len(train_list))

    token_len_list = []
    for question in train_list+squad_pickle["dev_list"]:
        token_len_list.append(len(question["question"].split(" ")))

    token_len_doc_list = []
    for resp in resp_list:
        doc = sent_list[int(resp[0])]+doc_list[int(resp[1])]
        token_len_doc_list.append(len(doc.split(" ")))

    print("avg query len:", sum(token_len_list)/len(token_len_list), "avg doc len:", sum(token_len_doc_list)/len(token_len_doc_list))

    return 0

def get_squad_example():
    with open(generated_data_path+"squad_useqa/squad_retrieval_data.pickle", "rb") as handle:
        squad_pickle = pickle.load(handle)

    train_list = squad_pickle["train_list"]
    dev_list = squad_pickle["dev_list"]

    sent_list = squad_pickle["sent_list"]
    doc_list = squad_pickle["doc_list"]
    resp_list = squad_pickle["resp_list"]

    for question in train_list:
        print("question:", question["question"])
        print("answer sent:", sent_list[int(resp_list[question["response"]][0])])
        print("answer doc:", doc_list[int(resp_list[question["response"]][1])])
        input("="*20)

def get_nq_example():
    with open(generated_data_path+"nq_retrieval_raw/nq_retrieval_data.pickle", "rb") as handle:
        squad_pickle = pickle.load(handle)

    train_list = squad_pickle["train_list"]

    sent_list=  squad_pickle["sent_list"]
    doc_list = squad_pickle["doc_list"]
    resp_list =squad_pickle["resp_list"]

    for question in train_list:
        print("question:", question["question"])
        print("answer sent:", sent_list[int(resp_list[question["response"]][0])])
        print("answer doc:", doc_list[int(resp_list[question["response"]][1])])
        input("="*20)



def get_nq_statistics():
    with open(generated_data_path+"nq_retrieval_raw/nq_retrieval_data.pickle", "rb") as handle:
        squad_pickle = pickle.load(handle)

    train_list = squad_pickle["train_list"]

    sent_list=  squad_pickle["sent_list"]
    doc_list = squad_pickle["doc_list"]
    resp_list =squad_pickle["resp_list"]

    print("squad n train:", len(train_list))

    token_len_list = []
    for question in train_list:
        token_len_list.append(len(question["question"].split(" ")))

    token_len_doc_list = []
    for resp in resp_list:
        doc = sent_list[int(resp[0])]+doc_list[int(resp[1])]
        token_len_doc_list.append(len(doc.split(" ")))

    print("avg query len:", sum(token_len_list)/len(token_len_list), "avg doc len:", sum(token_len_doc_list)/len(token_len_doc_list))

    return 0

#get_nq_statistics()


get_nq_example()
