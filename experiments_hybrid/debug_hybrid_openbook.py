from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel # used for compute cosine similarity for sparse matrix
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import openbook_retrieval_utils
import numpy as np
import os
import pickle

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower())]

def eval_tfidf(instances_list, tfidf_vectorizer, doc_matrix, kb,  saved_file_name):


    correct_count = 0
    justification_hit_ratio = list([])
    mrr = 0
    list_to_save = list([])
    count_top10 = 0
    for i, instance in enumerate(instances_list):
        query = instance["query"]
        query_matrix = tfidf_vectorizer.transform(query)

        cosine_similarities = linear_kernel(query_matrix, doc_matrix).flatten()
        rankings = list(reversed(np.argsort(cosine_similarities).tolist()))  # rankings of facts, from most relevant

        mrr+=1/(1+rankings.index(instance["documents"][0]))

        list_to_save.append({"id":instance["id"], "mrr": 1/(1+rankings.index(instance["documents"][0])), "top_score":np.max(cosine_similarities)})


    return list_to_save

def load_bert_scores(file_path):
    with open() as handle:


def main():
    train_list, dev_list, test_list, sci_kb = openbook_retrieval_utils.construct_retrieval_dataset_openbook()

    stop_words_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                       "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                       "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
                       "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                       "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
                       "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                       "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
                       "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
                       "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
                       "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                       "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list, tokenizer=LemmaTokenizer())
    doc_matrix = tfidf_vectorizer.fit_transform(
        sci_kb)

    eval_tfidf(dev_list, tfidf_vectorizer, doc_matrix, sci_kb, "dev_scores.pickle")
    eval_tfidf(test_list, tfidf_vectorizer, doc_matrix, sci_kb, "test_scores.pickle")

    return 0

main()


