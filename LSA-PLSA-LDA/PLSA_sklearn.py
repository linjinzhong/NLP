# coding: utf-8

'''
    implement the PLSA using scikit-learn
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
import os
import sys
import jieba
import numpy as np
import time



# Loading data
# file: if file_name exits
# dir: if file_name does not exits
def load_data(path, file_name=""):
    """
    Purpose：extract document list
    Input: 
        path: original text file path
        file_name: original text file name or None
    Output:
        doc_list: document list
    """
    print("Loading data ...")
    if not os.path.exists(path):
        print("The path is wrong!")
        return 
    doc_list = []
    if file_name:
        path_file = os.path.join(path, file_name)
        if not os.path.exists(path_file):
            print("The file does not exist!")
            return
        with open(path_file, 'r') as fin:
            doc_list = [line.strip() for line in fin.readlines()]
    else:
        file_names = os.listdir(path)
        for file_name in file_names:
            with open(os.path.join(path, file_name), 'r') as fin:
                for line in fin.readlines():
                    doc = line.strip()
                    doc_list.append(doc)
    print("Total number of documents: ", len(doc_list))    
    return doc_list
        


# Preprocessing data
def preprocess_data(doc_list, language='english', stop_words='english'):
    """
    Purpose: generate doc-term matrix(only use tf)
    Input:
        doc_list: document list
        language: chinese or english
        stop_words: 'english'/None or a list of words
    Output:
        doc_term_tf: document-term matrix(only use tf)
        term2id: term->id
        id2term: id->term 
    """
    print("Preprocessing data ...")
    if language == 'chinese':
        doc_list = [" ".join(jieba.cut(doc)) for doc in doc_list]
        tfidfvectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=10000, token_pattern=r"(?u)\b[^\d\s]+\b", stop_words=stop_words, norm=None, use_idf=False)
    else:     
        tfidfvectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=10000, stop_words=stop_words, norm=None, use_idf=False)
    
    doc_term_tf = tfidfvectorizer.fit_transform(doc_list)
    term2id = tfidfvectorizer.vocabulary_
    id2term = dict(zip(term2id.values(), term2id.keys()))
    return doc_term_tf, term2id, id2term
        
    

# Initialize parameters
def norm_parameters(pdz, pzw, N, M, K):
    """
    Purpose: nomr some parameters, so the sum of row equal to 1
    Input:
        pdz: P(z_k|d_i)the conditional probability of selecting topic z_k when document d_i is determined 
        pzw: P(w_j|z_k)the conditional probability of selecting term w_j when topic z_k is determined 
        N: number of documents
        M: number of terms
        K: number of topics
    Output:
        None
    """
    print("Norm pdz and pzw ...")
    pdz /= pdz.sum(axis=1).reshape(N,1)
    pzw /= pzw.sum(axis=1).reshape(K,1)



# expectation step
# update pdwz using pdz and pzw
def expectation_step(N, M, K, pdz, pzw, pdwz):
    """
    Purpose: compute pdwz
    pdwz = P(z_k|d_i,w_j)
         = P(z_k|d_i)P(w_j|z_k) / \sum_{l=1}^{K}P(z_l|d_i)P(w_j|z_l)
    """
    print("E-step: update pdwz ...")
    for i in range(N):
        for j in range(M):
            pdwz[i,j,:] = pdz[i,:] * pzw[:,j]
            denominator = pdwz[i,j,:].sum()
            if denominator == 0.:
                pdwz[i,j,:] = 0.
            else:
                pdwz[i,j,:] /= denominator
                
                

# maximization step
# update pdz and pzw using pdwz
def maximization_step(N, M, K, pdz, pzw, pdwz, doc_term_tf):
    """
    Purpose: update pdz and pzw
    pdz = \sum_{j=1}^{M}n(d_i,w_j)P(z_k|d_i,w_j) / n(d_i)
    pzw = \sum_{i=1}^{N}n(d_i,w_j)P(z_k|d_i,w_j) / \sum_{l=1}^{M}\sum_{i=1}^{N}n(d_i,w_l)P(z_k|d_i,w_l)
    """
    print("M-step: update pdz and pzw ...")
    # update pdz
    for i in range(N):
        for k in range(K):
            pdz[i,k] = doc_term_tf[i,:] * pdwz[i,:,k]
            denominator = doc_term_tf[i,:].sum()
            if denominator == 0.:
                pdz[i,k] = 1.0 / K
            else:
                pdz[i,k] /= denominator
    
    # update pzw
    for k in range(K):
        for j in range(M):
            pzw[k,j] = pdwz[:,j,k] * doc_term_tf[:,j]
        denominator = pzw[k,:].sum()
        if denominator == 0:
            pzw[k,:] = 1.0 / M
        else:
            pzw[k,:] /= denominator



# compute the log likelihood
def compute_loglikelihood(N, M, pdz, pzw, pdwz, doc_term_tf):
    """
    Purpose: compute the log likelihood
    L = \sum_{i=1}^{N}\sum_{j=1}^{M}n(d_i,w_j)log(\sum_{k=1}^{K}P(z_k|d_i)P(w_j|z_k))
    """
    print("Computer log likelihood ...")
    loglikelihood = 0.
    for i in range(N):
        for j in range(M):
            tmp = pdz[i,:].dot(pzw[:,j])
            if tmp > 0:
                loglikelihood += doc_term_tf[i,j] * np.log(tmp)
    return loglikelihood



# Main 
def plsa_model(path, file_name=None, language='english', stop_words='english', num_topics=100, max_iteration=30, threshold=10, num_words=10):
    """
    Purpose: main method
    Input: 
        path, file_name, language, stop_words, num_topics, num_topics, threshold, num_words
    Output: 
        doc_vec: the vectors of document list
    """
    print("Main function of PLSA model...")
    doc_list = load_data(path, file_name=file_name)
    doc_term_tf, term2id, id2term = preprocess_data(doc_list, language='english', stop_words='english')
    N, M = doc_term_tf.shape
    K = num_topics
    print("N={}, M={}, K={}".format(N, M, K))
    
    # pdz=P(z_k|d_i)
    pdz = np.random.random([N,K])
    # pzw=P(w_j|z_k)
    pzw = np.random.random([K,M])
    # pdwz = P(z_k|d_i,w_j)=P(z_k|d_i)P(w_j|z_k)/sum_{l=1}^{K}P(z_l|d_i)P(w_j|z_l)
    pdwz = np.zeros([N, M, K])
    norm_parameters(pdz, pzw, N, M, K)
    
    # EM algorithm
    oldLoglikelihood = 1
    print("begin to iteratively training ...")
    for i in range(max_iteration):
        expectation_step(N, M, K, pdz, pzw, pdwz)
        maximization_step(N, M, K, pdz, pzw, pdwz, doc_term_tf)
        newLoglikelihood = compute_loglikelihood(N, M, pdz, pzw, pdwz, doc_term_tf)
        
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'iteration: ', i+1, str(newLoglikelihood))    
        if (threshold and oldLoglikelihood != 1 and (newLoglikelihood - oldLoglikelihood) < threshold):
            break
        oldLoglikelihood = newLoglikelihood
    print("PLSA model has done!")
    
    # 产生每个topic的前10个词
    topicWords = []
    if num_words:
        for z in range(K):
            ids = pzw[z,:].argsort()
            topicWord = []
            for j in ids:
                topicWord.insert(0, id2term[j])
            topicWords.append(topicWord[0:min(10, len(topicWord))])

    pdz = Normalizer().fit_transform(pdz)

    # return document->topic
    return pdz, topicWords 



if __name__ == '__main__':
    path = "/home/jeson/Desktop/git_repository/NLP/LSA-PLSA-LDA/Data"
    file_name = "plsa_data_chinese.txt"
    language = 'chinese'
    stop_words = 'stopwords.txt'
    num_topics = 10
    max_iteration = 20
    threshold = 1.0
    num_words = 10
    if (len(sys.argv) == 9):
        path = sys.argv[1]
        file_name = sys.argv[2]
        language = sys.argv[3]
        stop_words = sys.argv[4]
        num_topics = int(sys.argv[5])
        max_iteration = int(sys.argv[6])
        threshold = float(sys.argv[7])
        num_words = int(sys.argv[8])
    
    if stop_words and stop_words != 'english':
        with open(os.path.join(path, stop_words), 'r') as fin:
            stop_words = [line.strip() for line in fin.readlines()]
        
    print("==============parameters====================")
    print("path: ", path)
    print("file_name: ", file_name)
    print("language: ", language)
    if stop_words and stop_words != 'english':
        print("number of stop_words: ", len(stop_words))
    else:
        print("stop_words: ", stop_words)
    print("num_topics: ", num_topics)
    print("max_iteration: ", max_iteration)
    print("threshold: ", threshold)
    print("num_words: ", num_words)
    print("==============parameters====================")

    
    doc_vec, topicWords = plsa_model(path, file_name=file_name, language=language, stop_words=stop_words, num_topics=num_topics, max_iteration=max_iteration, threshold=threshold, num_words=num_words)
    print(topicWords)



