# coding: utf-8

'''
    implement the LDA with gibbs sampling using scikit-learn
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
    Purpose: generate doc-term list(only use tf)
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
        # r"(?u)\b\w\w+\b" 默认
        # r"(?u)\b[^\d\s]+\b" 匹配非数字非空格
        tfidfvectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=10000, token_pattern=r"(?u)\b[^\d\s]+\b", stop_words=stop_words, norm=None, use_idf=False)
    else:
        tfidfvectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=10000, stop_words=stop_words, norm=None, use_idf=False)

    doc_term_tf = tfidfvectorizer.fit_transform(doc_list).astype(np.int32, copy=False)        
    term2id = tfidfvectorizer.vocabulary_    
    print("type(doc_term_tf): ", type(doc_term_tf))
    print("dictionary size: ", len(term2id))  
    print("doc_term_tf size: ", doc_term_tf.shape)      
    id2term = dict(zip(term2id.values(), term2id.keys()))

    # 将doc_term_tf矩阵转换为doc_tem_list，每行存储单词id列表
    doc_term_list = []
    for doc in doc_term_tf:
        data = doc.data
        ids = doc.nonzero()[1]
        term_list = []
        for i, w in enumerate(ids):
            term_list.extend([w]*data[i])
        doc_term_list.append(term_list)
    return doc_term_list, term2id, id2term
        
    

# 初始化，按照每个主题概率都相等的multinominal分布采样，更新采样出的主题的相关计数
def random_initialize(doc_term_list, ndz, nzw, nz, Z):
    """
    Input:
        doc_term_tf, ndz, nzw, nz, Z: 
    Output:
        None
    """
    print("Random initialize ...")
    for d, doc in enumerate(doc_term_list):
        zCur = []
        for w in doc:
            pz = np.divide(np.multiply(ndz[d,:], nzw[:,w]), nz)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            zCur.append(z)
            ndz[d,z] += 1
            nzw[z,w] += 1
            nz[z] += 1
        Z.append(zCur)
        


# gibbs采样，为文档中的每个单词重新采样主题
def gibbs_sampling(doc_term_list, Z, ndz, nzw, nz):
    """
    Input: doc_term_tf, Z, ndz, nzw, nz
    Output:
        None
    """
    # 为每个文档中的每个单词重新采样主题
    for d, doc in enumerate(doc_term_list):
        for index, w in enumerate(doc):
            z = Z[d][index]
            # 将当前文档当前单词原主题相关计数减去1
            ndz[d,z] -= 1
            nzw[z,w] -= 1
            nz[z] -= 1
            # 重新计算当前文档当前单词属于每个主题的概率
            pz = np.divide(np.multiply(ndz[d,:], nzw[:,w]), nz)
            # 按照计算出的分布进行采样
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            # 根据采样结果更新当前文档当前单词的主题
            Z[d][index] = z
            # 更加采样结果更新相关计数
            ndz[d,z] += 1
            nzw[z,w] += 1
            nz[z] += 1



# compute perplexity
def perplexity(ndz, nzw, nz, doc_term_list):
    nd = np.sum(ndz, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(doc_term_list):
        for w in doc:
            ll = ll + np.log(((nzw[:, w] / nz) * (ndz[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll/(-n))


# In[8]:

# Main 
def plsa_model(path, file_name=None, language='english', stop_words='english', num_topics=100, max_iteration=30, alpha=5, beta=0.1, num_words=10):
    """
    Purpose: main method
    Input: 
        path, file_name, language, stop_words, num_topics, max_iteration
        alpha, beta, num_words
    Output: 
        ndz: document->topic
        topicWords: the num_words of top words from each document
    """
    print("Main function of PLSA model...")
    doc_list = load_data(path, file_name=file_name)
    doc_term_list, term2id, id2term = preprocess_data(doc_list, language=language, stop_words=stop_words)
    N = len(doc_term_list)
    M = len(term2id)
    K = num_topics
    print("N={}, M={}, K={}".format(N, M, K))
    
    # 每篇文档采样的主题id
    Z = []
    # ndz[d,z]和nzw[z,w]表示针对文档d，单词w采样产生的主题计数
    ndz = np.zeros([N, K]) + alpha
    nzw = np.zeros([K, M]) + beta
    # nz[z]表示主题z产生的所有单词的总计数加伪计数
    nz = np.zeros([K]) + M * beta
    
    random_initialize(doc_term_list, ndz, nzw, nz, Z)
    for i in range(max_iteration):
        print("Gibbs sampling iteration {}".format(i))
        gibbs_sampling(doc_term_list, Z, ndz, nzw, nz)
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", perplexity(ndz, nzw, nz, doc_term_list))
    print("LDA model has done!")

    
    # 产生每个topic的前10个词
    topicWords = []
    if num_words:
        for z in range(K):
            ids = nzw[z,:].argsort()
            topicWord = []
            for j in ids:
                topicWord.insert(0, id2term[j])
            topicWords.append(topicWord[0:min(10, len(topicWord))])

    ndz = Normalizer().fit_transform(ndz)
        
    return ndz, topicWords


if __name__ == '__main__':
    path = "/home/jeson/Desktop/git_repository/NLP/LSA-PLSA-LDA/Data"
    file_name = "lda_data_chinese.txt"
    language = 'chinese'
    stop_words = 'stopwords.txt'
    num_topics = 10
    max_iteration=50
    alpha = 5
    beta = 0.1
    num_words = 10
    if (len(sys.argv) == 10):
        path = sys.argv[1]
        file_name = sys.argv[2]
        language = sys.argv[3]
        stop_words = sys.argv[4]
        num_topics = int(sys.argv[5])
        max_iteration = int(sys.argv[6])
        alpha = float(sys.argv[7])
        beta = float(sys.argv[8])
        num_words = int(sys.argv[9])
    
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
    print("alpha: ", alpha)
    print("beta: ", beta)
    print("num_words: ", num_words)
    print("==============parameters====================")


    doc_vec, topicWords = plsa_model(path, file_name=file_name, language=language, stop_words=stop_words, num_topics=num_topics, max_iteration=max_iteration, alpha=alpha, beta=beta, num_words=num_words)
    print(topicWords)



