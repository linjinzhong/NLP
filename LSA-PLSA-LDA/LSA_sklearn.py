'''
    implement the LSA using scikit-learn
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import os
import sys
import jieba


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
    print("Loading data...")
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
    Purpose: generate doc-term matrix
    Input:
        doc_list: document list
        language: chinese or english
        stop_words: 'english' or a list of words
    Output:
        doc_term: document-term matrix
        dictionary: term2id
    """
    print("Preprocessing data...")
    if language == 'chinese':
        doc_list = [" ".join(jieba.cut(doc)) for doc in doc_list]
        # r"(?u)\b\w\w+\b" 默认
        # r"(?u)\b[^\d\s]+\b" 匹配非数字非空格
        tfidfvectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=10000, token_pattern=r"(?u)\b[^\d\s]+\b", stop_words=stop_words, norm=None, use_idf=False)
    else:
        tfidfvectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=10000, stop_words=stop_words, use_idf=True)
    
    doc_term = tfidfvectorizer.fit_transform(doc_list)
    term2id = tfidfvectorizer.vocabulary_
    id2term = dict(zip(term2id.values(), term2id.keys()))
    return doc_term, term2id, id2term
        
    

# SVD decomposition
def use_svd(doc_term, num_topics):
    """
    Purpose: use truncated svd to get documnet vector in low dimentions
    Input:
        doc_term: document term matrix
        num_topics: number of topics,
    Output:
        doc_vec: the vectors of document list
    """
    print("SVD decomposition...")
    svd = TruncatedSVD(n_components=num_topics)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    doc_vec = lsa.fit_transform(doc_term)
    return doc_vec, svd



# Main 
def lsa_model(path, file_name=None, language='english', stop_words='english', num_topics=100, num_words = 10):
    """
    Purpose: main method
    Input: 
        path, file_name, language, stop_words, num_topics, num_words
    Output: 
        doc_vec: the vectors of document list
    """
    print("Main function of LSA model...")
    doc_list = load_data(path, file_name=file_name)
    doc_term, term2id, id2term = preprocess_data(doc_list, language='english', stop_words='english')
    N, M = doc_term.shape
    K = num_topics
    print("N={}, M={}, K={}".format(N, M, K))   

    doc_vec, svd = use_svd(doc_term, num_topics)
    print("LSA model has done!")

    # 产生每个topic的前10个词    
    topicWords = []
    if num_words:
        for z in range(K):
            ids = svd.components_[z].argsort()
            topicWord = []
            for j in ids:
                topicWord.insert(0, id2term[j])
            topicWords.append(topicWord[0:min(10, len(topicWord))])

    return doc_vec, topicWords



if __name__ == '__main__':
    path = "/home/jeson/Desktop/git_repository/NLP/LSA-PLSA-LDA/Data"
    file_name = "plsa_data_chinese.txt"
    language = "chinese"
    stop_words = "stopwords.txt"
    num_topics = 50
    num_words = 10
    if (len(sys.argv) == 6):
        path = sys.argv[1]
        file_name = sys.argv[2]
        language = sys.argv[3]
        stop_words = sys.argv[4]
        num_topics = int(sys.argv[5])
        num_words = int(sys.argv[6])
    print("==============parameters====================")
    print("path: ", path)
    print("file_name: ", file_name)
    print("language: ", language)
    print("stop_words: ", stop_words)
    print("num_topics: ", num_topics)
    print("==============parameters====================\n")

    doc_vec, topicWords = lsa_model(path = path, file_name=file_name, language=language, stop_words=stop_words, num_topics=num_topics, num_words=num_words)
    print(topicWords)
