'''
LSA: Latent Semantic Analysis (隐语义分析)

'''

# ============= Import the required library ================
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


# ============= Loading Data ===============================
# loading article.txt
def load_data(path, file_name):
    """
    Purpose: extract document list and title list from text file in which each line is a document 
    Input : 
        path: original text file path
        file_name: original text file name
    Output : 
        document_list: document list
        title_list: title list(initial 100 words of each document)
    """
    document_list = []
    title_list = []
    with open(os.path.join(path, file_name), 'r') as fin:
        for line in fin.readlines():
            doc = line.strip()
            document_list.append(doc)
            title_list.append(doc[0:min(len(doc),100)])
        print("Total Number of Documents: ", len(document_list))
    return document_list, title_list


# ============= Preprocessing Data ===============================
# aim for english document
# tokenize the documents (切分文档)
# remove stop words　（去除停用词）
# perform stemming on words（提取词干）
def preprocess_data(document_list):
    """
    Purpose: preprocess text (tokenize, remove stopwords, stemming)
    Input: 
        document_list: document list
    Output: 
        clean_document_list: preprocessed document list
    """
    # initial regexp tokenizer (初始化正则切分)
    tokenize = RegexpTokenizer(r'\w+')
    # create english stopwords (创建英文停用词表集合)
    en_stop = set(stopwords.words('english'))
    # create stem class (创建词干类)
    p_stemmer = PorterStemmer()
    # get clean document list after preprocessing each document (每篇文档进行分词去停用词取词干得到干净的文档列表)
    clean_document_list = []
    # iteratively preprocess each document (循环处理每篇文档)
    for doc in document_list:
        # transform to lower (转换为小写字母)
        raw_doc = doc.lower()
        # split words (分词)
        tokenized_words = tokenize.tokenize(raw_doc)
        # remove stopwords (去除停用词)
        stopped_words = [word for word in tokenized_words if not word in en_stop]
        # get stem words (取词干)
        stemmed_words = [p_stemmer.stem(word) for word in stopped_words]
        # add to clean document list (添加到干净的文档列表)
        clean_document_list.append(stemmed_words)
    return clean_document_list


# ============= Preprocessing Corpus ===============================
# create document-term matrix and dictionary of terms
def prepare_corpus(clean_document_list):
    """
    Purpose: create term dictionary and convert clean document list into document-term matrix
    Input: 
        clean_document_list: clean document list
    Output: 
        dictionary: term dictionary
        doc_term_matrix: document-term matrix
    """
    # create dictionary (创建词典)
    dictionary = corpora.Dictionary(clean_document_list)
    # create document-term matrix (生成词－文档矩阵，使用gensim里面的corpora.dictionary类的doc2bow（文档到词袋向量）方法)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_document_list]
    return dictionary, doc_term_matrix



# ============= Create an LSA model using Gensim ====================
# create lsa model using gensim
def create_gensim_lsa_model(clean_document_list, num_topics, num_words):
    """
    Purpose: create LSA model　using gensim
    Input: 
        clean_document_list: clean document lsit
        num_topics: number of topics
        num_words: the number of words associate with each topic
    Output: LSA model
    """
    # prepare corpora (准备语料)
    dictionary, doc_term_matrix = prepare_corpus(clean_document_list)
    # create LSA model (创建LSA模型)
    LSA_model = LsiModel(doc_term_matrix, num_topics = num_topics, id2word = dictionary)
    print(LSA_model.print_topics(num_topics = num_topics, num_words = num_words))
    return LSA_model



# ============== Determine the number of topics =====================
# generate coherence scores to determine an optimum number of topics
def compute_coherence_values(dictionary, doc_term_matrix, clean_document_list, stop, start = 2, step = 3):
    """
    Purpose: Compute coherence for various number of topics
    Input: 
            dictionary: gensim dictionary   
            doc_term_matrix:　document term matrix
            clean_document_list: preprocessed clean document list
            stop: max num of topics
            start: min number of topics
            step: change number of number of topics
    Output: model_list: list of LSA topic model
            coherence_values: Coherence values corresponding to the different number of topics
    """
    model_list = []
    coherence_values = []
    for num_topics in range(start, stop, step):
        model = LsiModel(doc_term_matrix, num_topics = num_topics, id2word = dictionary)
        model_list.append(model)
        coherence_model =  CoherenceModel(model = model, texts = clean_document_list, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherence_model.get_coherence())
    return model_list, coherence_values

# plot coherence score values
def plot_graph(clean_document_list, start, stop, step):
    """
    Purpose: plot graph to show the coherence difference with different number of topics
    Input:  
        clean_document_list: clean document list
        strat: min number of topics
        stop: max number of topics
        step: step size
    Output: 
        None
    """
    dictionary, doc_term_matrix = prepare_corpus(clean_document_list)
    model_list, coherence_valus = compute_coherence_values(dictionary, doc_term_matrix, clean_document_list, stop, start, step)
    # show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_valus)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_valus"), loc='best')
    plt.show()




# ============== Main ================================================
cur_path = os.getcwd()
print('cur_path: ', cur_path)
file_path = os.path.join(cur_path, 'Data')

# # determine the number of topics
# start, stop, step = 2, 12, 1
# document_list, title_list = load_data(file_path, 'articles.txt')
# clean_document_list = preprocess_data(document_list)
# plot_graph(clean_document_list, start, stop, step)


# the number of topics (主题数量)
num_topics = 4
# the number of words for each topic (每个主题打印词的个数)
num_words = 10
# load data (加载数据（生成原始文档列表，每篇文档是一个字符串）)
document_list, title_list = load_data(file_path, "articles.txt")
# preprocess data (预处理数据（统一到小写字母、切词、去停用词、取词干）)
clean_document_list = preprocess_data(document_list)

# 创建LSA模型
LSA_model = create_gensim_lsa_model(clean_document_list, num_topics, num_words)
