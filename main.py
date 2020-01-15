from konlpy.tag import Kkma
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import pymongo
from pymongo import MongoClient

class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.twitter = Twitter()
        self.stopwords = []

    def text_to_sentence(self, text):
        sentences = self.kkma.sentences(text)

        # 너무 짧은 문장은 앞 문장에 addition 처리
        for i in range(0, len(sentences)):
            if len(sentences[i]) <= 10:
                sentences[i-1] += (' ' + sentences[i])
                sentences[i] = ''
        
        return sentences
    
    def get_words(self, sentences):
        words = []

        for s in sentences:
            if s is not '':
                # twitter 모듈의 nouns(string) 함수를 통해 키워드 추출 후
                # 추출한 키워드 리스트 중 stop words가 아니고 두글자 이상인 경우 결과 리스트에 append
                words.append(' '.join([word for word in self.twitter.nouns(str(s)) if word not in self.stopwords and len(word) > 1]))
        
        return words


class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
    
    # TF-IDF 모델 적용한 sentence-term matrix 생성
    def create_sentence_graph(self, sentence):
        tfidf_matrix = self.tfidf.fit_transform(sentence).toarray()
        # sentence-matrix와 그 전치행렬을 곱하여 sentence correlation matrix 생성(가중치 계산)
        self.graph_sentence = np.dot(tfidf_matrix, tfidf_matrix.T)
        return self.graph_sentence

    # term count 방식을 활용한 sentence-term matrix 생성
    def create_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        # sentence 배열에서 추출한 vocabulary
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}
    

# TextRank 수식 계산
class TextRank(object):
    def get_rank(self, graph, d=0.85):
        A = graph
        matrix_size = A.shape[0]

        for i in range(matrix_size):
            A[i, i] = 0  # 대각선 부분 = 0
            link_sum = np.sum(A[:,i])  # A[:, i] = A[:][i]

            if link_sum != 0:
                A[:,i] /= link_sum
            A[:,i] *= -d
            A[i,i] = 1

        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)  # solve Ax = B
        
        return {idx: r[0] for idx, r in enumerate(ranks)}
    
    
class Ranking(object):
    def __init__(self, doc):
        self.sentence_tokenize = SentenceTokenizer()

        self.sentences = []
        for text in doc:
            self.sentences += self.sentence_tokenize.text_to_sentence(text)
        self.words = self.sentence_tokenize.get_words(self.sentences)

        self.graph_matrix = GraphMatrix()
        self.sentence_graph = self.graph_matrix.create_sentence_graph(self.words)
        self.words_graph, self.idx2word = self.graph_matrix.create_words_graph(self.words)

        self.textRank = TextRank()
        self.sentence_rank_idx = self.textRank.get_rank(self.sentence_graph)
        self.sorted_sent_rank_idx = sorted(self.sentence_rank_idx, key=lambda k: self.sentence_rank_idx[k], reverse=True)
        
        self.word_rank_idx = self.textRank.get_rank(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
        
    def keywords(self, word_num=20):
        keywords = []
        index = []

        for idx in self.sorted_word_rank_idx[:word_num]:
            index.append(idx)

        for idx in index:
            keywords.append(self.idx2word[idx])
        
        return keywords

# MongoDB connection & querying data
client = MongoClient()
db = client.allreview
collection = db.review
posts = db.posts

document = []
for post in posts.find():
    document.append(post['context'])

# Top20 keywords extraction
rank = Ranking(document)

print(rank.keywords())