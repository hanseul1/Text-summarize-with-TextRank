import SentenceTokenizer
import GraphMatrix
import TextRank 

class Ranking(object):
    def __init__(self, text):
        self.sentence_tokenize = SentenceTokenizer()

        self.sentences = self.sentence_tokenize.text_to_sentence(text)
        self.words = self.sentence_tokenize.get_wrods(self.sentences)

        self.graph_matrix = GraphMatrix()
        self.sentence_graph = self.graph_matrix.create_sentence_graph(self.words)
        self.words_graph, self.idx2word = self.graph_matrix.create_words_graph(self.words)

        self.textRank = TextRank()
        self.sentence_rank_idx = self.textRank.get_ranks(self.sentence_graph)
        self.sorted_sent_rank_idx = sorted(self.sentence_rank_idx, key=lambda k: self.sentence_rank_idx[k], reverse=True)
        
        self.word_rank_idx = self.textRank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
        
    def keywords(self, word_num=10):
        keywords = []
        index = []

        for idx in self.sorted_word_rank_idx[:word_num]:
            index.append(idx)

        for idx in index:
            keywords.append(self.idx2word[idx])
        
        return keywords
