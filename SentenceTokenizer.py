from konlpy.tag import Kkma
from konlpy.tag import Twitter

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