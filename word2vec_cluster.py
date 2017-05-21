# -*- coding: utf-8 -*-
import argparse
import time
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import json
import io, os, re

from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from collections import defaultdict
from konlpy.tag import Twitter; t = Twitter()

posts = []
docs = []
num_task = 3

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


def readConfig(filename):
    f = open(filename, 'r', encoding='utf-8')
    js = json.loads(f.read())
    f.close()
    return js


def jsonParsing():
    rep = {}  # define desired replacements here

    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    # text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

    for i in range(num_task):
        docs.append([])

    # docs = open("./crawled.txt", "w", encoding='utf-8')
    for root, dirs, files in os.walk('./data/'):
        for fname in files:
            with io.open('./data/' + fname, encoding="utf-8") as infile:
                data = json.load(infile)
            temp_study = ""
            temp_date = ""
            temp_rest = ""

            for d in data:
                if ('tags' not in d) or (len(d['tags']) == 0):
                    continue

                posts.append(" ".join(d['tags']))

                if any(word in '공부 시험 중간고사 기말고사 중간고사 시험 보고서 레포트 고시생 고시' \
                               '학원' for word in d['tags']):
                    temp_study += " ".join(d['tags'])
                                  # + '\n'

                if any(word in '럽스타그램 남자친구 사랑해 여자친구 데이트 달달 여친 남친 데이뚜 서방님' for word in d['tags']):
                    temp_date += " ".join(d['tags'])
                                 # + '\n'

                if any(word in '휴식 여유 힐링 일요일 빈둥 노가리 rest ' for word in d['tags']):
                    temp_rest += " ".join(d['tags'])
                                 # + '\n'

            if temp_study:
                docs[0].append(temp_study)
            if temp_date:
                docs[1].append(temp_date)
            if temp_rest:
                docs[2].append(temp_rest)
            # docs.write(pattern.sub(lambda m: rep[re.escape(m.group(0))], d['caption'])+'\n')


def make_model():
    # load
    global posts
    from konlpy.corpus import kobill
    # docs_ko = [kobill.open(i).read() for i in kobill.fileids()]
    docs_ko = []
    for i in range(num_task):
        docs_ko += docs[i]
    # print (docs_ko)

    # Tokenize
    from konlpy.tag import Twitter
    t = Twitter()
    pos = lambda d: ['/'.join(p) for p in t.pos(d)]
    texts_ko = [pos(doc) for doc in docs_ko]
    # print (texts_ko)

    # train
    from gensim.models import word2vec
    wv_model_ko = word2vec.Word2Vec(texts_ko)
    wv_model_ko.init_sims(replace=True)

    wv_model_ko.save('ko_word2vec_e.model')


def make_input():
    inputs = []
    tag_only = []

    rep = {"\n": " ", "#": " ", "ㅋ": "", "ㅎ": "", "・": "", "투썸플레이스": "", "twosomeplace": "", "투썸": ""}  # define desired replacements here

    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))

    for root, dirs, files in os.walk('./data/'):
        for fname in files:
            with io.open('./data/' + fname, encoding="utf-8") as infile:
                data = json.load(infile)

            for d in data:
                if ('tags' not in d) or (len(d['tags']) == 0):
                    continue
                tag_only.append(" ".join(d['tags']))

                if ('caption' in d) and len(d['caption']) > 0:
                    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], d['caption'])
                    refined = ""
                    for c in text:
                        if c.isalnum() and not c.isdigit():
                            refined += c
                        elif len(refined) > 0 and refined[-1] != " ":
                            refined += " "
                    # print (refined)
                    inputs.append(refined)

    return inputs, tag_only

if __name__ == "__main__":
    # jsonParsing()
    # make_model()

    w2v_model = Word2Vec.load('./ko.bin')
    w2v = dict(zip(w2v_model.wv.index2word, w2v_model.wv.syn0))
    test = MeanEmbeddingVectorizer(w2v)

    etree_w2v = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

    etree_w2v_tfidf = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

    X = [['시험', '기간', '공부', '책', '논문', '연구', '과제', '학점', '종강', '도서관', '학기'],
         ['러브', '데이트', '커플', '신랑', '신부'],
         ['혼자', '혼', '홀로', '잉여', '고독', '일상', '공허', '몸', '직장인', '일기'],
         ['친', '친구', '수다', '우정', '죽마고우', '불알', '멤버', '합체', '브로', '학년'],
         ['모임', '가족', '파티', '계', '팟', '형', '맥주', '언니', '수다', '만남'],
         ['베이비', '아가', '육아', '맘', '아기', '임산부'],
         ['이벤트', '오픈', '예정', '할인', '지하', '뒷편', '샵', '층', '무료'],
         ['본사', '직원', '사장님', '매장', '메뉴', '장님']]

    y = ['공부', '데이트', '혼자', '친구랑', '모임', '아이랑 놀기', '광고', '매장이야기']

    keywords = []
    for i in range(len(X)):
        for j in range(len(X[i])):
            keywords.append(X[i][j])

    stop_words = ['그램', '첫줄', '대전', '땜', '카페', '점', '대서문', '둔산동', '대전대', '유성', \
                  '아메리카노', '맛스타', '스타', '아이스', '초코', '모카', '치즈케이크', '여행', '동유럽', '휴일', \
                  '크림', '케이크', '커피', '투썸플레이스', '오늘', '크리', '국내', '맞팔', '팔로우', \
                  '스', '서문', '대전대학교', '끝', '일요일', '월요일', '화요일', '수요일', '목요일', \
                  '금요일', '토요일', '주말', '투썸', '콜드', '브루', '그린티', '라떼', '데', '궁', '시렁', '마트', \
                  '하니', '지니', '스트로베리', '요거트', '프라', '페', '가나', '슈', '타', '티', '세트', '아메', '댓', \
                  '청주', '씨제이', '푸드', '현빈', '치즈', '베리', '다이어트', '하리', '브라징', '대학교', '아이스크림', \
                  '크리스마스', '오해', '거', '바', '봄', '못', '어요', '올해', '또', '푸딩', '스페셜', '딸기', '저녁', \
                  '아침', '오후', '오전', '충', '바닐라', '샌드위치', '디저트', '일리', '간식', '만세', '얼', '셀', '연휴', \
                  '급', '칩', '짱', '시', '덕분']


    etree_w2v.fit(X, y)
    etree_w2v_tfidf.fit(X, y)

    caption, tag_only = make_input()
    # print (inputs)

    from konlpy.tag import Twitter
    t = Twitter()
    pos = lambda d: [p[0] for p in t.pos(d) if p[1] == "Noun"]
    texts_ko = [pos(doc) for doc in tag_only]
    texts_ko_cap = [pos(doc) for doc in caption]

    refined = []
    for i in range(len(texts_ko)):
        flag = 0
        for tag in texts_ko[i]:
            if (tag in w2v_model.wv.vocab) and (not tag in stop_words):
                flag = 1
                break

        if flag == 1:
            after_stop = []

            for tag in texts_ko[i]:
                if (tag in w2v_model.wv.vocab) and (not tag in stop_words):
                    after_stop.append(tag)

            for tag in texts_ko_cap[i]:
                if tag in keywords and (not tag in after_stop):
                    after_stop.append(tag)

            refined.append(after_stop)

    ### Replacement for word2vec dictionary ###
    for tag in refined:
        for i in range(len(tag)):
            if tag[i] == '럽': tag[i] = '러브'
            if tag[i] == '예랑': tag[i] = '예비신랑'
            if tag[i] == '예신': tag[i] = '예비신부'
            if tag[i] == '남친': tag[i] = '남자친구'
            if tag[i] == '여친': tag[i] = '여자친구'

    outputs1 = etree_w2v.predict(refined)
    outputs2 = etree_w2v_tfidf.predict(refined)

    for i in range(len(refined)):
        # print ((refined[i], outputs1[i], outputs2[i]))
        print ((refined[i], outputs1[i]))