from konlpy.corpus import kobill
from konlpy.tag import Twitter; t = Twitter()

from matplotlib import font_manager, rc
font_fname = 'C:\Windows\Fonts\malgun.ttf'     # A font of your choice
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

# -*- coding: utf-8 -*-

import nltk
#nltk.download()

files_ko = kobill.fileids()
doc_ko = kobill.open('1809890.txt').read()
tokens_ko = t.morphs(doc_ko)

ko = nltk.Text(tokens_ko, name='대한민국 국회 의안 제 1809890호')
ko.collocations()

tags_ko = t.pos("작고 노란 강아지가 페르시안 고양이에게 짖었다")
print (tags_ko)

parser_ko = nltk.RegexpParser("NP: {<Adjective>*<Noun>*}")
chunks_ko = parser_ko.parse(tags_ko)
chunks_ko.draw()
