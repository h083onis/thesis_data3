from gensim.models import word2vec

def exe_word2vec():
    model = word2vec.Word2Vec(sents, size=512, window=3, min_count=3, iter=5)
    
if 