import gensim
#gensim.models.KeyedVectors.load_word2vec_format(args.model.strip(), )
model = gensim.models.KeyedVectors.load_word2vec_format('german.model', binary=True,)
print('Loaded model.')