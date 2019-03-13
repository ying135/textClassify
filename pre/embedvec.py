import pickle
import numpy
import os

# load the pre-trained word-embedding vectors
# embeddings_index is a dictionary of all the wiki word
embeddings_index = {}
for i, line in enumerate(open('wiki-news-300d-1M.vec','r', encoding='UTF-8')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

with open(os.path.join('..', 'data', 'data.pkl'), 'rb') as f:
    data = pickle.load(f)
dict_src = data['dicts']['src']

embedding_matrix = numpy.zeros((len(dict_src), 300))
for i in range(len(dict_src)):
    word = dict_src.idx2word[i]
    vector = embeddings_index.get(word)
    if vector is not None:
        embedding_matrix[i, :] = vector[:]

output = open(os.path.join('..', 'data', 'embedmatrix.pkl'), 'wb')
pickle.dump(embedding_matrix, output)
output.close()

output = open(os.path.join('..', 'data', 'embed.vec'), 'w')
output.write(" ".join(list(map(str, embedding_matrix.tolist()))) + '\n')
output.close()
