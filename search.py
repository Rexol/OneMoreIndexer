import pickle
import pandas as pd
import numpy as np
import re
import io

from nltk import wordnet, pos_tag
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from itertools import islice

# Loads Word2Vec
def load_vectors(fname, limit):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in islice(fin, limit):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


index = []
inverse_index = {}
sw = set(stopwords.words('english'))
vecs = load_vectors('crawl-300d-2M.vec', 300000)
zero = sum(vecs.values()) / len(vecs)

def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': wordnet.wordnet.ADJ,
        'V': wordnet.wordnet.VERB,
        'N': wordnet.wordnet.NOUN,
        'R': wordnet.wordnet.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.wordnet.NOUN


def my_lemmatizer(sent):
    lemmatizer = WordNetLemmatizer()
    tokenized_sent = sent.split()
    pos_tagged = [(word, get_wordnet_pos(tag))
                  for word, tag in pos_tag(tokenized_sent)]
    return ' '.join([lemmatizer.lemmatize(word, tag)
                     for word, tag in pos_tagged])


def normalize(query):
    html_pattern = re.compile('<script[\s\S]*?/script>[\s\S]*?|href=[\s\S]*?>[\s\S]*?|<br />|<ul[\s\S]*?/ul>[\s\S]*?|<li[\s\S]*?/li>[\s\S]*?|<style type[\s\S]*?/style>[\s\S]*?|<object[\s\S]*?/object>[\s\S]*?|(<a href[\s\S]*?>[\s\S]*?)|(\b(http|https):\/\/.*[^ alt]\b)|</ul>|</li>|<br/>|<!--[\s\S]*?-->[\s\S]*?|<div style[\s\S]*?>[\s\S]*?|<img[\s\S]*?>[\s\S]*?|<div id[\s\S]*?>[\s\S]*?|<div class[\s\S]*?>[\s\S]*?|</object>|<embed[\s\S]*?/>[\s\S]*?|<param[\s\S]*?/>[\s\S]*?|<noscript>[\s\S]*?</noscript>[\s\S]*?|<link rel[\s\S]*?>[\s\S]*?|<p style="text-align: center;">|<iframe[\s\S]*?</iframe>[\s\S]*?')
    href_removed = re.sub(html_pattern,'',query)
    words_only = re.sub(r'[^a-z ]+', r'', href_removed.lower())
    lemmatized = my_lemmatizer(words_only)
    normalized = ' '.join([word for word in lemmatized.split() if not word in sw])
    return normalized


class Document:
    def __init__(self, topic='', question='', answer='', d = {}):
        if 'C' in d.keys() and 'Q' in d.keys() and 'A' in d.keys():
            topic = d['C']
            question = d['Q']
            answer = d['A']
        self.topic = topic
        self.question = question
        self.answer = answer
    
    def format(self, query):
        # returns pair text-title formated to query.
        return [self.topic, self.question + ' ...', self.answer]

    def read_doc(self, id):
        fname = "files/" + str(id) + ".pkl"
        with open(fname, "rb") as f:
            doc = pickle.load(f)
            self.topic = doc['C']
            self.question = doc['Q']
            self.answer = doc['A']


# Reads data from .csv and creates inverted index.
def build_index():
    df = pd.read_csv('JEOPARDY_CSV.csv')
    df = df.dropna(axis=0)
    df = df.reset_index(0, drop=True)
    print('BUILD INDEX: BEFORE CYCLE')
    for i, row in df.iterrows():
        index.append(Document(row[' Category'], row[' Question'], row[' Answer']))
        if i%(df.shape[0]/4) == 0:
            print("BUILD INDEX: AT {:.2f}%".format(i*100/df.shape[0]))
    print('BUILD INDEX: END OF CYCLE')
    del df


# Reads inverted index from file.
def read_inverted_index(fname):
    with open(fname, "rb") as a_file:
        tmp = pickle.load(a_file)
        for k, v in tmp.items():
            inverse_index[k] = v


# Returns unsorted list of relevant documents.
# return size <= 100
def retrieve(query):
    query = normalize(query)
    print("RETRIEVE: QUERY IS {}".format(query))
    s_query = query.split()

    candidates = {}

    for w in s_query:
        if w in inverse_index.keys():
            # print('RETRIEVE: FOUND {} IN INV_IND'.format(w))
            for d in inverse_index[w]:
                if d[0] in candidates.keys():

                    candidates[d[0]] += d[1]
                else:
                    candidates[d[0]] = d[1]

    # candidates = sorted(candidates, reverse=True, key= lambda x: x[1])
    candidates = {k: v for k, v in sorted(candidates.items(), key=lambda item: item[1], reverse=True)}
    
    if len(candidates) > 100:
        return candidates[:100]
    else:
        return candidates[:]

# Gets embedding of given text using word2vec
def vectorize_text(text):
    splitted = text.split()
    return sum(list(map(lambda w: np.array(list(vecs.get(w, zero))), splitted))) / (
        len(splitted) if len(splitted) != 0 else 1)

# Process query and return relevant results.
def sort_relevance(query):
    candidates = retrieve(query)
    query_emb = vectorize_text(normalize(query))
    rating = []
    for i, freq in candidates:
        curr_doc = Document.read_doc(i)
        rating.append(((np.average(vectorize_text(normalize(curr_doc.question+curr_doc.answer))-query_emb))**2, curr_doc))
    rating.sort(key=lambda item: item[0])
    relevant_docs = [item[1] for item in rating]
    return relevant_docs