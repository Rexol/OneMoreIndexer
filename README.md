# OneMoreIndexer
Search engine project from Tinkoff generation ML course

1. (sudo) pip install flask
2. (sudo) python server.py
3. download word2vec

dataset from https://www.kaggle.com/tunguz/200000-jeopardy-questions

I apply normalization to query (remove stop-words, remove unnecessary data(links, symbols), apply lemmatization)

Then I get 100 most relevant docs based on frequensy of words in query. Then the array of words sorted based on diffrense in query vector and document vector (using word2vec normalization)
