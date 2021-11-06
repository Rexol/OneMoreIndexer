from flask import Flask, render_template, request
from search import sort_relevance, read_inverted_index
from time import time

app = Flask(__name__, template_folder='.')
# print("TASK: START BUILD DIRECT INDEX")
# build_index()
print("DONE: DIRECT INDEX READ")
read_inverted_index("inverse_index.pkl")
print("DONE: INVERSE INDEX READ")


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = sort_relevance(query)
    print(len(documents))

    results = [doc.format(query) for doc in documents]
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Aonex', # Another one indexer
        results=results
    )


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=80)
