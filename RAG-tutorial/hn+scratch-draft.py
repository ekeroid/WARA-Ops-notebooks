# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
"""
# A bare-bones RAG implementation
"""

# %% [markdown] jp-MarkdownHeadingCollapsed=true
"""
## Table of contents

0. Introduction
1. Data preparation
2. Data ingestion and chunking
3. Embedding cunks
4. Feeding a vector store with embeddings
5. **Retrieval** of relevant (embedded) facts
6. **Augmented** prompt preparation
7. **Generation** of a better answer 
"""

# %% [markdown]
"""
## 0. Introduction
"""

# %% [markdown]
"""
Retrieval Augmented Generation (RAG) is intended to alleviate some of the most obvious [problems][1] displayed by Large Language Models (LLMs). RAG consists of three steps; *retrieval* of context-specific information, *augmenting* (i.e. adding context to) the LLM [prompt][6], and letting the LLM *generate* an answer taking that context into account.

This notebook intends to demystify the steps involved by performing RAG, getting by without the help of tools like [LangChain][2] that streamlines the process but also obscures what is really going on.

The example in this notebook is (loosely) based on [this blog post][3] but getting rid of a few dependecies and using [Ollama][4] instead of [ChatGPT][5]. The steps unique to RAG, the context generation and retrieval, will be explored in some detail.

[1]: https://youtu.be/T-D1OfcDW1M?si=nKf8KC93tcsbbAlO
[2]: https://www.langchain.com
[3]: https://vigneshwarar.substack.com/p/hackernews-support-page-using-retrieval
[4]: https://ollama.com
[5]: https://openai.com/chatgpt/
[6]: https://medium.com/thedeephub/llm-prompt-engineering-for-beginners-what-it-is-and-how-to-get-started-0c1b483d5d4f
"""

# %% [markdown]
"""
The figure below outlines the RAG process, and this notebook begins in the top left corner of the figure with the process of turning textual facts into searchable information after a quick dive into the required software and a brief look at interacting with LLMs **without** RAG.
"""

# %% [markdown]
"""
<!--![image.png](img/image1.png)-->
<img src="img/image1.png" alt="Big picture" style="width: 400px;"/>
"""

# %% [markdown] editable=true slideshow={"slide_type": ""}
"""
### Prerequisites

The very first step is to make sure all requirements (in terms of python modules) are satisfied.
My suggestion is to open a command line interface **in Jupyter** (`File -> New... -> Terminal`) and run the commands there instead of in this notebook:

```
jovyan@jupyter-user:~$ pip install sentence-transformers faiss-cpu==1.8.0 qdrant-client ollama
```

Now we can start with a python preamble free from [cargo-cult](https://en.wikipedia.org/wiki/Cargo_cult_programming) imports:
"""
# %%
# This file is empty :)

# %% [markdown]
"""
### Baseline

Before getting into RAG, and what it contributes, let's establish a baseline by querying our LLM without using RAG.

First, connect to the LLM (Ollama) service that is provided by ERDC:
"""
# %%
import ollama
from ollama import Client

ollama_host = 'http://10.129.20.4:9090'
client = Client(host=ollama_host)
# %% [markdown]
"""
Next, prepare a _prompt_ with the question and some instructions for the LLM:
"""
# %%
prompt = """You are an AI assistant. Your task is to understand the user question, and provide an answer.

Your answers are short, to the point, and written by an domain expert.
If you don't know the answer, simply state, "I don't know"

User question: What is special about HackerNews?
"""
# %% [markdown]
"""
Feed the prompt to the LLM and sit back and wait for the answer:
"""

# %%
ollama_model = 'llama3:70b'
reply = client.chat(
    model=ollama_model,
    messages=[{'role': 'user', 'content': prompt}],
    stream=False,
)

print(reply['message']['content'])
# %% [markdown] editable=true slideshow={"slide_type": ""}
r"""
## 1. Data preparation
In order to demonstrate RAG capabiliteis, we need some focussed facts (context), to work with.
Some of the info-pages from HackerNews (legal.html, newsfaq.html, newsguidelines.html, security.html) was downloaded and converted to plain text and put in JSON-files<sup>1</sup> like so:

```json
{
    "content": "Hacker News Guidelines\n...",
    "url": "https://news.ycombinator.com/newsguidelines.html"
}
```

and stored in a `data` directory:

```
data/
    legal.json
    newsfaq.json
    newsguidelines.json
    security.json
```

You can download the data here: [data.tgz](https://frontend-compute.wara-ops.org/user/eperspe/files/llama/data.tgz?_xsrf=2%7C6ec57497%7Ca87c71b678f374f83bcd07f644e9307a%7C1717766621)

Unpack it in your working directory using the command `tar -xvzf data.tgz`

---
<p><small>1. The reason for keeping the url is to be able to (manually) track and reference the original document in the response as a post-query operation.</small></p>

---
"""

# %% [markdown]
"""
## 2. Data ingestion and chunking


Next, we have to somehow prepare our data for use with an LLM prompt; _ingest_ it.

The goal is to make our custom data ready for _semantic querying_ (see e.g. [King – Man + Woman = Queen](https://www.technologyreview.com/2015/09/17/166211/king-man-woman-queen-the-marvelous-mathematics-of-computational-linguistics/))

Basically, there are two steps required; 1) _chunking_, i.e. turning data into smaller pieces, and 2) turning those chunks into a _semantic vectors_ of high dimension (a.k.a _embeddings_) that captures the semantics, the meaning, of the corresponding chunk.

Next, we'll look at chunking in some detail, and postpone explanation of semantic vectors/embeddings to later, just keep the concept in mind for now.
"""
# %% [markdown]
"""
### Chunking

Since a (semantic) vector of finite length, can carry only a limited amount of information, we need to limit the contextual scope of each vector – a process known as chunking. Too small chunks doesn't have enough context, but too large chunks may contain unrelated contexts. This is not an exact science, and domain knowledge and understanding of the process helps in guiding the trade-off choices. For this example, ordinary sentences will make up the chunks, but e.g. paragraphs might proove to be a good alternative.

For the sake of clarity, I will show how to split the data into senteces (chunks) using plain python. For a real world deployment a natural language processing library like [spaCy](https://spacy.io) would be a better choice, but for this simple demo it would be total overkill and just obscure what is going on.
"""

# %%
#
# Split the input data into sentence-sized chunks
#
import re
import json

chunks = []
index = 0

filenames = ["newsfaq.json", "newsguidelines.json", "security.json", "legal.json"]
# Iterate over the entries in data/ and read each JSON file in turn
for filename in filenames:
    filepath = f"./data/{filename}"
    with open(filepath) as fd:
        data = json.load(fd)
        
    url = data['url']
    text = data['content']
    # Split the file's text contents into sentences using python regex:
    #   A sequence of characters is deemed a sentence if followed by a
    #   full stop (.), question mark (?), or an exclamation mark (!)
    #   immediately followed by one or more whitespaces.
    sentences = re.split(r"(?<=\.|\?|!)\s+", text)
    # Each sentence make up a chunk, store it with references (url and id)
    for sentence in sentences:
        chunks.append({'id': index, 'text': sentence, 'url': url})
        index += 1

# Write the resulting array to file:
with open('chunks.json', 'w') as fd:
    json.dump(chunks, fd)

# %%
# Just a sanity check, it should be ~570 chunks
len(chunks)


# %% [markdown]
"""
<!--![image.png](img/image2.png)-->
<img src="img/image2.png" alt="Drawing" style="width: 400px;"/>

Figure. An illustration of the chunk data format used
"""
# %% [markdown]
"""
For reference, the code from the original blog post using spaCy is reproduced below:
"""

# %%
# import spacy
#
# nlp = spacy.load("en_core_web_sm")
#
# def process_file(file_path):
#     with open(file_path) as f:
#         data = json.load(f)
#         content = data['content']
#         url = data['url']
#         doc = nlp(content)
#
#         return [{'text': sent.text, 'url': url} for sent in doc.sents]
#
# chunks = [chunk for file in os.listdir('data') for chunk in process_file(os.path.join('data', file))]
#
# chunks = [{'id': i, **chunk} for i, chunk in enumerate(chunks)]
#
# with open('chunks.json', 'w') as f:
#     json.dump(chunks, f)


# %% [markdown]
r"""
## 3. Embedding

Embedding is the process of turning data into semantic vectors representing that data in a way that makes it suitable for computers. With that sufficiently vague statement it is worth pointing out that data could be almost anything; text, images, etc. that can be represented as a coordinate (embedded) in a semantic space ${\mathbb R}^n$, where $n$ is large, typically $512$, $768$ or some such. If that didn't help in understanding, consider the two-dimensional space ${\mathbb R}^2$ with `redness`and `blueness` on the axes as shown in the picture below. The colour purple, which is a linear combination of red and blue in RGB-colorspace, would be somewhere along the diagonal $y \approx x$.
"""

# %% [markdown]
"""
<img src="img/image3.png" alt="Example" style="width: 400px;"/>

Figure. A two-dimensional example embedding of red- and blueness, where purpleness emerges as a combination of the two.
"""

# %% [markdown]
"""
Embedding, i.e. converting a chunk to a semantic vector by a _vectorizer_, can be done in many ways, most often using a trained neural network. The state-of-the-art is progressing rapidly, and be sure to check out some kind of leaderboard for the latest and greatest for a real world deployment. That said, the purpose of any vectorizer is simple: construct a set of vectors that represent the semantic information in the best possible way. For an example, see e.g. [semantic textual similarity](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html).

Many vectorizers are proprietary and/or come with complicated licenses for their use, so we'll limit this discussion to a free vectorizer:

- sentence_transformers (SBERT/Hugging Face) <https://www.sbert.net>

We'll be using it with the pre-trained model `all-mpnet-base-v2`. Any difference in performance and quality compared to the current best-in-class will be negligible in this example.
"""

# %%
from sentence_transformers import SentenceTransformer

# Gather the sentences from our chunks
sentences = [chunk['text'] for chunk in chunks]
model_name = 'sentence-transformers/all-mpnet-base-v2'
# Get the model
model = SentenceTransformer(model_name)
model.max_seq_length
# Vectorize, i.e. create embeddings
embeddings = model.encode(sentences, show_progress_bar=True)

# %% [markdown]
"""
Disregard warnings above like `TqdmExperimentalWarning: ...`, it is beyond our control.
"""

# %% [markdown]
r"""
## 4. Feeding a vector store (lib or db)

Now that we have our embeddings, we need to store them somwhere **and** make them searchable.
Storing is straightforward, but searching is in practice not so simple.

The reason searching is the bottleneck, is due to the sheer number of vectors to search. The search operation, _similarity search_ is basically a very simple operation:

Given a set of semantic vectors ${\bf x}_i \in \left\{x_1,\ldots,x_n\right\}$ (i.e. in ${\mathbb R}^n$), find the vector(s) most closely matching a _query vector_ ${\bf x}$ by finding

$i = {\textit argmin}_i||{\bf x} - {\bf x}_i||$,

i.e. (the index of) the most similar vector, where $||\cdot||$ is the Euclidean distance (${\textrm L}_2$) in ${\mathbb R}^n$.

To continue the example from above, a search for the (red, blue) vector that most closely matches <img src="img/query.png" alt="query color" width="17" height="17" style="vertical-align:middle">, whose _embedded query vector_ is (0.3, 0.6), yields an answer of (0.25, 0.5) equivalent to <img src="img/answer.png" alt="query color" width="17" height="17" style="vertical-align:middle">.<sup>2</sup>

To store and search vectors, one can either use a database- or library-based solution.
Vector libraries store vector embeddings in (transient) in-memory structures in order to perform search as efficiently as possible and they tend to be a bit messy to maintain. Vector databases generally trade speed for persistance, flexibility and maintainability. Which solution to choose depends, as always, on the particular case at hand, but fundamentally they perform the same task – store and search semantic vectors.

In the following we'll take a closer look at:

- Libraries (in-memory)
    - faiss <https://github.com/facebookresearch/faiss>
- Databases
    - Qdrant <https://qdrant.tech/qdrant-vector-database/>

For a comparison between the major vector database alternatives have a look at e.g. <https://benchmark.vectorview.ai/vectordbs.html>

---
<p><small>2. While the above example might look simple and intuitive, be warned that the ${\textrm L}_2$-norm behaves quite differently in ${\mathbb R}^n$ for $n>3$ than our experience from $n=2$ and $n=3$ lead us to believe. For reasons we're not going into here, ${\textrm L}_1$ or ${\textrm L}_\infty$ could be good options, and most vector stores offer alternative target metrics, typically defaulting to _cosine similarity_ <https://en.wikipedia.org/wiki/Cosine_similarity>.</small></p>

---
"""
# %% [markdown]
"""
## 5. Retrieval

Retrieval is the process of gathering contextual information, given a user query, to augment the LLM prompt in the hope of getting a more accurate answer.

The first step is to encode the plain text query into a semantic vector _using the same model_ as was used to create the embeddings.
The query embedding lets us search the set of semantic vectors for vectors (and thus chunks/sentences) that are semantically close to the query. We'll sanity check each example by requesting the two most relevant (according to the vectorizing model) sentences, using a query we know for a fact is part of the set. it will look something like:

```
print(sentences[5])
> How is a user's karma calculated?

query = sentences[5]
query_embedding = model.encode([query])
distances, indices = store.search(query_embedding, 2)
for idx, l2 in zip(indices[0], distances[0]):
    print(f"{idx}: {sentences[idx]}, ({l2:.4f})")
> 5: How is a user's karma calculated?, (0.0000)
> 9: Do posts by users with more karma rank higher?, (0.1439)
```

Let's try out some examples (they are meant to be studied in order).
"""
# %% [markdown]
"""
### Example 1: Faiss (in memory)

Let's start with faiss, a FOSS vector library, and create storage for our embeddings. Faiss can do a lot of tricks, but we'll take the safe option to make it perform an _exact_ search of our embeddings. By using a `Flat` option we get exact search, and at this point we also specify how distance is calculated (`IndexFlatL2` => exact search using Euclidean distance, see <https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlat.html>)
"""
# %%
import faiss

faiss_store = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
faiss_store.add(embeddings)
# %% [markdown]
"""
Once the index is built, it is no longer possible to change it, but OTOH querying is very fast (not that it matters for our ~570 vectors).

First the sanity check:
"""
# %%
print(sentences[5])

# %%
query = sentences[5]
query_embedding = model.encode([query])
# Return the two closest matches and their similarity scores (in parenthesis)
k = 2
distances, indices = faiss_store.search(query_embedding, k)
for idx, l2 in zip(indices[0], distances[0]):
    print(f"{idx}: {sentences[idx]}, ({l2:.4f})")

# %% [markdown]
"""
With the sanity check done, let's try with the more open question that we used in the baseline test, and retrieve some relevant context for the LLM query in the next section.
"""

# %%
query = 'What is special about HackerNews?'
query_embedding = model.encode([query])

# %%
# Number of matches for search to return
k = 30
_, indices = faiss_store.search(query_embedding, k)
# Context for LLM
example_name = "faiss"
context = '\n'.join([f'{i}. {sentences[idx]}' for i, idx in enumerate(indices[0])])

# %%
print(context)

# %% [markdown]
"""
#### hnswlib

Another FOSS in-memory vector store is [hnswlib](https://github.com/nmslib/hnswlib/blob/master/README.md) which performs an _approximative_ nearest-neighbour search of the embeddings. The name is derived from the search algorithm Hierarcical Navigable Small Worlds, see e.g. <https://www.pinecone.io/learn/series/faiss/hnsw/> for a good introduction.
HNSW can be used in faiss, which has many search strategies, but hnswlib is [claimed to be faster](https://ann-benchmarks.com) as it is a one-trick-pony.

From an API-perspective the biggest difference to faiss in this simple example is the order of the return values in a search - `(indices, distances)`instead of `(distances, indices)`, and that the documentation refer to indices as `labels`.
"""
# %% [markdown]
"""
### Example 2: Qdrant (database)

For this example we'll use a [Qdrant](https://qdrant.tech/qdrant-vector-database/), a database service running on the same server as Ollama.
A database is slightly more verbose to work with compared to faiss/hnswlib, but typically has features such as ability to update embeddings, store (any kind of) objects with an embedding, and embed data (or in some cases entire documents) on-the-fly when adding data. Here we'll stick to just storing and searching the generated embeddings as in the previous example.

[Documentation for Qdrant client](https://python-client.qdrant.tech).

First, let's set up a space for our toy example and upload the embeddings:
"""
# %%
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Create a client connecting to the service
qdrant_store = QdrantClient(host="10.129.20.4", port=6333)

# Create a named _collection_ making up our corner of the database (it is a shared resource)
collection_name = "hacker_news"

# Check if collection (for this toy example) already exist, and remove if so
if qdrant_store.collection_exists(collection_name=collection_name):
   qdrant_store.delete_collection(collection_name=collection_name)

# Create a named collection and set vector dimension and metric (EUCLID => L2)
qdrant_store.create_collection(
    collection_name = collection_name,
    vectors_config = VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.EUCLID),
)

# Upload our embeddings, one variant of many

# If ids are _not_ provided, Qdrant Client will replce them with random UUIDs (not good in this case).
# Optional _payload_ not utilized, could in this example be e.g. the URL associated with each embedding
n = len(embeddings)
qdrant_store.upload_collection(
    collection_name = collection_name,
    ids = range(n),
    vectors = embeddings,
)

# %% [markdown]
"""
First the sanity check (note that qdrant's similarity search is _guided_ by `SearchParams`, see <https://qdrant.tech/documentation/concepts/search/>, and here we stay with an approximate hnsw search):
"""

# %%
print(sentences[5])

# %%
from qdrant_client import models

query = sentences[5]
query_embedding = model.encode(query)
# Return the two closest matches
search_results = qdrant_store.search(
    collection_name = collection_name,
    search_params = models.SearchParams(hnsw_ef=10, exact=False),
    query_vector = query_embedding,
    limit = 2,
)

ids_and_dists = [(result.id, result.score) for result in search_results]
for idx, l2 in ids_and_dists:
    print(f"{idx}: {sentences[idx]}, ({l2:.4f})")

# %% [markdown]
"""
Comparing to the baseline test:
"""

# %%
query = 'What is special about HackerNews?'
query_embedding = model.encode(query)


# %%
# Number of matches for search to return
k = 30
search_results = qdrant_store.search(
    collection_name = collection_name,
    search_params = models.SearchParams(hnsw_ef= 50, exact=False),
    query_vector = query_embedding,
    limit = k,
)

indices = [res.id for res in search_results]

# Context for LLM
example_name = "qdrant"
context = '\n'.join([f'{i}. {sentences[idx]}' for i, idx in enumerate(indices)])

# %%
print(context)

# %% [markdown]
"""
## 6. Prompt preparation (Augmented)
"""

# %% [markdown]
"""
### Prompt format
"""

# %% [markdown]
"""
The best prompt format for RAG-augmentation is unclear as Llama3 doesn't have a specific prompt-format, unlike eg anthropic <https://github.com/meta-llama/llama-recipes/issues/450>. We'll keep it simple use the following prompt template:
"""

# %%
base_prompt = """You are an AI assistant. Your task is to understand the user question, and provide an answer using the provided context.

Your answers are short, to the point, and written by an domain expert. Provide references to the context where appropriate.
If the provided context does not contain the answer, simply state, "The provided context does not have the answer."

User question: {}

Context:
{}
"""

# %%
prompt = f'{base_prompt.format(query, context)}'

# %% [markdown]
"""
Uncomment next line if you want to see what gets fed into the LLM 
"""

# %%
# prompt

# %% [markdown]
"""
## 7. Answer Generation
"""
# %%
stream = client.chat(
    model=ollama_model,
    messages=[{'role': 'user', 'content': prompt}],
    stream=True,
)

print(f"Using embeddings generated by '{model_name}' retrieved from '{example_name}':\n")
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
