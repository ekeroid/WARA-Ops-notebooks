# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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

# %% [markdown]
"""
# RAG From Scratch – a Three Part Tutorial

Retrieval Augmented Generation (RAG) is intended to alleviate some of the most obvious [problems][1] displayed by Large Language Models (LLMs), such as hallucination (where an LLM starts making up "facts" and static them as such) and not being able to reference the source for its claims. RAG consists of three steps; *retrieval* of context-specific information, *augmenting* (i.e. adding context to) the LLM [prompt][2], and letting the LLM generate an answer taking that context into account, providing references to the context.

The tutorials build upon the previous ones, so even if you know the subject of tutorial it is recommended that you browse through it rather than skip it.

1. [A bare-bones RAG implementation][3] using as little magic as possible to explore the fundamental concepts using an illustrative toy example
2. [RAG with LangChain][4] utilizes the popular LangChain environment to implement the same toy example as in the previous tutorial.
3. [A RAG-enhanced chatbot][5] builds the core of a fully functional chatbot that get help from RAG to stay on track. 

[1]: https://youtu.be/T-D1OfcDW1M?si=nKf8KC93tcsbbAlO
[2]: https://medium.com/thedeephub/llm-prompt-engineering-for-beginners-what-it-is-and-how-to-get-started-0c1b483d5d4f
[3]: hn+scratch-draft.ipynb
[4]: hn+langchain.ipynb
[5]: chatbot+RAG.ipynb
 
"""

