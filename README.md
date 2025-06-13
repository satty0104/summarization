 We are working on improving the accuracy(rouge score) of abstractive summaries of articles by leveraging the LLMs.

The approach uses keyword-guided abstractive summarization, which works by extracting keywords from a text. By feeding the article and keywords to an LLM to generate better summaries, we have fine-tuned some LLMs on our custom dataset.

The keywords extraction logic contains a mixture of techniques like TF-Idf, uses NER(Name Entity Recognition), Keybert, etc.

The project is ongoing
