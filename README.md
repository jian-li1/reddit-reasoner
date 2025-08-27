# Reddit Reasoner
Reddit Reasoner aims to excel at domain-specific question-answering based on conversations from a specific subreddit, or a Reddit forum. Using Retrieval-augmented Fine-tuning (RAFT), the model learns to filter out irrelevant discussion threads and identify the thread that is most relevant to the query. The Chain-of-Thought reasoning allows the model to interpret the thread by reviewing each message and highlighting specific details that address the query. In the case where all retrieved threads are irrelevant to the query, the model can still provide an answer using knowledge that was internalized during fine-tuning.

This project is still in the early development phase, so more documentation will be added here soon.
