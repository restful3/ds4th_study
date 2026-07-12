# Agentic Quiz

## Question 1
What is the primary characteristic that distinguishes AI agents from traditional software programs?

- [ ] The ability to follow a predetermined set of instructions without deviation.
- [x] Autonomy and the ability to make decisions based on real-time environmental changes.
- [ ] The requirement for a massive training dataset for every specific downstream task.
- [ ] A reliance on local processing rather than cloud-based APIs.

**Hint:** Consider how the system reacts when conditions change during execution.

## Question 2
Which of the following best describes the 'knowledge cutoff' limitation of LLMs?

- [ ] The maximum number of tokens a model can process in a single prompt.
- [ ] The inability of a model to access sensitive internal company data.
- [x] The fact that models cannot answer questions about events occurring after their training ended.
- [ ] The point at which a model begins to hallucinate due to lack of parameters.

**Hint:** Think about why a model might not know today's weather forecast.

## Question 3
How does Retrieval-Augmented Generation (RAG) serve as a 'grounding' technique for LLMs?

- [ ] It retrains the model on private data to ensure the internal weights are updated.
- [x] It limits the model's scope for answer generation to the provided external context.
- [ ] It reduces the financial cost of running the model by using smaller prompts.
- [ ] It encrypts the user's question to ensure data privacy before processing.

**Hint:** Focus on how adding specific documents to a prompt changes the model's behavior.

## Question 4
What is a significant limitation of vector-based RAG regarding semantic search?

- [ ] It can only handle structured data like CSVs and tables.
- [x] It may fail to retrieve a document even if it contains the exact keyword requested by the user.
- [ ] It is computationally cheaper than standard keyword search.
- [ ] It requires the LLM to have seen the documents during its original training.

**Hint:** Think about the example where 'Lauritsen' was missing from two of the top three retrieved documents.

## Question 5
What distinguishes a 'text-paired graph' from other graph structures in the context of Graph RAG?

- [x] Its nodes and relationships are directly associated with the source documents they originated from.
- [ ] It uses mathematical vectors instead of nodes and relationships.
- [ ] It is primarily used for identifying influencers in social networks.
- [ ] It consists solely of metadata like publication dates and authors.

**Hint:** Consider the traceability of information back to its original source.

## Question 6
In the ReAct (Reason and Act) framework, what is the purpose of the 'Thought' step?

- [ ] To execute the final API call to the LLM for the answer.
- [ ] To retrieve embeddings from the vector database.
- [x] To iteratively plan the next task or tool execution based on the current state of information.
- [ ] To store the conversation history in the agent's memory.

**Hint:** Think about the internal reasoning an agent does before choosing a tool.

## Question 7
Why are Knowledge Graphs (KGs) better than vector search for answering 'aggregate' questions that span multiple documents?

- [ ] KGs use less energy because they do not require GPU acceleration.
- [x] KGs can connect dots across documents through structured relationships, such as shared research topics.
- [ ] KGs automatically summarize all documents into a single dense vector.
- [ ] KGs are newer and more compatible with modern LLMs like GPT-4.

**Hint:** Consider the challenge of finding shared topics between two different universities mentioned in separate entries.

## Question 8
What is the 'distraction phenomenon' in the context of LLM retrieval?

- [ ] When a user asks too many follow-up questions, causing the model to lose focus.
- [x] When irrelevant or noisy documents in the context confuse the model and degrade the answer quality.
- [ ] When the agent chooses the wrong tool to answer a simple question.
- [ ] When a model provides a correct answer but in the wrong language.

**Hint:** Think about what happens when you provide an LLM with 10 pages of text when only one sentence is relevant.

## Question 9
In the provided Python implementation of a Graph RAG agent, what is the role of the 'KG-enhanced document retriever' tool?

- [ ] To act as a backup when the Knowledge Graph is offline.
- [x] To identify specific documents where a relationship between two known entities is discussed.
- [ ] To translate natural language questions into the Cypher query language automatically.
- [ ] To generate embeddings for new documents added to the database.

**Hint:** Recall how the agent answered 'What did person X say about person Y?'

## Question 10
Which of the following is an example of a 'text-attributed graph' property?

- [ ] A list of every document ID that mentions a specific person node.
- [x] A node representing a 'Person' with attributes like 'Name' and 'Bio'.
- [ ] The cosine similarity score between two document embeddings.
- [ ] A hard-coded Cypher query used for document selection.

**Hint:** Think about nodes and relationships having textual attributes.
