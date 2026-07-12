# Agent Flashcards

## Card 1

**Front:** What is the primary characteristic that distinguishes AI agents from traditional software programs?

**Back:** AI agents exhibit autonomy and adaptability rather than following a fixed set of instructions.

---

## Card 2

**Front:** What are the three essential components required to build a basic conversational AI agent?

**Back:** A pretrained Large Language Model (LLM), a system message for scope, and memory for message history.

---

## Card 3

**Front:** In the context of LLMs, what does the term 'hallucinations' refer to?

**Back:** The generation of plausible-sounding but factually fabricated or inaccurate information.

---

## Card 4

**Front:** What is the 'knowledge cutoff' of an LLM?

**Back:** The fixed point in time after which the model has no training data regarding real-world events.

---

## Card 5

**Front:** How does the 'transparency' issue affect LLM deployment in enterprise solutions?

**Back:** Models provide answers without insight into their reasoning processes or the reliability of their sources.

---

## Card 6

**Front:** What is the core purpose of Retrieval-Augmented Generation (RAG)?

**Back:** To ground LLM responses in external, accurate, and up-to-date context retrieved from a data source.

---

## Card 7

**Front:** How does RAG function as a 'grounding' technique for AI models?

**Back:** It limits the scope of answer generation to the provided external context, reducing the likelihood of hallucinations.

---

## Card 8

**Front:** In vector-based RAG, what are 'embeddings'?

**Back:** Fixed-length vector representations of text chunks that capture their semantic meaning.

---

## Card 9

**Front:** How are text documents typically prepared for indexing in a vector database?

**Back:** They are chunked into smaller portions and converted into dense vector embeddings.

---

## Card 10

**Front:** What is 'context fragmentation' in the context of vector-based RAG?

**Back:** The loss of complex relationships between documents due to treating text chunks independently.

---

## Card 11

**Front:** Why might vector search return 'noise' in a RAG system?

**Back:** Similarity search may retrieve documents that are linguistically similar but semantically irrelevant to the specific query.

---

## Card 12

**Front:** What is a 'miss in retrieval' regarding vector-based RAG?

**Back:** The failure to include the most relevant documents in the context due to the limitations of approximate similarity search.

---

## Card 13

**Front:** How does Graph RAG mitigate the limitations of purely vector-based retrieval?

**Back:** It uses structured relational patterns and multi-hop connections within a Knowledge Graph to provide precise context.

---

## Card 14

**Front:** What defines a 'text-attributed graph' in a Knowledge Graph design?

**Back:** A graph where nodes and relationships contain specific textual attributes or properties.

---

## Card 15

**Front:** What is the defining feature of a 'text-paired graph'?

**Back:** Nodes and relationships are explicitly associated with the documents from which they were extracted.

---

## Card 16

**Front:** In a Graph RAG system, what is the role of a 'KG retriever' tool?

**Back:** It uses the Knowledge Graph schema to generate queries and retrieve structured, condensed information.

---

## Card 17

**Front:** What is the primary function of a 'KG-enhanced document retriever'?

**Back:** It identifies specific documents based on relationships or entities found in the Knowledge Graph rather than semantic similarity.

---

## Card 18

**Front:** How does 'combined retrieval' handle questions spanning multiple data sources?

**Back:** An agent uses a Knowledge Graph to identify specific names or entities and then uses those to search a document database.

---

## Card 19

**Front:** What is the 'ReAct' framework for AI agents?

**Back:** A method that integrates reasoning and acting capabilities by using an iterative feedback loop.

---

## Card 20

**Front:** What are the three repeating steps in a ReAct agent's problem-solving loop?

**Back:** Thought, Action, and Observation.

---

## Card 21

**Front:** Why is well-written tool description critical when designing an AI agent?

**Back:** It helps the model accurately determine which specific tool to execute for a given task or question.

---

## Card 22

**Front:** How do Knowledge Graphs facilitate answer generation for aggregate questions across multiple documents?

**Back:** They 'connect the dots' by linking information shared between disparate documents into a single structured network.

---

## Card 23

**Front:** What is a 'precanned' Cypher query in a Graph RAG implementation?

**Back:** A pre-written, parameterized database query used for frequently repeated types of questions to ensure reliability.

---

## Card 24

**Front:** What is the 'distraction phenomenon' in LLM response generation?

**Back:** When irrelevant or noisy information in the provided context confuses the model and degrades output quality.

---

## Card 25

**Front:** In a multi-agent system, how would a 'Reviewer' agent function?

**Back:** It evaluates the output of a 'Writer' agent to ensure quality and suggest improvements before publication.

---

## Card 26

**Front:** Why are vector search approaches computationally expensive for larger corpora?

**Back:** Comparing a query embedding against millions of high-dimensional document vectors requires significant processing power.

---

## Card 27

**Front:** How does the use of Knowledge Graphs improve user confidence in AI systems?

**Back:** They provide a human-accessible format that allows domain experts to validate the data grounding the AI's answers.

---

## Card 28

**Front:** What is the main risk of relying solely on the probabilistic nature of LLMs?

**Back:** The models are designed to predict the next likely token, which can lead them 'astray' even with provided context.

---

## Card 29

**Front:** Which tool acts as a 'backup' in the illustrated Graph RAG agent design?

**Back:** The semantic retriever (vector search tool).

---

## Card 30

**Front:** What is the benefit of adding a 'self-correction loop' to a Cypher-generating tool?

**Back:** It allows the LLM to double-check and fix syntax errors in its own generated database queries.

---

## Card 31

**Front:** How does RAG address the 'data privacy' concerns of using LLMs?

**Back:** It allows organizations to use private data as context without needing to retrain the model on sensitive information.

---

## Card 32

**Front:** What is the purpose of 'document re-ranking' after initial context selection?

**Back:** To improve the relevance of retrieved documents and limit the size of the final context window.

---

## Card 33

**Front:** In the Rockefeller Archive KG example, what does a 'TALKED_ABOUT' relationship represent?

**Back:** An interaction where one person mentions or discusses another person, as extracted from historical documents.

---

## Card 34

**Front:** Why is 'chunking' a critical part of vector-based RAG design?

**Back:** It determines the size and granularity of text represented by a single embedding, directly impacting retrieval accuracy.

---

## Card 35

**Front:** How does a 'KG retriever' handle queries that don't require full original texts?

**Back:** It returns structured node and relationship data directly from the Knowledge Graph to answer facts or totals.

---

## Card 36

**Front:** What defines the 'brain' of an AI agent in modern intelligent systems?

**Back:** A pretrained Large Language Model (LLM) that processes inputs and plans tasks.

---

## Card 37

**Front:** What is the limitation of 'static' precomputed vectors in a changing knowledge environment?

**Back:** They are less adaptive to new nomenclature or evolving information compared to dynamic retrieval methods.

---

## Card 38

**Front:** In a ReAct agent, what does the 'Observation' step provide?

**Back:** Real-time outcomes from a tool's execution that the agent uses to refine its next plan.

---

## Card 39

**Front:** Why might a vector similarity search fail to retrieve a document mentioning a specific entity name?

**Back:** Embeddings encode overall semantic meaning, which might prioritize a different linguistic pattern over a literal name match.

---

## Card 40

**Front:** How can a Knowledge Graph track provenance in a RAG system?

**Back:** By using a text-paired graph design where every graph element is linked back to its originating document source.

---

## Card 41

**Front:** In the context of RAG, what does the term 'probabilistic' imply about model outputs?

**Back:** The model calculates the most probable next word rather than retrieving a deterministic, hard-coded fact.

---

## Card 42

**Front:** What is the purpose of using an '.env' file in a Python-based AI agent script?

**Back:** To securely provide environment variables like API keys without hard-coding them into the script.

---

## Card 43

**Front:** How do 'multi-agent systems' emulate human workflow?

**Back:** Different agents assume specialized roles and communicate to achieve a complex goal, much like a team of people.

---

## Card 44

**Front:** What is the primary cost-related concern regarding high-end LLMs?

**Back:** The immense computational power required for training leads to high financial expenses and a large carbon footprint.

---

## Card 45

**Front:** How does a KG-powered RAG system reduce 'misses in retrieval'?

**Back:** By using structured entity links to find all related documents, regardless of whether their embedding scores are the highest.

---

## Card 46

**Front:** In Graph RAG, why is 'metadata' (like publication date) stored in the KG?

**Back:** It allows the system to filter or select documents based on criteria like the most recent version or specific authors.

---

## Card 47

**Front:** What defines the 'autonomy' of an AI agent?

**Back:** The ability of the entity to make decisions and respond to changing conditions in real time without human intervention.

---

## Card 48

**Front:** What is the benefit of identifying 'communities' in a Knowledge Graph for RAG?

**Back:** Each community can be represented by a summary, providing high-level context that spans multiple documents.

---

## Card 49

**Front:** In the ReAct loop, what is the role of 'reasoning'?

**Back:** It allows the agent to evaluate if the obtained context is sufficient to answer the original user question.

---

## Card 50

**Front:** What is the main limitation of using 'approximate' vector similarity algorithms?

**Back:** They sacrifice retrieval accuracy for faster performance in large-scale databases.

---

## Card 51

**Front:** How does 'sparsity' in an embedding model's training data affect retrieval?

**Back:** It leads to inaccurate vector representations for terms that were underrepresented during training.

---

## Card 52

**Front:** What does a 'KG document retriever' do that a 'Semantic retriever' cannot?

**Back:** It retrieves documents based on specific, high-confidence relationships defined in a structured schema.

---

## Card 53

**Front:** Why is the Rockefeller Foundation grant process a good use case for Graph RAG?

**Back:** It involves a complex influence network with proprietary data that isn't available in public LLM training sets.

---

## Card 54

**Front:** What characterizes an 'enterprise-grade' AI application according to the text?

**Back:** A system that is useful in production scenarios, often requiring agents, RAG, and LLMOps.

---

## Card 55

**Front:** In the code examples, which library is used to build the AI-based systems?

**Back:** LangChain.

---

## Card 56

**Front:** What is the relationship between 'cosine similarity' and vector-based RAG?

**Back:** It is a mathematical measure used to rank document chunks by their semantic proximity to the user's question.

---

## Card 57

**Front:** How does a Knowledge Graph improve the 'transparency' of a RAG system?

**Back:** It makes information sources and reasoning processes accessible and traceable for human users.

---

## Card 58

**Front:** What is the role of 'prompt engineering' in building AI agents?

**Back:** It involves crafting instructions that guide the LLM's behavior, tool usage, and decision-making logic.

---

## Card 59

**Front:** How does the ReAct framework handle 'complex environments'?

**Back:** By iteratively adjusting its plan based on observations from intermediate steps.

---

## Card 60

**Front:** What is the risk of using a 'simple chunking strategy' in vector RAG?

**Back:** Relevant information might be split between chunks, making it difficult for the retriever to provide a complete context.

---

## Card 61

**Front:** How does a Knowledge Graph act as a 'central knowledge repository'?

**Back:** It integrates raw texts, metadata, and structured data sources like CSVs or ontologies into a unified format.

---

## Card 62

**Front:** In the provided context, what does LLMOps stand for?

**Back:** LLM operations, referring to the implementation and maintenance of LLMs in production.

---

## Card 63

**Front:** How do specialized smaller LLMs compare to large models regarding 'cost'?

**Back:** They are reducing the expense of AI deployment while remaining effective for specific tasks.

---

## Card 64

**Front:** In the Graph RAG agent example, what happens if the Knowledge Graph tools fail?

**Back:** The agent falls back to a semantic retriever (vector search) to find relevant context.

---

## Card 65

**Front:** What is the purpose of 'binding' a model, prompt, and tools in LangChain?

**Back:** It creates a unified agent executor that can process user queries through a defined workflow.

---

## Card 66

**Front:** How does a 'text-paired graph' support the retrieval of relationship details?

**Back:** It allows the system to identify and pull the exact documents that describe a specific interaction between two entities.

---

## Card 67

**Front:** What is 'multihop reasoning' in the context of Knowledge Graphs?

**Back:** The ability to traverse multiple relationships (e.g., from Person A to Person B to Institution C) to answer a question.

---

## Card 68

**Front:** How does 'Graph RAG' improve the density of relevant information in the context?

**Back:** By selecting only the specific data points or documents directly linked to the entities in the query, reducing noise.

---

## Card 69

**Front:** What is a 'Reason and Act' agent's response to an unsatisfactory tool output?

**Back:** It uses reasoning to refine its plan and acts again using a different tool.

---

## Card 70

**Front:** Why is the 'human-in-the-loop' concept important for RAG systems?

**Back:** Human feedback and validation are essential to supervise probabilistic models and ensure the accuracy of high-stakes products.

---

## Card 71

**Front:** What is the significance of the year 2023 in the context of the source material?

**Back:** It was marked by 'AI upheaval' following the release of GPT-3.5, transforming the natural language processing domain.

---

## Card 72

**Front:** How can 'ontologies' be integrated into Graph RAG systems?

**Back:** They serve as structured data sources within the Knowledge Graph to provide high-confidence definitions and relationships.

---

## Card 73

**Front:** In the Dorothy M. Wrinch example, why was the final answer 'impeccable'?

**Back:** It was straight to the point, factual, and clearly grounded in the specific private documents retrieved.

---

## Card 74

**Front:** What is the advantage of using 'structured relational patterns' in information retrieval?

**Back:** It allows for higher precision and the ability to answer complex questions that span multiple sources.

---

## Card 75

**Front:** How does RAG help address 'ethical concerns and biases' in LLMs?

**Back:** By grounding responses in vetted, factual datasets rather than allowing the model to rely solely on potentially biased training data.

---
