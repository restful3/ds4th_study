# AI Flashcards

## Card 1

**Front:** What is the primary role of Knowledge Graphs (KGs) in addressing the 'hallucination' limitation of Large Language Models (LLMs)?

**Back:** KGs provide a factual foundation of structured, verified knowledge to ground LLM responses.

---

## Card 2

**Front:** Which technology provides the natural language understanding (NLU) capabilities necessary to make complex knowledge structures accessible?

**Back:** Large Language Models (LLMs).

---

## Card 3

**Front:** According to the text, what defines the 'killer combination' of KGs and LLMs regarding graph construction?

**Back:** LLMs can extract entities and relationships from unstructured text to build KGs more efficiently.

---

## Card 4

**Front:** In a Knowledge Graph, what do 'nodes' represent?

**Back:** Real-world entities such as people, places, diseases, or proteins.

---

## Card 5

**Front:** In a Knowledge Graph, what is the function of 'relationships'?

**Back:** They define meaningful connections between nodes, such as a person born in a specific place.

---

## Card 6

**Front:** What do 'properties' add to the entities and relationships within a Knowledge Graph?

**Back:** Contextual details such as birth dates, geographic coordinates, or organizational history.

---

## Card 7

**Front:** What is a major barrier to the wide adoption of Knowledge Graphs related to their creation?

**Back:** They are expensive to build and maintain in terms of time, effort, and money.

---

## Card 8

**Front:** Why is 'coreference resolution' vital for accurate text comprehension in KG construction?

**Back:** It clarifies what a pronoun refers to, ensuring the correct entity is linked to a relationship.

---

## Card 9

**Front:** How does an LLM assist in 'querying' a Knowledge Graph?

**Back:** By extracting precise information from natural language questions to support graph search and navigation.

---

## Card 10

**Front:** What is 'Transfer Learning' in the context of LLMs?

**Back:** The ability to reuse patterns learned in generic tasks for specific tasks like relation extraction.

---

## Card 11

**Front:** What architecture is the foundation of modern Pretrained Language Models (PLMs)?

**Back:** Transformer architectures.

---

## Card 12

**Front:** What is the typical parameter scale for a model to be considered a 'Large Language Model'?

**Back:** Tens or hundreds of billions of parameters.

---

## Card 13

**Front:** According to 'Scaling Laws for Neural Language Models', how does model size affect data requirements?

**Back:** Larger neural models require fewer data samples to reach the same performance level.

---

## Card 14

**Front:** What is the primary difference between a model-centric paradigm and a data-centric paradigm?

**Back:** The model-centric paradigm focuses on architecture/hyperparameters, while the data-centric paradigm prioritizes the quality and quantity of training data.

---

## Card 15

**Front:** What is the purpose of 'tokenization' in LLM building blocks?

**Back:** It breaks text into smaller units called tokens to simplify text representation and improve processing efficiency.

---

## Card 16

**Front:** How do 'high-dimensional embeddings' represent tokens in an LLM?

**Back:** As continuous vectors that capture semantic meanings and encode relationships among tokens.

---

## Card 17

**Front:** What mechanism allows transformers to focus on the most relevant parts of an input sentence?

**Back:** Attention mechanisms.

---

## Card 18

**Front:** What is 'GraphRAG'?

**Back:** A system that organizes knowledge into clusters and community summaries to provide LLMs with current information from updated KGs.

---

## Card 19

**Front:** How do KGs enable explainability in AI systems?

**Back:** They provide transparent information paths and structured reasoning that users can trace and validate.

---

## Card 20

**Front:** Which KG pillar emphasizes the ability to continuously ingest and unify information without overhauling the existing structure?

**Back:** Evolution.

---

## Card 21

**Front:** Which KG pillar makes the meaning of data explicit through typed entities and relationships?

**Back:** Semantics.

---

## Card 22

**Front:** Which KG pillar serves as a central reference connecting information from multiple disparate data sources?

**Back:** Integration.

---

## Card 23

**Front:** Which KG pillar involves performing machine learning algorithms to infer information not explicitly encoded in the graph?

**Back:** Learning.

---

## Card 24

**Front:** In the Learning pillar of KGs, what is the goal of 'community analysis'?

**Back:** To recognize groups of similar nodes within the network.

---

## Card 25

**Front:** In the context of drug discovery, what role do LLMs play regarding scientific publications?

**Back:** They process unstructured data to ensure consistency and infer potential relationships between biological entities.

---

## Card 26

**Front:** Why might a conversational AI provide uninformative or repetitive answers without KG grounding?

**Back:** It lacks the specific domain context and structured knowledge to deepen the conversation beyond general expertise.

---

## Card 27

**Front:** Which KG paradigm focuses on knowledge representation as a collection of statements or 'triplets'?

**Back:** Resource Description Framework (RDF).

---

## Card 28

**Front:** Which KG paradigm focuses on graph structure and provides advantages in pathfinding and traversal operations?

**Back:** Labeled Property Graphs (LPG).

---

## Card 29

**Front:** What is the standard query language for the Resource Description Framework (RDF)?

**Back:** SPARQL.

---

## Card 30

**Front:** How do taxonomies organize data categories?

**Back:** In a hierarchical dimension using broader-narrower relationships.

---

## Card 31

**Front:** How do ontologies enhance a KG beyond simple hierarchies?

**Back:** They define complex relationships such as identity, synonyms, disjointness, and cardinality restrictions.

---

## Card 32

**Front:** What is the 'just enough semantics' approach in modern KGs?

**Back:** Selecting a subset of ontology features to address current issues without being overly prescriptive or rigid.

---

## Card 33

**Front:** According to McKinsey & Company, how much can addressing data fragmentation reduce annual data spending in the short term?

**Back:** $5\%$ to $15\%$.

---

## Card 34

**Front:** What is the 'third wave of AI' intended to support?

**Back:** Mission-critical applications requiring contextual information like social norms and environmental characteristics.

---

## Card 35

**Front:** Concept: Zero-shot/Few-shot tasking

**Back:** Definition: The ability of an LLM to serve multiple purposes just by changing the prompt instructions without extensive model engineering.

---

## Card 36

**Front:** Process: Creating a 'text-to-cypher' translation

**Back:** Purpose: To allow natural language questions to be converted into precise graph queries for reliable information extraction.

---

## Card 37

**Front:** In the healthcare domain, what relationship connects a 'disease' node and a 'drug' node in a typical KG?

**Back:** TREAT or USED FOR.

---

## Card 38

**Front:** Why is the LPG model considered advantageous for 'graph traversal'?

**Back:** Because each edge has a unique identity and properties, facilitating efficient pathfinding.

---

## Card 39

**Front:** In RDF, how are relationships treated differently than in LPG?

**Back:** Relationships in RDF are global predicates that can be reused across statements throughout the knowledge base.

---

## Card 40

**Front:** What does the 'Evolution' pillar of KGs allow a system to incorporate seamlessly?

**Back:** New interactions or content without needing a complete overhaul of the existing structure.

---

## Card 41

**Front:** Which logic-based feature of an ontology specifies that two categories cannot be the same?

**Back:** Disjointness.

---

## Card 42

**Front:** What is the benefit of the 'business-focused' mindset in building data-driven applications?

**Back:** It prioritizes business goals first, then data, and finally algorithms.

---

## Card 43

**Front:** How does the 'Learning' pillar utilize centrality analysis?

**Back:** To identify influential nodes within the graph structure.

---

## Card 44

**Front:** What does the term 'PLM' stand for in the context of LLM development?

**Back:** Pretrained Language Model.

---

## Card 45

**Front:** What specific challenge is mentioned regarding pronouns like 'he' in sentences like 'John saw Bob and he waved'?

**Back:** Ambiguity in coreference resolution (uncertainty if 'he' refers to John or Bob).

---

## Card 46

**Front:** How do LLMs contribute to the 'Summarizing' phase of a KG-based solution?

**Back:** By transforming the raw results of a graph query into a simple, natural language summary.

---

## Card 47

**Front:** What does the integration of LLMs and KGs enable regarding 'unstructured data'?

**Back:** It allows systems to handle and interpret text-based data while maintaining the accuracy of a structured database.

---

## Card 48

**Front:** Which query language is associated with the 'TinkerPop' framework?

**Back:** Gremlin.

---

## Card 49

**Front:** In the definition of a Knowledge Graph, what does it mean that the structure is 'ever-evolving'?

**Back:** The graph can be continuously updated with new entities, attributes, and relationships as domain knowledge grows.

---

## Card 50

**Front:** What is the primary goal of the 'Integration' pillar for an analyst?

**Back:** To overcome challenges related to data types, formats, and provenance by connecting information across multiple sources.

---

## Card 51

**Front:** Why are predefined queries sometimes considered a limitation for Knowledge Graph systems?

**Back:** They limit the types of users and the flexibility of the questions the system can answer.

---

## Card 52

**Front:** According to the source, what is the result of applying 'Transfer Learning' to NLP tasks?

**Back:** A shift from training many small, task-specific models to using a few large, reusable models.

---

## Card 53

**Front:** What is the 'unreasonable effectiveness' factor related to the corpus in GPT models?

**Back:** Both the massive size and the high quality of the training data.

---

## Card 54

**Front:** In KG terms, what is the 'hierarchical dimension' of data called?

**Back:** Taxonomy.

---

## Card 55

**Front:** What term describes a KG that integrates multiple medical specialties like oncology or cardiology as needed?

**Back:** Pragmatic or 'organic' ontology expansion.

---

## Card 56

**Front:** How does a Knowledge Graph ground a conversational AI?

**Back:** By providing the concepts and meaningful relationships that drive the conversational flow and background context.

---

## Card 57

**Front:** What type of data source includes relational databases or files in CSV and JSON formats?

**Back:** Structured or semi-structured data.

---

## Card 58

**Front:** What is the 'first step' in building a KG from heterogeneous sources?

**Back:** Recognizing and extracting relevant entities and connections.

---

## Card 59

**Front:** Why are Knowledge Graphs often difficult for 'non-experts' to interpret?

**Back:** Results may be scattered across multiple nodes and relationships, requiring complex querying and specific interfaces.

---

## Card 60

**Front:** What allows LLMs to mimic human-like language after being trained on vast datasets?

**Back:** Generation capacity.

---

## Card 61

**Front:** Concept: Network Analysis

**Back:** Definition: A method used in the Learning pillar to detect the shortest path between nodes in a Knowledge Graph.

---

## Card 62

**Front:** Which organization defines the standards for RDF and SPARQL?

**Back:** The World Wide Web Consortium (W3C).

---

## Card 63

**Front:** What is the specific utility of 'GraphRAG' over standard RAG?

**Back:** It uses meaningful clusters and community summaries to provide more contextually accurate and up-to-date information.

---

## Card 64

**Front:** In the Drug Discovery use case, how do 'typed relationships' help?

**Back:** They represent domain meaning better, enabling transitive bonds and logical inference.

---

## Card 65

**Front:** What 'paradigm shift' occurs when intelligent behavior is encoded in a unique source of truth?

**Back:** It allows a single knowledge base to empower various applications and diverse tasks consistently.

---

## Card 66

**Front:** How does an LLM help with the 'multiple languages' challenge of unstructured data?

**Back:** By understanding grammatical rules, vocabulary, and nuances across different writing systems and scripts.

---

## Card 67

**Front:** What is the 'core abstraction' for incorporating human knowledge into machines according to the text summary?

**Back:** Knowledge Graphs (KGs).

---

## Card 68

**Front:** What is the benefit of 'parallel processing' in transformer architectures?

**Back:** It allows for efficient training on massive datasets, leading to high-performance language models.

---

## Card 69

**Front:** In a Knowledge Graph, what is a 'disjoint' relationship in an ontology?

**Back:** A rule specifying that two classes, such as 'Car' and 'Bicycle', cannot contain the same entity.

---

## Card 70

**Front:** What is the role of 'inference rules' in the Learning pillar of a Knowledge Graph?

**Back:** To derive new information that is not explicitly encoded within the existing graph data.

---

## Card 71

**Front:** What is the 'starting point' for the four pillars of Knowledge Graphs in the provided diagram?

**Back:** The integration of Evolution and Semantics feeding into a central Knowledge Graph.

---

## Card 72

**Front:** How does the 'Integration' pillar address data provenance?

**Back:** By focusing on the meaning of data, allowing users to connect information despite differences in provenance or format.

---

## Card 73

**Front:** What is the primary 'drawback' of Labeled Property Graphs (LPG) compared to RDF?

**Back:** They may lack the web-scale interoperability and standardized semantic consistency provided by RDF's global predicates.

---

## Card 74

**Front:** What allows an LLM to serve multiple purposes without retraining?

**Back:** Prompt configuration (instructions provided by the user).

---

## Card 75

**Front:** Why is 'data quality' considered equally crucial to 'corpus size' in LLM training?

**Back:** Faulty or low-quality data can lead to faulty predictions and lower model accuracy.

---

## Card 76

**Front:** What defines 'Mission-Critical' AI applications according to the text?

**Back:** Applications where domain knowledge, high accuracy, and explainability are essential.

---

## Card 77

**Front:** What does a 'directed, labeled edge' in a healthcare KG represent?

**Back:** A semantic relationship between two entities, such as 'CAUSES' or 'AFFECTS'.

---

## Card 78

**Front:** What is the main benefit of 'Transfer Learning' for model developers?

**Back:** It significantly reduces the training data and computational resources required compared to training from scratch.

---

## Card 79

**Front:** In the LPG approach, what distinguishes edges from those in the RDF approach?

**Back:** Each edge in LPG has a unique identity and its own properties.

---

## Card 80

**Front:** Which logic restriction in ontologies limits the number of relationships an entity can have?

**Back:** Cardinality restrictions.

---

## Card 81

**Front:** What is the 'pragmatic approach' to building KGs in healthcare?

**Back:** Focusing on a specific medical domain first and expanding organically rather than enforcing a rigid, complete taxonomy.

---

## Card 82

**Front:** What is 'text-to-cypher' translation?

**Back:** The process of converting natural language questions into precise graph query language commands.

---

## Card 83

**Front:** In the Customer Support use case, how are 'concepts' connected to ground conversations?

**Back:** Through the Knowledge Graph, which establishes meaningful relationships to drive the conversational flow.

---
