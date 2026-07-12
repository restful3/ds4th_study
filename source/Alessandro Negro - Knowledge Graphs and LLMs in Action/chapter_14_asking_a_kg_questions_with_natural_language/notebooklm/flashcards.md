# Graph Flashcards

## Card 1

**Front:** What is the core objective of the 'expert emulation' method in the context of Knowledge Graphs?

**Back:** To capture and apply the specialized information retrieval expertise of human professionals using LLMs.

---

## Card 2

**Front:** Which four pillars form the framework for an advanced question-answering system on KGs?

**Back:** Question routing, knowledge representation, expert-like reasoning for queries, and meaningful result presentation.

---

## Card 3

**Front:** In complex scenarios, what is a primary limitation of the standard Retrieval-Augmented Generation (RAG) approach?

**Back:** Fragility due to incomplete retrieval, where missing a single piece of context can lead to incorrect conclusions.

---

## Card 4

**Front:** How does the 'expert emulation' approach differ from traditional RAG regarding data access?

**Back:** It focuses on converting natural language into structured, formal queries (like Cypher) rather than just retrieving text passages.

---

## Card 5

**Front:** According to the policing example, why is a Knowledge Graph (KG) considered a 'single source of truth'?

**Back:** It structures and connects disparate roles, processes, and data sources into a unified, navigable system.

---

## Card 6

**Front:** Why is it beneficial for non-technical domain experts (like police analysts) to interact directly with KGs?

**Back:** It allows them to apply their unparalleled contextual knowledge to frame nuanced queries without technical bottlenecks.

---

## Card 7

**Front:** How does the utility of an LLM change for users who are already experts in their field?

**Back:** Its utility diminishes for basic inquiries and only becomes beneficial when it can expedite complex tasks like network analysis.

---

## Card 8

**Front:** The phenomenon where a RAG system provides an opposite conclusion because a specific document was not retrieved is known as _____.

**Back:** retrieval fragility

---

## Card 9

**Front:** In the witness statement example, what specific detail confirmed the suspect was likely left-handed?

**Back:** The observation of the perpetrator holding a phone to their ear with their left hand while walking.

---

## Card 10

**Front:** RAG systems perform best when information sources are divided into 'passages' of what specific characteristic?

**Back:** Fine-grained, independent pieces of information.

---

## Card 11

**Front:** What happens to the context provided to an LLM when a retriever fails to accurately assess relevance?

**Back:** The context becomes fragmented, potentially pushing the model toward incorrect assumptions.

---

## Card 12

**Front:** What acts as the 'blueprint' of a Knowledge Graph when an expert begins to construct a query?

**Back:** The graph schema.

---

## Card 13

**Front:** What is the first step an expert takes to convert a natural language request into a precise graph traversal?

**Back:** Mapping the concepts in the query to specific nodes and relationships in the graph schema.

---

## Card 14

**Front:** Concept: Constrained Traversal

**Back:** Definition: A graph navigation pattern refined by specific conditions (e.g., color, time, location) to pinpoint data.

---

## Card 15

**Front:** How does understanding a graph schema compare to navigating a city?

**Back:** It allows a user to identify where specific types of information are stored and how they are linked before searching.

---

## Card 16

**Front:** In the 'red Camaro' example, what schema node would the car model property belong to?

**Back:** The Vehicle (or Car) node.

---

## Card 17

**Front:** Which component of the system architecture determines how to handle a user's question by classifying it?

**Back:** Intent detection.

---

## Card 18

**Front:** What is the purpose of the 'Schema extraction' phase in the expert emulation pipeline?

**Back:** To transform the database schema into a format that LLMs can process and understand for query generation.

---

## Card 19

**Front:** Why is Intent Detection performed at the very beginning of the question-answering process?

**Back:** To identify the appropriate visualization type (e.g., map, graph) and route the request to the correct pipeline.

---

## Card 20

**Front:** Which visualization type is most appropriate for a user asking about the distribution of crimes over time?

**Back:** A chart.

---

## Card 21

**Front:** What is the default visualization 'catch-all' class for data queries in this system?

**Back:** The graph representation.

---

## Card 22

**Front:** A good classification prompt for intent detection should include _____ cases to help the model understand nuances.

**Back:** boundary

---

## Card 23

**Front:** Why is a 'reason' field included in the JSON output of the intent detection module?

**Back:** It serves as a tool for debugging and helps in understanding the LLM's classification logic.

---

## Card 24

**Front:** Into what three main categories can user interactions be grouped beyond simple data queries?

**Back:** Data-related questions, system-related questions, and feedback/complaints.

---

## Card 25

**Front:** What are the two subcategories of 'system-related questions' in the intent detection framework?

**Back:** Documentation-related and schema-related questions.

---

## Card 26

**Front:** Why might smaller or quantized LLMs fail at classifying questions about system update frequency?

**Back:** They lack sufficient context and may incorrectly classify system metadata questions as data-related queries.

---

## Card 27

**Front:** What is the primary difference between a 'technical schema' and a 'conceptual schema'?

**Back:** A technical schema includes administrative and helper nodes, while a conceptual schema focuses only on essential domain entities.

---

## Card 28

**Front:** Which Neo4j APOC function is used to compute the technical schema of a graph database?

**Back:** apoc.meta.schema

---

## Card 29

**Front:** Why does a conceptual schema reduce the 'cognitive load' for an LLM?

**Back:** It removes implementation-specific noise, allowing the model to focus on meaningful domain relationships.

---

## Card 30

**Front:** What are the two common methods for transitioning from a technical schema to a conceptual schema?

**Back:** Manually curating the schema or defining a 'skip list' to filter out unneeded technical elements.

---

## Card 31

**Front:** In a schema configuration file, what does the 'skip' section define?

**Back:** Classes, relationships, and properties that should be excluded from the LLM-ready schema representation.

---

## Card 32

**Front:** Why are 'descriptive annotations' necessary for properties like 'vehicle color' in a KG?

**Back:** To inform the LLM about specific data abbreviations, such as 'BLK' representing 'black'.

---

## Card 33

**Front:** How can relationship ambiguity (e.g., COMMITTED vs CO_OFFENDS) be resolved for an LLM?

**Back:** By providing annotations that explain the specific semantics and purpose of each relationship type.

---

## Card 34

**Front:** Which configuration format is recommended for managing schema skips and descriptions?

**Back:** YAML.

---

## Card 35

**Front:** What prompting technique introduces intermediate steps into a model's response to encourage calculated reasoning?

**Back:** Chain-of-thought prompting.

---

## Card 36

**Front:** What is the 'scratchpad' technique in LLM query generation?

**Back:** A method where the model produces intermediate 'workings' or reasoning tokens before generating the final output.

---

## Card 37

**Front:** Why should an LLM be prompted to provide its reasoning *before* its final answer?

**Back:** To prevent 'output consistency' bias where the model justifies a potentially incorrect answer it already committed to.

---

## Card 38

**Front:** The tendency for LLMs to generate reasoning that supports their initial, potentially biased tokens is called _____.

**Back:** semantic consistency

---

## Card 39

**Front:** How does the 'cumulative context' problem affect LLM token generation?

**Back:** Errors occurring early in the process propagate through subsequent tokens, reinforcing initial mistakes.

---

## Card 40

**Front:** In the Query Generation prompt, why is the user's question wrapped in HTML-like tags (e.g., <QUESTION>)?

**Back:** To help the model clearly identify the boundaries of the question and separate it from the instructions.

---

## Card 41

**Front:** What instruction is given to the LLM when the intended output type is 'graph' or 'map'?

**Back:** To return all matched nodes and relationships using named (not anonymous) relationship variables.

---

## Card 42

**Front:** How should an LLM handle a question that refers to a 'selected node' if the current selection is empty?

**Back:** It should set the 'success' flag to false and highlight the error in the output.

---

## Card 43

**Front:** Why is the user's question repeated at the end of a long instruction prompt?

**Back:** To reinforce the context and intent, compensating for the 'middle' tokens that might be weighted less heavily by the model.

---

## Card 44

**Front:** What is the purpose of the 'success' field in the JSON output of a query generator?

**Back:** To indicate whether a valid Cypher query following the schema could be successfully generated.

---

## Card 45

**Front:** Why is the LLM asked to list 'relationships to traverse' before writing the actual Cypher query?

**Back:** To reduce the risk of hallucinating relationship types that do not exist in the schema.

---

## Card 46

**Front:** How does the Response Summarization component bridge the gap for the user?

**Back:** It provides textual context and highlights key insights that might be hidden within node properties or graph structures.

---

## Card 47

**Front:** What is unique about the Summarization step compared to all previous steps in the pipeline?

**Back:** It is the only component that has access to the actual data records retrieved from the database.

---

## Card 48

**Front:** Why is it important to filter out 'irrelevant data' during the summarization phase?

**Back:** Graph queries often return extra nodes/paths for visualization that do not directly answer the user's specific question.

---

## Card 49

**Front:** In the summarization prompt, what does the 'results_analysis' flag indicate?

**Back:** Whether the user question requested an implicit or explicit analysis of the returned raw data.

---

## Card 50

**Front:** What is the primary benefit of using a 'multistage' classification approach for intent detection?

**Back:** It provides more granular control and accuracy for complex questions and edge cases compared to a single broad prompt.

---

## Card 51

**Front:** Term: Intent Detection

**Back:** Definition: The process of analyzing user input to determine the required response type, such as a data query, system help, or feedback.

---

## Card 52

**Front:** What information should be included in the 'information' field of the query generation prompt during a retry?

**Back:** The error message from the previous failed database execution to allow the model to correct its query.

---

## Card 53

**Front:** How does the system ensure consistency when using 'few-shot' examples in prompts?

**Back:** Examples are structured to follow the exact same JSON format and field order expected in the final output.

---

## Card 54

**Front:** What does the '...' notation in prompt examples signify?

**Back:** A placeholder indicating that the model should generate the content for that field autonomously.

---

## Card 55

**Front:** In the context of KG querying, what is a 'traversal'?

**Back:** A path through the graph following specific relationship types between nodes.

---

## Card 56

**Front:** Why might a domain expert's Energy be wasted if they are forced to write technical Cypher queries?

**Back:** Their energy should be devoted to domain mastery and solving complex problems, not learning technical query syntax.

---

## Card 57

**Front:** What role do 'ANPR' cameras play in the policing KG example?

**Back:** They act as nodes that capture vehicle data, creating a relationship between a car and a specific location/time.

---

## Card 58

**Front:** How does 'Expert Emulation' help maximize the ROI (Return on Investment) of Knowledge Graphs?

**Back:** By empowering non-technical analysts to fully utilize the KG, ensuring the structured data actually impacts decisions.

---

## Card 59

**Front:** What is the benefit of a 'narrative schema representation' for human readers?

**Back:** It provides rich context and natural language examples that make complex structures easier to understand.

---

## Card 60

**Front:** Why is the narrative schema format often less effective for LLMs than a structured format?

**Back:** It can be verbose, unsystematic, and harder to parse efficiently as the schema size grows.

---

## Card 61

**Front:** What logic governs the 'Reasoning-first' approach in JSON generation?

**Back:** The model generates tokens sequentially, so placing 'reasoning' first forces the logic to be established before the 'answer'.

---

## Card 62

**Front:** How can a system handle 'feedback and complaints' identified during intent detection?

**Back:** By routing them to a specific pipeline for logging user issues or enhancement requests rather than querying the KG.

---

## Card 63

**Front:** What is the significance of 'terminological mapping' in schema annotations?

**Back:** It ensures the LLM correctly associates natural language terms with specific database abbreviations or codes.

---

## Card 64

**Front:** In the provided architecture, what follows 'Query execution' in the pipeline?

**Back:** Visualization and Summary generation.

---

## Card 65

**Front:** Which visualization type would be triggered by the question: 'Where were the last 10 narcotics crimes committed?'

**Back:** A map.

---

## Card 66

**Front:** Why does the summarization prompt include the 'Cypher query' that was executed?

**Back:** To provide the LLM with the context of how the user's question was technically interpreted.

---

## Card 67

**Front:** What does a 'success: false' output in query generation communicate to the application layer?

**Back:** That the system was unable to generate a valid query based on the schema and the user should be informed why.

---

## Card 68

**Front:** How does the system ensure that Cypher queries do not use 'anonymous' relationships?

**Back:** By explicitly instructing the LLM to use named variables like `[rel0:TYPE]` instead of just `[:TYPE]`.

---

## Card 69

**Front:** In the policing domain, what characteristic often outweighs technical expertise for interpretation?

**Back:** Sharp intuition and deep contextual understanding of patterns.

---

## Card 70

**Front:** What is a 'boundary case' in intent detection classification?

**Back:** A question that could arguably belong to more than one category, requiring clear examples to define the rule.

---

## Card 71

**Front:** Why is the 'JSON format' strictly required for the LLM output?

**Back:** To enable the system to programmatically process the reasoning, query, and success flag for downstream steps.

---

## Card 72

**Front:** What happens if an LLM 'hallucinates' a relationship in a Cypher query?

**Back:** The query will fail to execute because the relationship type does not exist in the database schema.

---

## Card 73

**Front:** How does 'filtering' during schema extraction help prevent query errors?

**Back:** It removes technical implementation details that might confuse the model into generating invalid traversal patterns.

---

## Card 74

**Front:** What is 'expert-emulated question answering' designed to mimic?

**Back:** The systematic reasoning and data retrieval patterns used by skilled human information retrieval experts.

---

## Card 75

**Front:** In the architecture roadmap, what component receives feedback about execution errors?

**Back:** The Query generation module.

---

## Card 76

**Front:** What is the primary advantage of a 'single broad prompt' for intent detection?

**Back:** Reduced management overhead and simpler implementation.

---

## Card 77

**Front:** What determines the choice of 'presentation method' for a query response?

**Back:** The user's identified intent and the specific nature of the retrieved data (e.g., locations vs. aggregations).

---

## Card 78

**Front:** Why is a 'schema reminder' sometimes necessary in long-context prompts?

**Back:** To ensure the model adheres to the valid nodes and relationships even when the prompt contains many other instructions.

---

## Card 79

**Front:** How does the 'integration-first' approach influence the design of query responses?

**Back:** It dictates the structure of query responses to ensure they are compatible with front-end graphical interfaces.

---
