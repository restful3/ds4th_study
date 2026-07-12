# LangGraph Flashcards

## Card 1

**Front:** What library is used as the orchestration framework for the Knowledge Graph querying pipeline?

**Back:** LangGraph

---

## Card 2

**Front:** LangGraph is specifically designed for building _____, multi-actor applications powered by LLMs.

**Back:** stateful

---

## Card 3

**Front:** How do components communicate in LangGraph to maintain decoupled interactions?

**Back:** They interact with a shared global state object.

---

## Card 4

**Front:** In the architectural core of LangGraph, what do the nodes in a directed graph represent?

**Back:** Distinct agent functions

---

## Card 5

**Front:** What determines the execution flow between nodes in a LangGraph workflow?

**Back:** The edges in the graph

---

## Card 6

**Front:** The ability of LangGraph to choose the next node based on previous agent output is called _____.

**Back:** Dynamic edge resolution

---

## Card 7

**Front:** Which frontend interface is used to make the QA system user-friendly and accessible?

**Back:** Streamlit

---

## Card 8

**Front:** What is the primary responsibility of the Intent Detection agent?

**Back:** Determining the appropriate visualization format for the user's question.

---

## Card 9

**Front:** Which agent is responsible for converting technical graph schema into an LLM-friendly format?

**Back:** Schema Extraction agent

---

## Card 10

**Front:** The Text-to-Cypher agent transforms natural language into what specific query language?

**Back:** Cypher

---

## Card 11

**Front:** What is the role of the Query Execution agent in the pipeline?

**Back:** Running the generated query against the database and retrieving results.

---

## Card 12

**Front:** Which component manages prompt templates and system settings used by the agents?

**Back:** Configuration provider

---

## Card 13

**Front:** What does the Question Processing Interface transform the pipeline execution into for the frontend?

**Back:** A real-time event stream

---

## Card 14

**Front:** What shared memory space allows agents to read and write data sequentially in LangGraph?

**Back:** The state object

---

## Card 15

**Front:** Which state field captures the detected visualization format, such as a table or graph?

**Back:** output_type

---

## Card 16

**Front:** What state field is used to track errors encountered during database query execution?

**Back:** results_error

---

## Card 17

**Front:** The mechanism that allows the system to attempt query execution multiple times upon failure is the _____.

**Back:** retry mechanism

---

## Card 18

**Front:** What is the maximum number of retry attempts allowed by the post-query execution routing logic?

**Back:** Three attempts

---

## Card 19

**Front:** In the state object, the `information` and `retries` fields are used specifically for _____.

**Back:** retry logic for failed queries

---

## Card 20

**Front:** What specific visualization format causes the Query Execution agent to convert results into a pandas DataFrame?

**Back:** Tabular output

---

## Card 21

**Front:** The Intent Detection agent updates the state with the visualization type and what other metadata?

**Back:** The reasoning behind the selected visualization type.

---

## Card 22

**Front:** Which agent allows users to reference selected nodes without explicitly describing them in text?

**Back:** Text-to-Cypher agent

---

## Card 23

**Front:** What component is used to exclude technical nodes and implementation-specific properties from the conceptual schema?

**Back:** A skip list within the configuration-based transformation approach.

---

## Card 24

**Front:** The process of transforming technical schema into business-oriented terminology is called _____.

**Back:** Schema translation

---

## Card 25

**Front:** What does the Summary Generation agent use to create a comprehensive context for its output?

**Back:** Query results and schema selection

---

## Card 26

**Front:** In the LangGraph assembly, which node is set as the entry point?

**Back:** intent_detection

---

## Card 27

**Front:** What Python function type is used by the interface layer to maintain a linear flow while producing intermediate events?

**Back:** Generator function

---

## Card 28

**Front:** The event stream produced by the processing interface typically consists of a _____ of (type, payload, state).

**Back:** triplet

---

## Card 29

**Front:** Which Streamlit mechanism is used to show immediate, transient feedback as the pipeline processes a question?

**Back:** Placeholders

---

## Card 30

**Front:** What component in the Streamlit application accumulates the permanent state of a conversation?

**Back:** MessageHistory

---

## Card 31

**Front:** When does the Streamlit UI re-render using MessageHistory's final display logic?

**Back:** Upon receiving the END event from the pipeline.

---

## Card 32

**Front:** In the expert-emulating approach, what does the system mimic to interact with graph databases?

**Back:** The workflow and reasoning of human graph experts.

---

## Card 33

**Front:** What does the schema provider extract programmatically before transforming it for the LLM?

**Back:** Technical schema information

---

## Card 34

**Front:** The Intent Detection agent analyzes the user's question to determine one of which three visualization formats?

**Back:** Table, graph, or map

---

## Card 35

**Front:** Which agent resets the retries counter upon a successful schema update?

**Back:** Schema extraction agent

---

## Card 36

**Front:** The Text-to-Cypher agent adds query generation annotations from the _____ before executing the prompt.

**Back:** configuration provider

---

## Card 37

**Front:** What is the function of conditional edges in a LangGraph setup?

**Back:** Implementing routing logic based on the current state of the workflow.

---

## Card 38

**Front:** In the post-query execution logic, what event occurs if the `results_error` field is empty and the output type is 'table'?

**Back:** The workflow proceeds directly to the END state.

---

## Card 39

**Front:** What is the primary output of the Summary Generation agent stored in the state?

**Back:** The summary text, reasoning, and analysis flags.

---

## Card 40

**Front:** Which file manages prompt templates to separate template authoring from rendering logic?

**Back:** config.yaml

---

## Card 41

**Front:** What does the 'Selection' column in the Streamlit interface allow users to do?

**Back:** Select specific nodes to provide context for follow-up natural language questions.

---

## Card 42

**Front:** Why is the generator function `processQuestion` used for frontend communication?

**Back:** It allows the frontend to track pipeline progress and intermediate updates in real time.

---

## Card 43

**Front:** In the context of the state-based model, what is the 'whiteboard' metaphor used to describe?

**Back:** The shared state where each agent can read previous work and append new results.

---

## Card 44

**Front:** How does the system bridge the gap between technical schema and business concepts?

**Back:** By using a configuration-based transformation approach and skip lists.

---

## Card 45

**Front:** Which architectural component is responsible for managing the connection to the Neo4j database?

**Back:** Schema provider

---

## Card 46

**Front:** What defines the end of a successful workflow in the LangGraph implementation?

**Back:** The END state

---

## Card 47

**Front:** What state fields capture the reasoning for visualization intent?

**Back:** output_type_reason (or output_reason)

---

## Card 48

**Front:** Which agent uses the `Neo4jSchema` object to facilitate its primary task?

**Back:** Schema extraction agent

---

## Card 49

**Front:** How does the Query Execution agent handle graph or map visualizations differently than tables?

**Back:** It preserves results in their native list-of-records format.

---

## Card 50

**Front:** The routing logic that chooses between 'retry', 'summarize', or 'END' is implemented as a _____ edge.

**Back:** dynamic (or conditional)

---

## Card 51

**Front:** What is the benefit of the reactive pattern in Streamlit integration?

**Back:** It manages both temporary progress updates and the permanent conversation history.

---

## Card 52

**Front:** Which component forwards questions to the LangGraph workflow and yields state updates?

**Back:** Question processing interface

---

## Card 53

**Front:** What concept refers to organizing mental models of KGs at different levels of detail for experts?

**Back:** Multilayer schema management

---

## Card 54

**Front:** In Figure 15.3, what do the dashed arrows represent?

**Back:** Conditional paths based on query execution outcomes.

---

## Card 55

**Front:** The `AgentState` definition in Listing 15.11 represents the _____ of the entire pipeline.

**Back:** global state schema

---

## Card 56

**Front:** What does the term 'context-aware query generation' imply in this system?

**Back:** Combining schema knowledge, user selections, and conversational history for better queries.

---

## Card 57

**Front:** What is the main advantage of the 'expert-emulating' approach over simple RAG?

**Back:** It handles complex reasoning and multi-step investigation processes more effectively.

---

## Card 58

**Front:** Which agent populates the `query_reasoning` field in the state?

**Back:** Text-to-Cypher agent

---

## Card 59

**Front:** What is the purpose of 'operational notes' in the pipeline configuration?

**Back:** To guide the LLM's query generation using domain knowledge and best practices.

---

## Card 60

**Front:** In the `processQuestion` generator, what does the 'result' event type signify?

**Back:** An intermediate output such as reasoning, error messages, or a summary.

---

## Card 61

**Front:** Which field in the state object determines if the pipeline should route to the 'generate_summary' node?

**Back:** output_type (when it is 'graph' or 'map')

---

## Card 62

**Front:** What library's `StateGraph` class is used to assemble the pipeline structure?

**Back:** LangGraph

---

## Card 63

**Front:** What is a 'thread_id' used for in the pipeline configuration?

**Back:** Tracking individual conversation sessions for state persistence.

---

## Card 64

**Front:** The Intent Detection agent is described as the _____ point of the pipeline.

**Back:** entry

---

## Card 65

**Front:** What format is the LLM response expected in for the Intent Detection agent to map state fields?

**Back:** JSON format

---

## Card 66

**Front:** How does the Schema Provider ensure technical accuracy while providing a business view?

**Back:** By balancing programmatic schema extraction with configuration-based transformations.

---

## Card 67

**Front:** What is 'dynamic edge resolution' used for after query execution?

**Back:** To decide whether to retry a failed query or move to the summary phase.

---

## Card 68

**Front:** Which Streamlit column displays the investigative progress and answers?

**Back:** The Chat column

---

## Card 69

**Front:** In the context of LangGraph, what is an 'agent function'?

**Back:** A function representing a node in the graph that reads from and updates the global state.

---

## Card 70

**Front:** What is the primary role of the `checkpointer` in LangGraph's `compile` method?

**Back:** Managing memory and state persistence across steps.

---

## Card 71

**Front:** Which state field stores the original user request throughout the pipeline?

**Back:** question

---

## Card 72

**Front:** What does 'expert emulation' mean in the context of Knowledge Graph querying?

**Back:** Mimicking how human experts understand schema and construct reasoned queries.

---

## Card 73

**Front:** In Listing 15.15, what is used to extract selected nodes from the canvas?

**Back:** Node IDs

---

## Card 74

**Front:** What is the significance of the `output_analysis` flag in the summary generation?

**Back:** It indicates whether additional analytical steps were performed during summarization.

---

## Card 75

**Front:** The 'Schema' field in the state carries the graph schema in what specific format?

**Back:** LLM-friendly string format

---

## Card 76

**Front:** Which component acts as a bridge between the core backend pipeline and frontend applications?

**Back:** Question processing interface

---

## Card 77

**Front:** What is the final event yielded by the `processQuestion` generator function?

**Back:** The 'END' event

---

## Card 78

**Front:** Why is transparency in the system pipeline important for users according to Chapter 15?

**Back:** It allows users to validate the reasoning process and provides debugging insight.

---

## Card 79

**Front:** The approach of creating separate agents for intent, schema, and query generation is a _____ architecture.

**Back:** modular (or multi-agent)

---

## Card 80

**Front:** What future enhancement involves agents that analyze data patterns to enrich base schema?

**Back:** Schema enrichment agents

---
