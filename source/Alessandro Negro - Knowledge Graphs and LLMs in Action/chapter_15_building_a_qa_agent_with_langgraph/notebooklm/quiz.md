# LangGraph Quiz

## Question 1
What is the primary role of LangGraph within the knowledge graph (KG) querying pipeline described in Chapter 15?

- [ ] It serves as the primary graph database for storing criminal record data.
- [x] It acts as the orchestration framework that manages stateful, multi-actor workflows.
- [ ] It provides the frontend interface components for user interaction and visualization.
- [ ] It functions as the Large Language Model that generates the final natural language summary.

**Hint:** Consider the framework used to coordinate the discrete stages of the pipeline.

## Question 2
In LangGraph's architecture, how do individual agent functions communicate with one another?

- [ ] By passing data packets directly from one node's output to the next node's input.
- [ ] Through a centralized API gateway that translates protocols between functions.
- [x] By interacting with a shared, global state object that acts as a 'whiteboard'.
- [ ] By writing temporary JSON files to a local directory for other agents to read.

**Hint:** Think about the shared memory concept mentioned in the core principles of the library.

## Question 3
Why does the system include a Schema Translation service instead of simply using the technical schema from the database?

- [ ] Technical schemas are often too large to fit within the context window of modern LLMs.
- [ ] LLMs require the data to be converted into SQL-compatible formats to generate Cypher queries.
- [x] To bridge the gap between technical database elements and the conceptual/business view needed by the LLM.
- [ ] The technical schema contains encrypted properties that the LLM cannot read without translation.

**Hint:** Reflect on the difference between how a database is structured technically and how a human expert understands the domain.

## Question 4
During the 'Intent Detection' stage, which two key fields are typically updated in the state object?

- [ ] query_string and cypher_statement
- [x] output_type and output_reason
- [ ] user_id and session_token
- [ ] schema_nodes and relationship_filters

**Hint:** Look for the variables that determine how the final results will be visualized and why.

## Question 5
What is the benefit of using a generator function in the question-processing interface layer?

- [ ] It automatically translates Cypher queries into natural language.
- [x] It allows the application to stream intermediate events and provide real-time feedback to the user.
- [ ] It increases the maximum number of nodes the graph database can process per second.
- [ ] It encrypts the communication between the Streamlit frontend and the LangGraph backend.

**Hint:** Consider how the user experience is improved when a process takes a long time to complete.

## Question 6
In the investigative use case, why was the vehicle with plate EB16946 flagged as a 'point of interest'?

- [ ] It was the only vehicle detected by the ANPR camera on the day of the incident.
- [x] It appeared twice in the results, suggesting a potential circuit of the area during the incident.
- [ ] The license plate exactly matched the victim's partial description of 'EB'.
- [ ] The vehicle was registered to a known associate of the victim.

**Hint:** Look for patterns in the detection events rather than just the vehicle's physical properties.

## Question 7
The 'expert-emulating' approach suggests that system improvements should be grounded in what principle?

- [ ] Reducing the total number of agents to minimize computational cost.
- [x] Modeling the system's behavior on how a human expert would reason through the problem.
- [ ] Automating all schema changes without any human oversight or configuration.
- [ ] Using the largest possible LLM regardless of the specific task requirements.

**Hint:** The name of the approach itself provides a clue to its foundational philosophy.

## Question 8
What happens in the 'Query Execution' agent if an error occurs during the first attempt at running a Cypher query?

- [ ] The system immediately terminates the entire pipeline and returns the error to the user.
- [x] The agent increments a retry counter and enriches the state with error information for a potential retry.
- [ ] The system switches to a backup SQL database to see if the data exists there instead.
- [ ] The agent automatically deletes the problematic nodes from the graph to prevent future errors.

**Hint:** Check the logic related to the 'retries' field in the state object.

## Question 9
Which component is responsible for filtering out technical elements (like 'skip lists') and enriching elements with business descriptions?

- [ ] Intent Detection Agent
- [x] Neo4jSchema class in the Schema Provider
- [ ] Streamlit MessageHistory object
- [ ] Text-to-Cypher Agent

**Hint:** This component bridges the gap between the technical database view and the LLM's conceptual view.

## Question 10
How does the Streamlit application handle the visualization of different data types (e.g., tables vs. maps)?

- [ ] It uses a single generic text window that describes the data in bullet points.
- [x] It uses a match statement to handle different response types and render appropriate components like canvas or tables.
- [ ] The user must manually select the visualization type before the query is even processed.
- [ ] It converts all outputs into static images to reduce the browser's memory usage.

**Hint:** Think about how the code routes different 'output_type' values to specific UI elements.
