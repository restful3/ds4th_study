# Graph Quiz

## Question 1
According to the source material, why can the retrieval-augmented generation (RAG) approach fail in complex law enforcement scenarios?

- [x] Retrieval may be incomplete, leading the LLM to reach conclusions that are the opposite of reality.
- [ ] LLMs are inherently unable to summarize witness statements or legal definitions.
- [ ] RAG systems require too much computational power to navigate large knowledge graphs.
- [ ] The policing domain uses data that is too structured for standard RAG pipelines to process.

**Hint:** Focus on what happens when the retriever fails to identify critical pieces of context.

## Question 2
What is the primary function of the 'Intent Detection' module in the proposed system architecture?

- [x] To classify the user's question and route it to the appropriate visualization pipeline.
- [ ] To extract all possible nodes and relationships from the database before the user finishes typing.
- [ ] To determine the technical credentials of the user to limit their database access.
- [ ] To automatically correct spelling errors in the user's Cypher query.

**Hint:** Consider why the system needs to know if a user wants a map versus a table at the beginning.

## Question 3
Why is a 'conceptual schema' preferred over a 'technical schema' when providing context to an LLM?

- [x] It reduces cognitive load by filtering out administrative nodes and technical metadata.
- [ ] It allows the LLM to write the query in SQL instead of Cypher.
- [ ] Technical schemas are typically too small for an LLM to find any patterns.
- [ ] Conceptual schemas include the actual data records rather than just the structure.

**Hint:** Think about the 'noise' that might distract an AI from the core domain logic.

## Question 4
In the context of schema representation, how does an 'annotated schema' help the LLM generate more accurate queries?

- [x] It provides a 'cheat sheet' explaining abbreviations and relationship meanings.
- [ ] It encrypts the database structure to ensure that the LLM cannot see sensitive properties.
- [ ] It converts the graph into a flat vector space for semantic search.
- [ ] It automatically generates a summary of the data before the query is even run.

**Hint:** Consider how an expert might explain specific data codes to a colleague.

## Question 5
The text discusses 'Chain-of-thought' and 'Scratchpad' techniques. What is the intended effect of these methods on an LLM?

- [x] They encourage the model to allocate more computational resources to step-by-step reasoning.
- [ ] They reduce the number of tokens generated to make the system faster.
- [ ] They allow the model to ignore the schema and rely purely on its training data patterns.
- [ ] They ensure the LLM never makes a syntax error in the Cypher code.

**Hint:** Think about the phrase 'giving the model time to think' mentioned in the chapter.

## Question 6
What is a potential disadvantage of an 'answer-first' prompt structure in classification tasks?

- [x] The model may commit to an incorrect answer early and use reasoning only to justify that bias.
- [ ] The model will refuse to provide any reasoning if the answer is given first.
- [ ] It significantly increases the financial cost of each API call.
- [ ] The model becomes unable to format the output as a valid JSON object.

**Hint:** Consider the concept of 'cumulative context' and how it acts as a guardrail for the LLM.

## Question 7
When generating a Cypher query, why does the system ask the LLM to list intended relationships 'out loud' before writing the code?

- [x] To reduce the chance of the model hallucinating relationships not found in the schema.
- [ ] To ensure that the final result is always displayed as a geographic map.
- [ ] To allow the database to pre-index those specific relationships for faster execution.
- [ ] To satisfy the requirement that all Cypher queries must be written in uppercase.

**Hint:** Think about the specific error where an LLM invents a relationship type that doesn't exist.

## Question 8
According to the chapter, what should a well-crafted response summary focus on?

- [x] Complementing the visual representation by highlighting insights that may not be immediately apparent.
- [ ] Repeating every node and property name visible on the graph canvas.
- [ ] Explaining the technical reasons why the Cypher query failed to run.
- [ ] Providing the legal definition of the terms used in the user's question.

**Hint:** Think about the 'dual nature' of the presentation: graph plus text.

## Question 9
How does the system handle a user question like 'The system is too slow and keeps freezing'?

- [x] It classifies it as 'Feedback and Complaints' through the Intent Detection module.
- [ ] It generates a Cypher query to find all 'Slow' status nodes in the graph.
- [ ] It automatically restarts the graph database server.
- [ ] It ignores the input as it does not contain a specific entity like a 'Person' or 'Car'.

**Hint:** Look at the categories in the expanded intent detection architecture.

## Question 10
What role does the YAML configuration file play in the schema extraction process?

- [x] It manages skip lists to filter out unnecessary classes and properties from the raw technical schema.
- [ ] It stores the actual natural language questions asked by users for future training.
- [ ] It converts Cypher queries into SQL for cross-database compatibility.
- [ ] It acts as the primary vector database for RAG retrieval.

**Hint:** Consider how the system transitions from a 'Technical Schema' to a 'Conceptual Schema'.
