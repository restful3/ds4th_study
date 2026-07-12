# Disambiguation Quiz

## Question 1
According to the source material, why might traditional Named Entity Disambiguation (NED) tools like scispaCy fail to disambiguate 'Zika' in a sentence without specific contextual words like 'congenital'?

- [x] They do not use existing relationships and paths between entities for the disambiguation task.
- [ ] They lack pretrained models for the biomedical domain, making them unable to recognize virus names.
- [ ] They are designed to be general-purpose and lack specialized vocabularies like the Unified Medical Language System (UMLS).
- [ ] They require the full ontology to be loaded into local memory, which exceeds standard hardware limits.

**Hint:** Consider the structural information available in an ontology that traditional tools might overlook.

## Question 2
What is the primary reason the proposed system uses Neo4j's full-text search instead of an LLM for the Candidate Selection (CS) phase?

- [x] The size of the domain ontology is too large to fit entirely within an LLM's prompt window.
- [ ] LLMs are incapable of performing string-based matching between text mentions and entity names.
- [ ] Neo4j's Graph Data Science library is required to translate JSON outputs into natural language.
- [ ] Full-text search is the only way to incorporate hierarchical information into the candidate pool.

**Hint:** Think about the technical constraints of context length when dealing with large external knowledge bases.

## Question 3
In the context of local deployment with Ollama, what specific characteristic makes Llama 3.1 8B suitable for the described NED system?

- [x] It is optimized for multilingual information processing and efficient deployment on consumer-grade hardware.
- [ ] It has a context length limited to 8,000 tokens, which prevents it from being overwhelmed by long documents.
- [ ] It is the only open-source model that supports the OpenAI Chat Completions API protocol.
- [ ] It eliminates the need for an external database by embedding the SNOMED ontology into its parameters.

**Hint:** Consider the hardware requirements and language capabilities mentioned in the deployment section.

## Question 4
During the Candidate Disambiguation (CD) process, what is the role of 'Path-to-text translation'?

- [x] It converts graph-based relational links into natural language sentences that the LLM can more easily interpret.
- [ ] It automatically assigns a SNOMED ID to a text mention based on its alphabetical position.
- [ ] It summarizes all possible candidates into a single JSON object to reduce token usage.
- [ ] It filters out 'hub nodes' from the graph to ensure only the most generic connections are kept.

**Hint:** Focus on the format change that allows a language model to 'understand' graph connections.

## Question 5
What is the purpose of the 'Summarizing textual paths' step in the NED workflow?

- [x] To reduce the cognitive load on the LLM by condensing multiple relationship sentences into a synthetic explanation.
- [ ] To identify the character indices of every named entity within the input document.
- [ ] To calculate the degree of each node in the SNOMED ontology using Neo4j GDS.
- [ ] To translate medical terms from Latin into English for easier processing by Llama 3.1.

**Hint:** Think about why a developer would want to provide a 'distilled' version of context to an LLM.

## Question 6
Which of the following describes the output of the Named Entity Recognition (NER) stage in the multi-step NED process?

- [x] A structured JSON object containing sentences and an array of identified entities with labels like 'Disease' or 'Organism'.
- [ ] A list of Neo4j Cypher queries designed to retrieve the shortest path between candidates.
- [ ] A prioritized list of SNOMED IDs that have already been disambiguated using contextual context.
- [ ] A set of path summaries that describe the relationships between archetypal entities.

**Hint:** Consider the fields shown in the JSON result following Listing 8.10 in the text.

## Question 7
How does the system handle 'hub nodes' when searching for shortest paths in the SNOMED graph?

- [x] It excludes highly connected nodes to focus on more meaningful and specific relationships.
- [ ] It prioritizes them as 'archetypal entities' because they contain the most propagated information.
- [ ] It uses them as the only permitted bridges between clinical findings and organism labels.
- [ ] It requires the LLM to identify which nodes are hubs before the Neo4j query can be executed.

**Hint:** Reflect on why a very common, generic word might be unhelpful when trying to find a specific connection.

## Question 8
In the final disambiguation result, the LLM maps 'Zika' to a specific SNOMED concept. Which concept was selected in the example involving microcephaly?

- [x] Congenital Zika virus infection
- [ ] Zika virus (organism)
- [ ] Zika virus disease (general)
- [ ] Chikungunya fever

**Hint:** The answer relates to the specific medical condition caused during pregnancy mentioned in the text.

## Question 9
True or False: The SNOMED ontology propagation process helps categorize deep nodes based on information from first-level archetypal entities.

- [x] True
- [ ] False

**Hint:** Look at the description of Figure 8.1 in the source.

## Question 10
Which Neo4j tool is specifically mentioned for identifying minimal-length connections between candidates?

- [x] Graph Data Science (GDS) library
- [ ] Ollama chat interface
- [ ] UMLS Lexical Tool
- [ ] scispaCy pipeline

**Hint:** This library is often used for running mathematical operations and algorithms on data in Neo4j.
