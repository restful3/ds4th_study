# Graph Quiz

## Question 1
What is the primary function of the 'Metagraph layer' in the three-layer knowledge graph design described in the source?

- [ ] To store the final, normalized, and cleansed entities such as Person and Organization.
- [ ] To act as a repository for the raw, uncleaned text extracted from digitized page files.
- [x] To hold unmerged entity mentions and their relations while maintaining links to their source pages.
- [ ] To perform optical character recognition on scanned analog documents.

**Hint:** Consider which layer acts as an intermediate step between raw data and the final resolved network.

## Question 2
Why did off-the-shelf coreference models fail to resolve shortened names like 'U.Cal.' in the Rockefeller Archive Center documents?

- [x] The documents were written using uncommon linguistic conventions and abbreviations specific to the program officers.
- [ ] The OCR system frequently misspelled institution names, making them unrecognizable to the models.
- [ ] Coreference models are generally unable to link organizations to geographical locations.
- [ ] There was a lack of high-relational complexity in the text for the models to analyze.

**Hint:** Think about the unique writing styles of the individuals who created the original diaries.

## Question 3
In the graph-based entity resolution process, which algorithm is used to identify groups of mentions that should be resolved to the same final KG entity?

- [ ] PageRank
- [x] Weakly Connected Components (WCC)
- [ ] Betweenness Centrality
- [ ] Louvain Community Detection

**Hint:** Look for the algorithm that identifies disconnected subgraphs within a network of similarity links.

## Question 4
Which centrality measure is specifically highlighted as useful for identifying 'Bridges'—people who connect different communities of researchers?

- [ ] Eigenvector centrality
- [ ] Node degree
- [x] Betweenness centrality
- [ ] PageRank

**Hint:** Think about the metric that focuses on the shortest paths between any pair of nodes in the graph.

## Question 5
According to the source, why is 'Democratization' considered a value of Knowledge Graphs in the LLM era?

- [ ] KGs allow anyone to train their own massive Large Language Models for free.
- [x] Organizations can use an expensive LLM once to produce a KG and then use the KG for long-term analysis.
- [ ] KGs automatically translate complex data into all known languages for global access.
- [ ] KGs remove the need for expert data scientists, allowing anyone to perform advanced ML.

**Hint:** Focus on the economic and resource-related benefits mentioned in section 6.4.

## Question 6
Which challenge in the Rockefeller Archive Center project specifically refers to the inability to use reference dictionaries for disambiguation?

- [x] Historical documents
- [ ] High relational complexity
- [ ] Domain-specific named entities
- [ ] Uncommon linguistic conventions

**Hint:** Consider the status of the research disciplines discussed in documents from 1939.

## Question 7
The presence of 'Laurence Irving' in the cyclotron influence network (Figure 6.5) is cited as an example of what?

- [ ] A successful discovery of a hidden historical connection.
- [x] A failure in the Relation Extraction (RE) task by the LLM.
- [ ] A breakthrough in unsupervised entity resolution of occupations.
- [ ] A data normalization success where a misspelled name was correctly linked.

**Hint:** Recall the warning that LLMs are not 'magic' and can make specific types of mistakes.

## Question 8
What cleansing strategy was implemented to handle GPT's tendency to include academic titles as part of a person's name?

- [ ] Lowercasing all person names to ensure string similarity matches.
- [x] Using Cypher queries to strip irrelevant tokens/titles from the name property.
- [ ] Re-running the optical character recognition with a title-exclusion filter.
- [ ] Manually deleting every person node that contained a title.

**Hint:** Think about how data consistency is maintained when the LLM ignores specific instructions.

## Question 9
How does the knowledge graph schema enable 'Explainability' in applications?

- [ ] By using LLMs to write long paragraphs explaining why each node exists.
- [x] By allowing users to view the original text snippets from which entities and relations were identified.
- [ ] By limiting the number of hops allowed in a query to avoid complex logic.
- [ ] By black-boxing the extraction process so users are not confused by technical details.

**Hint:** Consider the relationship between the 'Entity' nodes in the Metagraph and the 'Page' nodes.

## Question 10
Which approach is recommended for the unsupervised resolution of 'Occupations' with varying granularity, such as 'nuclear physics' and 'heavy nitrogen'?

- [ ] Using string similarity rules based on surnames and first names.
- [ ] Executing PageRank to find the most popular research topics.
- [x] Creating embeddings and clustering them using agglomerative hierarchical clustering.
- [ ] Mapping every occupation to a single 'Research' node to maximize connectivity.

**Hint:** Look for a method that handles semantic relationships rather than just literal text matching.
