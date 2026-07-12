# Disambiguation Quiz

## Question 1
What is the primary objective of Named Entity Disambiguation (NED) that distinguishes it from Named Entity Recognition (NER)?

- [x] To connect identified mentions in text to specific entities in a reference knowledge base.
- [ ] To identify relevant named entities and assign them to broad categories like people or locations.
- [ ] To extract the raw text from unstructured documents like PDFs for further processing.
- [ ] To automatically translate medical terminology into different languages for global interoperability.

**Hint:** Consider what happens after an entity is 'recognized' but before its specific, unique identity in a database is confirmed.

## Question 2
During the Named Entity Disambiguation process, what occurs during the 'Candidate Selection' phase?

- [x] Identifying a set of plausible entities from a knowledge base that could match a textual mention.
- [ ] Assigning a numerical score to each potential match based on the context of the sentence.
- [ ] Aggregating information from multiple ontologies to enrich the representation of a target entity.
- [ ] Defining the schema for the knowledge graph, including node labels and relationship types.

**Hint:** Think about the first step in narrowing down possibilities from a large reference database.

## Question 3
Why is context-aware candidate ranking crucial for disambiguating terms like 'Zika' in medical texts?

- [x] Because the same term can refer to distinct entities, such as a virus, a disease, or a specific congenital syndrome.
- [ ] Because NER models are unable to distinguish between capitalized and lowercase versions of medical terms.
- [ ] Because medical documents always use the same term to refer to the exact same ontological concept.
- [ ] Because it allows the system to ignore entities that do not have a corresponding ID in the UMLS knowledge base.

**Hint:** Think about how a human expert distinguishes between two different medical conditions that share the same name.

## Question 4
Which phase of the NED system architecture enriches the target entity by aggregating information from multiple sources like SNOMED and UMLS?

- [x] Ontology integration
- [ ] Candidate ranking
- [ ] Document ingestion
- [ ] Candidate selection

**Hint:** This step involves connecting the results to external knowledge structures to add contextual value.

## Question 5
What is a significant limitation of using generic LLMs like ChatGPT for medical entity disambiguation compared to specialized tools like scispaCy?

- [x] LLMs may not have specific domain knowledge bases like UMLS directly incorporated into their internal parameters.
- [ ] LLMs are incapable of identifying complex relationships between words in a natural language sentence.
- [ ] LLMs cannot process unstructured text into any form of structured knowledge representations.
- [ ] LLMs are limited to identifying only broad categories like 'organizations' or 'locations'.

**Hint:** Consider the difference between 'understanding the meaning of a word' and 'knowing its specific unique ID in a formal registry'.

## Question 6
In the construction of a Knowledge Graph, what is the purpose of propagating information from 'first-level nodes' to 'deep nodes' in an ontology?

- [x] To ensure that specific entities inherit broader classifications or archetypes from their parent nodes.
- [ ] To reduce the total number of relationships in the graph to save storage space.
- [ ] To ensure that every entity in the graph is connected to a single disease node like 'Zika'.
- [ ] To identify which documents mention specific medical entities for the first time in a corpus.

**Hint:** Think about how inheritance works in a taxonomy where a specific sub-category belongs to a more general main category.

## Question 7
How does 'Conceptual Search' differ from a standard full-text search in a Knowledge Graph system?

- [x] It uses ontology information, including names and aliases, to expand the search space beyond the exact string provided.
- [ ] It relies solely on the engine score of indexed documents to rank results by keyword density.
- [ ] It restricts the search to find exact character-by-character string matches within the document text.
- [ ] It is performed without the need for an underlying knowledge graph or formal domain ontology.

**Hint:** Consider what happens if a user searches for 'Headache' but the document only contains the medical term 'Cephalalgia'.

## Question 8
In the Knowledge Graph schema described in Chapter 7, what does the relationship 'DISAMBIGUATED_TO' signify?

- [x] The link between a specific mention in the text and its corresponding unique medical entity node.
- [ ] The hierarchical connection between a parent category and a child concept in an ontology.
- [ ] The relationship between two distinct entities that appear within the same sentence.
- [ ] The structural connection between a file node and the individual page nodes it contains.

**Hint:** This relationship represents the final output of the candidate selection and ranking phases.

## Question 9
What is a primary benefit of incorporating 'Co-occurrence' relationships into a medical Knowledge Graph?

- [x] It helps uncover and validate relationships between entities that may not be formally encoded in existing ontologies.
- [ ] It identifies the unique UMLS identifier for every entity mentioned in a given document.
- [ ] It reduces graph complexity by merging similar entities from different ontologies into a single node.
- [ ] It allows the system to recognize named entities like organizations and geographic locations.

**Hint:** Think about how seeing two words in the same sentence suggests a relationship, even if a formal dictionary doesn't list it yet.

## Question 10
According to the CRISP-DM model adaptation for Knowledge Graphs, why is the 'Data Preparation' phase fundamental before ingestion?

- [x] To process different document formats and ontologies into a state where they can be properly ingested and mapped.
- [ ] To define the high-level business goals and use cases for the SoHO policy officer.
- [ ] To create the final interactive visualizations for stakeholders to perform advanced pattern detection.
- [ ] To eliminate the need for domain experts to review medical documents in the healthcare field.

**Hint:** Consider the necessary steps to convert a PDF or raw medical file into a format that a graph database like Neo4j can interpret.
