# Healthcare Quiz

## Question 1
Which of the following best describes the primary obstacle to data integration in healthcare, as discussed in the text?

- [ ] The high cost of maintaining relational and document-oriented storage technologies simultaneously.
- [ ] The use of different information syntaxes for identical dates and numerical values.
- [x] Semantic heterogeneity, where different terms identify the same concept or identical acronyms define distinct concepts.
- [ ] The inability of modern LLMs to process structured data formats like XML and JSON.

**Hint:** Focus on the difficulty of harmonizing the underlying meaning of terms like 'pulmonary embolism' versus 'physical examination'.

## Question 2
What is the primary role of a 'reference ontology' when constructing a knowledge graph from disparate sources?

- [x] To act as an intermediary vocabulary and schema that bridges heterogeneous data through mapping.
- [ ] To provide a storage technology that replaces relational and document-oriented databases.
- [ ] To automate the extraction of unstructured text from clinical journals without human supervision.
- [ ] To serve as a high-speed query engine for real-time patient monitoring.

**Hint:** Think about how a common language helps people from different regions communicate.

## Question 3
How does a Labeled Property Graph (LPG) differ from a standard Resource Description Framework (RDF) regarding relationship metadata?

- [ ] RDF allows metadata to be attached to individual edges, whereas LPG defines relationships globally.
- [x] LPG supports unique edges with specific properties, while RDF relationships are defined globally so metadata affects all instances.
- [ ] RDF uses Cypher for querying edge properties, while LPG relies exclusively on SPARQL.
- [ ] LPG requires the use of blank nodes to represent relationship metadata, whereas RDF uses direct key-value pairs.

**Hint:** Consider which model uses 'key-value pairs' directly on the connections between nodes.

## Question 4
What is the purpose of the $RDF^*$ (RDF-star) specification currently under development?

- [ ] To replace the SPARQL query language with a more efficient vector-based retrieval system.
- [x] To provide a way to add properties to edges in RDF, narrowing the functional gap between RDF and LPG.
- [ ] To enforce a strict schema that prevents the use of hierarchical relationships in ontologies.
- [ ] To convert all LPG databases into a single XML-based storage format.

**Hint:** Think about a way to combine the semantic strength of one technology with the edge-property flexibility of the other.

## Question 5
During the 'Data Understanding' phase of the clinician use case, which two types of information are extracted from the Human Phenotype Ontology (HPO)?

- [ ] Patient genomes and historical financial records for medical treatments.
- [x] A set of standardized phenotypic anomalies (ontology) and a dataset of disease-feature associations (annotations).
- [ ] A list of drug interactions and the real-time availability of hospital beds.
- [ ] Social media data about patient symptoms and hospital review scores.

**Hint:** The HPO repository provides both a 'vocabulary' and 'links' between diseases and traits.

## Question 6
In the provided Cypher script for ingesting HPO annotations, why is the 'FOREACH' clause combined with a 'CASE' statement?

- [x] To ensure that properties are only set on the relationship if the corresponding value in the TSV row is not null.
- [ ] To delete any existing relationship before creating a new one to avoid duplicates.
- [ ] To iterate through multiple phenotypic features stored in a single column of the CSV.
- [ ] To speed up the query by parallelizing the ingestion across multiple database shards.

**Hint:** Think about how to prevent errors when some rows in a table are missing information.

## Question 7
What is the primary benefit of using a 'blank node' in an RDF representation of a medical annotation?

- [ ] It assigns a permanent global URI that can be accessed by any web application.
- [x] It groups related metadata—such as source, author, and date—together without needing a unique identifier for the annotation itself.
- [ ] It automatically compresses the data to reduce the size of the triple store.
- [ ] It serves as a root node for the entire ontology hierarchy to improve traversal speed.

**Hint:** Think of it as an 'anonymous object' in a programming language.

## Question 8
Which phase of the adapted CRISP-DM model involves defining the algorithms for machine learning tasks on top of the KG?

- [x] Modeling
- [ ] Data Preparation
- [ ] Evaluation
- [ ] Business Understanding

**Hint:** This term is commonly used in data science to describe the process of creating mathematical representations of patterns.

## Question 9
What powerful capability does deductive reasoning provide to a clinician using a Knowledge Graph, according to the 'Reasoning over the KG' section?

- [ ] The ability to automatically generate new phenotypic traits using generative AI.
- [x] The ability to discover diseases implicitly linked to broad phenotypic categories through hierarchical subclass relationships.
- [ ] The ability to encrypt patient data to prevent unauthorized access by other clinicians.
- [ ] The ability to predict the financial cost of a patient's treatment based on their symptoms.

**Hint:** Think about how being a 'subclass' of something allows you to inherit the properties of the parent category.

## Question 10
Why did the authors conclude that LPG was the 'best solution' for the clinician use case involving disease-phenotype annotations?

- [ ] LPG is the only technology that can be interpreted by Large Language Models like GPT-4.
- [x] LPG's ability to represent annotation details directly as key-value pairs on relationships is highly accessible and efficient for metadata-rich associations.
- [ ] LPG provides a more rigorous semantic standard that is regulated by the World Wide Web Consortium (W3C).
- [ ] LPG automatically converts natural language symptoms into medical codes without a reference ontology.

**Hint:** Think about the convenience of putting notes directly on the line connecting two dots.
