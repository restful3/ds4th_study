# Graph Flashcards

## Card 1

**Front:** What is the primary shift in Knowledge Graph (KG) construction discussed in Chapter 4 compared to earlier chapters?

**Back:** The shift from using a single ontology to integrating multiple structured data sources in graph-friendly formats.

---

## Card 2

**Front:** In the context of KG construction from multiple sources, what are the three primary areas of focus for modeling?

**Back:** Graph modeling decisions, integration strategies, and analysis methods.

---

## Card 3

**Front:** Which specific biomedical data sources are used to demonstrate the KG techniques in this chapter?

**Back:** Relational databases, CSV files, and APIs.

---

## Card 4

**Front:** What role do Large Language Models (LLMs) play in the construction pipeline of structured data KGs?

**Back:** They serve as auxiliary tools rather than core components of the construction pipeline.

---

## Card 5

**Front:** According to the text, what is the 'primary approach' for integrating structured data sources like relational databases?

**Back:** Traditional data integration techniques.

---

## Card 6

**Front:** What is the specific goal of 'Precision Medicine' mentioned in the text?

**Back:** Using patient-specific information to support medical care.

---

## Card 7

**Front:** Define the term 'Multi-omic' in biological analysis.

**Back:** An approach that uses multiple 'omics' datasets, such as genomes, proteomes, and transcriptomes.

---

## Card 8

**Front:** In molecular biology, what does the suffix '-ome' indicate?

**Back:** The totality of a specific biological category (e.g., genome refers to all genetic information).

---

## Card 9

**Front:** Term: Genome

**Back:** Definition: The entire genetic complement of a living organism, typically made of DNA.

---

## Card 10

**Front:** Term: Transcriptome

**Back:** Definition: A collection of RNA molecules that direct the synthesis of the proteome.

---

## Card 11

**Front:** Term: Proteome

**Back:** Definition: The totality of proteins expressed by a cell, tissue, or organism.

---

## Card 12

**Front:** What standard coding system did Yang et al. use to unify disease identifiers from different databases?

**Back:** Unified Medical Language System (UMLS) codes.

---

## Card 13

**Front:** Which graph algorithm is preferred for finding disconnected subgraphs within a PPI network?

**Back:** Weakly Connected Component (WCC).

---

## Card 14

**Front:** How does the Louvain modularity algorithm differ from WCC in identifying groups?

**Back:** It finds 'communities' where nodes are more densely connected to each other than to nodes outside the group.

---

## Card 15

**Front:** In the context of the Louvain algorithm, what is a 'modularity score'?

**Back:** An evaluation of how well groups have been partitioned into communities compared to a random network.

---

## Card 16

**Front:** What is an 'in-memory representation' of a graph in Neo4j GDS library called?

**Back:** A projection.

---

## Card 17

**Front:** A _____ is a subgraph of the protein–protein interaction (PPI) network defined by a set of proteins associated with a specific disease.

**Back:** disease pathway

---

## Card 18

**Front:** What does the graph metric 'Density' measure in a disease pathway?

**Back:** The fraction of all possible edges that actually appear between nodes in the pathway.

---

## Card 19

**Front:** Formula: Graph Density for a subgraph $H_d$ with nodes $V_d$ and edges $E_d$

**Back:** $Density = \frac{|E_d|}{|V_d|(|V_d|-1)/2}$

---

## Card 20

**Front:** How is 'Conductance' defined in the context of disease pathway analysis?

**Back:** The independence of a subgraph from the rest of the graph, based on edges connecting internal nodes to external nodes.

---

## Card 21

**Front:** In conductance measurement, what does a 'lower value' indicate about the pathway?

**Back:** The pathway is a tighter-knit community that is well-separated from the rest of the network.

---

## Card 22

**Front:** What is the significance of the 'Relative size of the largest component' metric?

**Back:** It measures the fraction of proteins in the largest connected component relative to the total number of nodes.

---

## Card 23

**Front:** Why are disease pathways described as 'fragmented' in the PPI network?

**Back:** Because they often have a low percentage of proteins in the largest pathway component (median of 21%).

---

## Card 24

**Front:** Concept: Hetionet

**Back:** Definition: A heterogeneous network (hetnet) that integrates knowledge from 29 public resources to connect compounds, diseases, and genes.

---

## Card 25

**Front:** What is a 'metagraph' in the context of Knowledge Graph schemas?

**Back:** A description of the structure of the database indicating the types of nodes and the types of relationships.

---

## Card 26

**Front:** What is a 'metapath'?

**Back:** A sequence of node and relationship classes that describe potential real paths between two node types.

---

## Card 27

**Front:** In Hetionet analysis, what does 'PC' stand for, and what does it measure?

**Back:** Path Count; the number of paths of a specified metapath between defined source and target nodes.

---

## Card 28

**Front:** What limitation of Path Count (PC) does Degree-Weighted Path Count (DWPC) address?

**Back:** PC does not adjust for the extent of graph connectivity along the path (it treats all paths as equal).

---

## Card 29

**Front:** What is the 'Path-Degree Product' (PDP) in the calculation of DWPC?

**Back:** An individual value associated with each path, calculated by multiplying exponentiated metaedge-specific degrees.

---

## Card 30

**Front:** Formula: Path-Degree Product ($PDP$)

**Back:** $PDP(path) = \prod_{d \in D_{path}} d^{-w}$

---

## Card 31

**Front:** In the DWPC formula, what does the variable '$w$' represent?

**Back:** The damping exponent, where $w \ge 0$.

---

## Card 32

**Front:** How is the final DWPC for a specific metapath calculated from individual paths?

**Back:** By summing the Path-Degree Products ($PDP$) of all paths belonging to that metapath.

---

## Card 33

**Front:** What is the primary benefit of using DWPC over simple path counting?

**Back:** It down-weights paths that go through 'well-known' (high-degree) nodes.

---

## Card 34

**Front:** Concept: Patient Journey Mapping

**Back:** Definition: An approach to understanding how people enter, experience, and exit health services, often used in clinical oncology.

---

## Card 35

**Front:** How do clinical KGs address privacy concerns while using Electronic Health Records (EHRs)?

**Back:** By storing statistical relationships and weights extracted from patient data rather than the actual raw patient data.

---

## Card 36

**Front:** What is the 'Clinical Knowledge Graph' (CKG)?

**Back:** A platform that harmonizes and integrates clinical data by connecting 33 node labels with 51 relationship types.

---

## Card 37

**Front:** In the CKG analysis example, what protein was found to be strongly associated with intrinsic cardiomyopathies?

**Back:** ACTC1 (P68032).

---

## Card 38

**Front:** How can LLMs support 'Clinical Decision Support' in the context of KG analysis?

**Back:** By synthesizing multidimensional clinical data into coherent treatment recommendations and research narratives.

---

## Card 39

**Front:** What does the 'DOID' acronym refer to in biomedical ontologies?

**Back:** Disease Ontology Identifier.

---

## Card 40

**Front:** Which Neo4j tool is used to execute graph algorithms like WCC and Louvain?

**Back:** Graph Data Science (GDS) library.

---

## Card 41

**Front:** In a Neo4j `LOAD CSV` command, what keyword is used to process data in batches to manage memory?

**Back:** IN TRANSACTIONS OF (e.g., 100 ROWS).

---

## Card 42

**Front:** What is the purpose of the `MERGE` clause in Cypher during KG ingestion?

**Back:** To ensure that a node or relationship exists, creating it only if it is not already present.

---

## Card 43

**Front:** In Python-based KG analysis, which library is typically used to handle graph objects and compute network metrics like connected components?

**Back:** networkx

---

## Card 44

**Front:** Why is 'homogenization' a necessary step when building a KG from multiple sources?

**Back:** To transform structured and semistructured schemas into a consistent, unified graph format.

---

## Card 45

**Front:** What is the 'monopartite' part of the disease discovery KG mentioned in the text?

**Back:** The protein–protein interaction (PPI) network.

---

## Card 46

**Front:** What is the 'bipartite' part of the disease discovery KG mentioned in the text?

**Back:** The disease–protein association network.

---

## Card 47

**Front:** The process of identifying and linking nodes from multiple sources that represent the same real-world concept is called _____.

**Back:** entity resolution

---

## Card 48

**Front:** How many proteins are contained in the largest connected component of the human PPI network described in the text?

**Back:** 21,521 proteins.

---

## Card 49

**Front:** What does a modularity score of $0.40$ (40%) suggest about the PPI network communities found by Louvain?

**Back:** It suggests a moderate level of partitioning into communities that are more densely connected than expected by chance.

---

## Card 50

**Front:** In the DWPC example, if an edge exists between IRF1 and CXCR4, where IRF1 has 4 interactions and CXCR4 has 2, what are the degree values used for that edge?

**Back:** 4 (outgoing) and 2 (incoming).

---

## Card 51

**Front:** When using GDS, what must be created before running an algorithm like Louvain on a specific set of nodes and relationships?

**Back:** A graph projection in memory.

---

## Card 52

**Front:** According to the text, what is the role of an Intelligent Advisor System (IAS) in disease protein discovery?

**Back:** To predict and report potential proteins and related pathways associated with a target disease.

---

## Card 53

**Front:** What is the primary objective of using 'Community Detection' in KG analysis?

**Back:** To find groups of nodes that are connected more densely to each other than to the rest of the network.

---

## Card 54

**Front:** In the analysis of Celiac Disease, what role does the LLM play regarding the T cell costimulation results?

**Back:** It explains how the quantitative DWPC score relates to the autoimmune nature and chronic inflammatory response of the disease.

---

## Card 55

**Front:** Which organization provides the `gene_info` file used to map protein codes to human-readable gene symbols?

**Back:** The National Institutes of Health (NIH).

---

## Card 56

**Front:** What is the relationship between 'clinical pathways' and 'patient journey maps'?

**Back:** Clinical pathways establish the standard of care, while patient journey maps provide the companion perspective of the patient's experience.

---

## Card 57

**Front:** In the formula for Density, what does the denominator $V(V-1)/2$ represent?

**Back:** The maximum number of possible edges in an undirected graph with $V$ nodes.

---

## Card 58

**Front:** What does the text mean by 'post-processing techniques' in the context of KG integration?

**Back:** Methods used to merge entities and relationships after the initial data transformation.

---

## Card 59

**Front:** Why is the SNAP dataset's human interactome considered 'simpler' than original multi-omic sources?

**Back:** Because it has already been compiled and filtered into a network format from more complex raw data.

---

## Card 60

**Front:** What is 'schema alignment' in the context of multisource KG integration?

**Back:** The process of mapping different source schemas to a single, homogeneous graph schema.

---

## Card 61

**Front:** What is a 'node key constraint' in Neo4j?

**Back:** A rule ensuring that all nodes with a specific label have a unique identifier and that the identifier is not null.

---

## Card 62

**Front:** In the Louvain algorithm, what does 'maximizing modularity' achieve?

**Back:** It finds an optimal partition of the network where internal community edges are maximized relative to external edges.

---

## Card 63

**Front:** If a disease pathway has a conductance of $0.98$, what does this imply about its connectivity?

**Back:** The pathway is highly integrated with the rest of the network and is not a well-isolated community.

---

## Card 64

**Front:** Which library allows for the conversion of Neo4j query results into a pandas DataFrame for analysis?

**Back:** The Neo4j Python driver (often integrated with pandas).

---

## Card 65

**Front:** True or False: LLMs can replace domain experts in interpreting clinical KG results.

**Back:** False (they are intended to empower and assist humans, not replace them).

---

## Card 66

**Front:** What is the specific value range of the DWPC metric?

**Back:** The text does not specify a fixed range, but values are relative to the path-degree products and sum of paths.

---

## Card 67

**Front:** What does 'undirected subgraph' mean in the context of protein–protein interactions?

**Back:** A network where interactions do not have a specific direction (A interacts with B implies B interacts with A).

---

## Card 68

**Front:** In the SNAP PPI network, how many documented interactions are recorded among human proteins?

**Back:** 342,354 documented interactions.

---

## Card 69

**Front:** What is the 'Human Interactome'?

**Back:** The complete set of molecular interactions in a human cell, particularly protein–protein interactions.

---

## Card 70

**Front:** What is the purpose of 'damping' ($w$) in the DWPC metric?

**Back:** To control the penalty applied to high-degree nodes; a higher $w$ increases the penalty for 'hub' nodes.

---

## Card 71

**Front:** Metapaths are used to identify 'prominent' processes in which specific ontology mentioned in the pharmaceutical section?

**Back:** Gene Ontology (GO).

---

## Card 72

**Front:** Which specific GO process was identified as having a 'strong dominance' in Celiac Disease interactomics?

**Back:** Processes related to glycoproteins.

---

## Card 73

**Front:** In Figure 4.10, if $w = 0.5$ and degree values are 4 and 1, what is the exponentiated degree calculation for that edge?

**Back:** $4^{-0.5} = 0.5$ and $1^{-0.5} = 1$.

---

## Card 74

**Front:** What does 'deidentified generic clinical KG' refer to?

**Back:** A knowledge graph built using anonymized or non-sensitive data to maintain patient privacy.

---

## Card 75

**Front:** What is the median number of connected components per disease pathway in the PPI network study?

**Back:** 16 connected components.

---

## Card 76

**Front:** The Louvain algorithm is described as 'modularity-based'. What does this mean?

**Back:** It works by iteratively grouping nodes to maximize a modularity score that measures the density of links inside communities.

---

## Card 77

**Front:** In Neo4j GDS, what is the command `gds.wcc.write` used for?

**Back:** Running the Weakly Connected Components algorithm and writing the results back to the database as a node property.

---

## Card 78

**Front:** In clinical applications, what do 'edges' typically represent between patient and drug nodes?

**Back:** Relationships such as a patient being treated with a specific drug.

---

## Card 79

**Front:** According to the summary, what are the three essential validation steps for building KGs from structured data?

**Back:** Entity resolution, schema alignment, and data quality validation.

---

## Card 80

**Front:** What is the main takeaway regarding the distribution of disease pathways compared to Louvain clusters?

**Back:** Disease pathways are much more fragmented and overlapping than the rigid partitions created by clustering algorithms.

---
