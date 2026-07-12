# Knowledge Flashcards

## Card 1

**Front:** What percentage of enterprise data is estimated to be comprised of unstructured text?

**Back:** 80% to 90%.

---

## Card 2

**Front:** Which two technologies complement each other to unlock value from unstructured information in intelligent systems?

**Back:** Knowledge graphs (KGs) and large language models (LLMs).

---

## Card 3

**Front:** How has the integration of LLMs impacted human labor in the domain of information extraction?

**Back:** It has reduced the need for human labor by automating the understanding and extraction of natural language content.

---

## Card 4

**Front:** In knowledge graph construction, what does the term 'Knowledge Representation' refer to?

**Back:** How information is modeled so computers and humans can access it autonomously to solve tasks.

---

## Card 5

**Front:** What is the primary objective of 'Knowledge Learning' in the context of building knowledge graphs?

**Back:** To mine insights from unstructured documents using frameworks like NLP and LLMs.

---

## Card 6

**Front:** What organization's historical records are used as a case study for extracting domain-specific knowledge in Chapter 5?

**Back:** The Rockefeller Archive Center (RAC).

---

## Card 7

**Front:** Who encouraged John D. Rockefeller to create "permanent corporate philanthropies for the good of Mankind"?

**Back:** Frederick Taylor Gates.

---

## Card 8

**Front:** Why were the diaries of Rockefeller Foundation program officers considered a 'treasure trove' of knowledge?

**Back:** They contain detailed notes on meetings, phone calls, and professional observations often written in domain-specific nomenclature.

---

## Card 9

**Front:** What are the two fundamental processes involved in transforming textual data into structured knowledge graphs?

**Back:** Identifying entities (NER) and extracting the relationships that connect them (RE).

---

## Card 10

**Front:** Term: Named Entity Recognition (NER)

**Back:** Definition: An ML classification system trained to identify mentions of named entities in raw text and assign them to categories.

---

## Card 11

**Front:** How can NER improve a company's data management?

**Back:** By improving information management, data governance, and search capabilities.

---

## Card 12

**Front:** Why might a simple dictionary-based NER system fail in domain-specific tasks like the RAC case study?

**Back:** It cannot identify complex, specialized topics like "theory of Raman spectra" or "meteorological research."

---

## Card 13

**Front:** Term: Relation Extraction (RE)

**Back:** Definition: The process of identifying semantic relations between entity pairs within textual documents.

---

## Card 14

**Front:** What is the key architecture behind modern Large Language Models (LLMs)?

**Back:** The Transformer architecture.

---

## Card 15

**Front:** The ability to reuse linguistic patterns learned in a generic task for a specific task is called _____.

**Back:** Transfer learning.

---

## Card 16

**Front:** How does model complexity (number of parameters) affect the data requirements for LLMs?

**Back:** Larger neural models require fewer data samples to reach the same performance levels.

---

## Card 17

**Front:** What characterizes the 'data-centric' paradigm in modern AI development?

**Back:** Focusing on data engineering to improve the quality and quantity of data rather than just the model architecture.

---

## Card 18

**Front:** What is 'hallucination' in the context of Large Language Models?

**Back:** The tendency of the model to make up facts or provide false reasoning when training data is insufficient.

---

## Card 19

**Front:** What is 'prompt engineering'?

**Back:** The iterative process of identifying the most performant natural language task formulation for an LLM.

---

## Card 20

**Front:** What is the difference between 'zero-shot' and 'one-shot' learning in prompt engineering?

**Back:** Zero-shot uses only a task description, while one-shot provides one example to guide the model.

---

## Card 21

**Front:** How does the generative nature of LLMs simplify coreference resolution?

**Back:** The model implicitly links references like 'he' or initials to the full entity name based on the document context.

---

## Card 22

**Front:** What is 'prediction instability' in LLM-based knowledge extraction?

**Back:** When the same prompt produces different outputs if run repeatedly on the same text.

---

## Card 23

**Front:** How can a developer limit prediction instability in an LLM?

**Back:** By using careful prompt engineering and setting a very low temperature value (e.g., 0).

---

## Card 24

**Front:** Which LLM temperature setting is recommended for fact-focused tasks like extracting entities and relations?

**Back:** 0.

---

## Card 25

**Front:** In the context of prompt engineering, what is a 'Prompt Milestone'?

**Back:** A stage reached after aggregating enough improvements to feel significant progress toward the original task.

---

## Card 26

**Front:** Why is 'unit-testing' a prompt important during development?

**Back:** It ensures that updates to the prompt do not lose achievements or accuracy from previous iterations.

---

## Card 27

**Front:** What is the purpose of 'eyeballing' a mini-KG during the prompt engineering process?

**Back:** To discover opportunities for improvement by visually identifying systemic failures in a sample of documents.

---

## Card 28

**Front:** How does an LLM handle abbreviated entity names like 'U. of PA' if properly instructed?

**Back:** It uses its general training knowledge to resolve them to full forms like 'University of Pennsylvania.'

---

## Card 29

**Front:** What metrics are used for the quantitative evaluation of an LLM's KG extraction performance?

**Back:** Precision, recall, and F1 score.

---

## Card 30

**Front:** What is the concept of 'LLM as a judge' in evaluation?

**Back:** Using a powerful LLM instead of a human to mark predictions as correct, incorrect, or missing.

---

## Card 31

**Front:** When should a developer consider an 'Initial Explorative KG'?

**Back:** When they have a vague idea of content and need to identify the kinds of entities and relations available to mine.

---

## Card 32

**Front:** Why should prompt engineering examples be 'ambitious' according to the text?

**Back:** To leverage the LLM's deep linguistic understanding for cleaning and normalizing entity names from the start.

---

## Card 33

**Front:** What is a primary advantage of traditional NLP over LLMs regarding infrastructure?

**Back:** It requires simpler and cheaper infrastructure, often running on CPUs without the need for GPUs.

---

## Card 34

**Front:** What is a significant disadvantage of traditional NLP for complex tasks like Relation Extraction?

**Back:** It requires high initial investment in human-annotated training datasets and specialized data science expertise.

---

## Card 35

**Front:** What is the 'all-in-one' NLP advantage of LLMs?

**Back:** One pass through the data can achieve NER, RE, and coreference resolution simultaneously.

---

## Card 36

**Front:** Which factor makes prediction speed a disadvantage for LLMs compared to traditional NLP?

**Back:** Massive model sizes imply much slower processing speeds even on powerful GPUs.

---

## Card 37

**Front:** How does the prediction cost of LLMs change when using a fine-tuned custom model via OpenAI's API?

**Back:** The cost can be up to 10 times higher than using a standard model.

---

## Card 38

**Front:** What role can LLMs play in training custom traditional NLP models?

**Back:** They can increase the efficiency of the data annotation process by doing pre-annotation.

---

## Card 39

**Front:** In knowledge graph generation, the process of linking entities to structured knowledge bases is called Named Entity _____.

**Back:** Disambiguation (NED).

---

## Card 40

**Front:** Why is the chosen terminology for entity and relation classes in a prompt important?

**Back:** It helps the LLM clarify the task and provide more normalized, consistent outputs.

---

## Card 41

**Front:** According to the RAC case study, what type of metadata might be stored as a property of a TALKED_ABOUT relation?

**Back:** The sentiment of the conversation.

---

## Card 42

**Front:** What is the primary benefit of using JSON as the output format for KG extraction prompts?

**Back:** It allows each entity and relation to store properties or attributes in a structured way.

---

## Card 43

**Front:** The phenomenon where a higher temperature makes LLMs more creative is beneficial for text generation but detrimental for _____ tasks.

**Back:** Fact-focused (or extraction).

---

## Card 44

**Front:** Which archivist challenge involves managing data across different linguistic styles and historical periods?

**Back:** The archives challenge.

---

## Card 45

**Front:** In the RAC example, what was the role of 'program officers'?

**Back:** To pinpoint research areas worthy of grants and build networks of researchers.

---

## Card 46

**Front:** Why is 'Domain-Specific Nomenclature' a challenge for traditional NLP models?

**Back:** Standard models lack the specialized training needed to recognize unique technical terms and abbreviations.

---

## Card 47

**Front:** Concept: Optical Character Recognition (OCR)

**Back:** Definition: The process of digitizing historical or typewriter-written documents into machine-readable text.

---

## Card 48

**Front:** What is 'Named Entity Disambiguation' (NED)?

**Back:** Linking identified entities to specific nodes in a structured knowledge base to resolve identity.

---

## Card 49

**Front:** Chapter 6 focuses on the workflow from document processing to _____.

**Back:** Graph analytics.

---

## Card 50

**Front:** How does the 'model-centric' approach differ from the 'data-centric' approach in ML?

**Back:** Model-centric focuses on architecture and tuning; data-centric focuses on the quality of training data.

---

## Card 51

**Front:** What is the benefit of including a list of 'Relation Classes' in an extraction prompt?

**Back:** It guides the LLM to use consistent, normalized labels rather than creating redundant semantic variations.

---

## Card 52

**Front:** The process of 'Entity Resolution' in post-processing is minimized by LLMs because of their _____.

**Back:** Generative nature and language understanding.

---

## Card 53

**Front:** Under what condition should a developer stop prompt engineering and switch to fine-tuning?

**Back:** When prompt engineering is no longer yielding satisfactory results despite multiple iterations.

---

## Card 54

**Front:** Which billionaire industrialist is cited alongside Rockefeller as a pioneer of modern targeted philanthropy?

**Back:** Andrew Carnegie.

---

## Card 55

**Front:** What is the purpose of Figure 5.1 in the text?

**Back:** It provides a mental model for the path from domain-specific unstructured data to structured knowledge.

---

## Card 56

**Front:** In the RAC case study, what entity might the abbreviation 'M.I.T.' be resolved to by an LLM?

**Back:** Massachusetts Institute of Technology.

---

## Card 57

**Front:** What does a 'Human Annotation' step provide during the fine-tuning workflow?

**Back:** A high-quality dataset of expected outputs used to train the specialized model.

---

## Card 58

**Front:** Why is security an advantage for traditional NLP in high-security domains?

**Back:** It is easier and cheaper to deploy on-premises in isolated environments without needing cloud-based GPUs.

---

## Card 59

**Front:** The use of 'Shortened Titles' like 'Assoc. in Chem.' in the diaries is an example of which data extraction challenge?

**Back:** Domain-specific nomenclature (or abbreviations).

---

## Card 60

**Front:** What does the term 'synergy' refer to in the context of Chapter 5?

**Back:** The combination of traditional knowledge representation and modern AI capabilities (LLMs).

---

## Card 61

**Front:** How can a developer use 'LLMs as a judge' to monitor model drift?

**Back:** By consistently evaluating prediction quality against a gold standard using an automated LLM evaluation pipeline.

---

## Card 62

**Front:** According to the RAC analysis, what precedes the funding of an idea?

**Back:** Internal discussions and recommendations from influential scientists or previous grantees.

---

## Card 63

**Front:** Concept: Knowledge Learning

**Back:** Definition: Mining insights from unstructured documents using a combination of frameworks like NLP and LLMs.

---

## Card 64

**Front:** What prevents a dictionary-based system from identifying 'R.' as 'C. G. Rossby'?

**Back:** Lack of contextual understanding and coreference resolution capabilities.

---

## Card 65

**Front:** What is the impact of 'Data Quality' in the data-centric paradigm?

**Back:** High quality (and quantity) data leads to better performance in high-complexity ML models.

---

## Card 66

**Front:** In the diary example, 'he' referring to 'Mayer' is a problem of _____ resolution.

**Back:** Coreference.

---

## Card 67

**Front:** What is the benefit of 'Pre-annotation' in the fine-tuning process?

**Back:** It provides a starting point for human annotators, speeding up the creation of a training dataset.

---

## Card 68

**Front:** Which architecture component allows LLMs to handle massive datasets for transfer learning?

**Back:** Transformer architecture.

---

## Card 69

**Front:** Why is 'Task Formulation' in natural language considered the key to using LLMs?

**Back:** It allows the model to generate answers without requiring specialized model engineering.

---

## Card 70

**Front:** Which chapter in the source material introduces named entity disambiguation (NED)?

**Back:** Chapter 7.

---

## Card 71

**Front:** What is the 'Rockefeller Archive Center' (RAC) dedicated to studying?

**Back:** Philanthropy and research sectors influenced by foundations, donors, and civil society.

---

## Card 72

**Front:** How does the 'generative nature' of LLMs affect post-processing needs?

**Back:** It reduces the need for cleansing, normalization, and entity resolution.

---

## Card 73

**Front:** What is a major financial disadvantage of running predictions with fine-tuned LLMs on cloud APIs?

**Back:** Higher costs per prediction (e.g., OpenAI charging significantly more for custom models).

---

## Card 74

**Front:** In the summary, what is the 'cleaner KG' benefit attributed to?

**Back:** The generative nature and language understanding of LLMs.

---

## Card 75

**Front:** What is the primary role of Chapter 8 in the context of NED?

**Back:** Combining LLMs with domain ontologies for contextually aware disambiguation.

---

## Card 76

**Front:** True or False: LLMs can handle typos in names and deduce context better than traditional NER.

**Back:** True.

---

## Card 77

**Front:** Which century was the Rockefeller Foundation founded in?

**Back:** The 20th century (specifically 1913).

---

## Card 78

**Front:** What is the 'all-in-one NLP' benefit for LLMs?

**Back:** The elimination of separate, dependent processing steps like coreference and RE.

---

## Card 79

**Front:** What is the recommended approach if prompt engineering fails to produce satisfactory results?

**Back:** Invest time in fine-tuning the LLM.

---

## Card 80

**Front:** What does a 'stateless' API query refer to in the context of Listing 5.2?

**Back:** A query where each call is independent and doesn't rely on previous interactions.

---
