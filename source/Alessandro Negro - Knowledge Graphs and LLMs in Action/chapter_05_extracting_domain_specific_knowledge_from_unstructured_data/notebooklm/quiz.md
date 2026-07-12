# Knowledge Quiz

## Question 1
According to the source material, approximately what percentage of enterprise data exists in an unstructured format today?

- [ ] 10% to 20%
- [ ] 50%
- [x] 80% to 90%
- [ ] Over 99%

**Hint:** Consider the high volume of daily communications like emails and chats compared to formal databases.

## Question 2
Which process is specifically defined as identifying semantic connections between pairs of entities within a document?

- [ ] Named Entity Recognition (NER)
- [x] Relation Extraction (RE)
- [ ] Optical Character Recognition (OCR)
- [ ] Entity Resolution

**Hint:** Think about the step that provides the 'edges' in a knowledge graph.

## Question 3
What is the primary risk associated with LLMs when they generate 'facts' or reasoning that have no justification in their training data?

- [ ] Overfitting
- [ ] Catastrophic Forgetting
- [x] Hallucination
- [ ] Tokenization Error

**Hint:** The term describes a deceptive sensory experience in humans applied to AI outputs.

## Question 4
For fact-focused tasks like extracting entities for a Knowledge Graph, what temperature setting is recommended for the LLM configuration?

- [x] 0
- [ ] 0.7
- [ ] 1.0
- [ ] $-1$

**Hint:** Lower values result in more predictable and stable outputs.

## Question 5
How does the 'generative nature' of LLMs simplify the task of coreference resolution compared to traditional NLP?

- [ ] It requires a separate pre-processing model for pronouns.
- [x] It resolves references implicitly using the document's entire context.
- [ ] It only works if pronouns are manually tagged by humans first.
- [ ] It eliminates the need for context entirely.

**Hint:** Think about how a human understands who 'he' is by reading the whole paragraph.

## Question 6
In the context of prompt engineering, what does 'one-shot learning' refer to?

- [ ] Fine-tuning the model weights with a single massive dataset.
- [x] Providing the model with a single example of the task within the prompt.
- [ ] Giving the model only a task description without any examples.
- [ ] The ability of a model to learn from a single GPU cycle.

**Hint:** Focus on the quantity of examples provided to the model during the prompt process.

## Question 7
Which of the following is a noted disadvantage of using LLMs over traditional NLP models for Knowledge Graph construction?

- [ ] Higher initial domain customization costs.
- [x] Slower prediction speeds requiring GPU infrastructure.
- [ ] Requirement for high-level data science expertise for simple prompts.
- [ ] Lower quality of contextual understanding.

**Hint:** Consider the hardware and time requirements for running billions of parameters.

## Question 8
What is the recommended approach if iterative prompt engineering fails to produce satisfactory results for a complex domain?

- [ ] Abandon the project entirely.
- [x] Invest time in fine-tuning the LLM.
- [ ] Switch exclusively to traditional NLP with no LLM involvement.
- [ ] Reduce the complexity of the data to fit the prompt.

**Hint:** Think about the more intensive process that follows prompting in the mental model.

## Question 9
In the RAC case study, why was 'Department of Physics' considered a failure in early extraction iterations?

- [ ] It is not a valid named entity.
- [x] It lacked university-level disambiguation (aliasing).
- [ ] The LLM could not recognize 'Physics' as a field.
- [ ] It was incorrectly categorized as a person.

**Hint:** Consider what information is missing if multiple universities have the same department name.

## Question 10
What 'paradigm' shift does the source mention as having gained traction by focusing on data engineering rather than model architecture?

- [ ] Model-centric paradigm
- [x] Data-centric paradigm
- [ ] Algorithmic-centric paradigm
- [ ] Prompt-centric paradigm

**Hint:** Think about the approach that emphasizes 'cleaning' the input rather than 'tweaking' the code.
