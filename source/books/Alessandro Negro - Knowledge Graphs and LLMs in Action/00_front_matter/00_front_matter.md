# Knowledge Graphs and IN ACTION

Alessandro Negro Giuseppe Futia Vlastimil Kus Fabio Montagna Forewords by Maxime Labonne Khalifeh AlJadda

![](images/1c10e370095e8c70e53b2c6ddcd9983807029a91e3bf3b66da6a2a8aa73cacce.jpg)

Users can ask questions using natural language.

![](images/bc6eaa29d554aa731bae3ac36f996edfa9fd9108886ff19d739c2ffff507dbe5.jpg)

Structured data contains entities and relationships.   
They must be mapped to the target schema.

Knowledge Graphs and LLMs in Action

### Knowledge Graphs and LLMs in Action

ALESSANDRO NEGRO

GIUSEPPE FUTIA

VLASTIMIL KŮS

FABIO MONTAGNA

FOREWORDS BY MAXIME LABONNE

AND KHALIFEH ALJADDA

For online information and ordering of this and other Manning books, please visit www.manning.com. The publisher offers discounts on this book when ordered in quantity. For more information, please contact

Special Sales Department   
Manning Publications Co.   
20 Baldwin Road   
PO Box 761   
Shelter Island, NY 11964   
Email: orders@manning.com

©2026 by Manning Publications Co. All rights reserved.

No part of this publication may be reproduced, stored in a retrieval system, or transmitted, in any form or by means electronic, mechanical, photocopying, or otherwise, without prior written permission of the publisher.

Many of the designations used by manufacturers and sellers to distinguish their products are claimed as trademarks. Where those designations appear in the book, and Manning Publications was aware of a trademark claim, the designations have been printed in initial caps or all caps.

Recognizing the importance of preserving what has been written, it is Manning’s policy to have the books we publish printed on acid-free paper, and we exert our best efforts to that end. Recognizing also our responsibility to conserve the resources of our planet, Manning books are printed on paper that is at least 15 percent recycled and processed without the use of elemental chlorine.

The author and publisher have made every effort to ensure that the information in this book was correct at press time. The author and publisher do not assume and hereby disclaim any liability to any party for any loss, damage, or disruption caused by errors or omissions, whether such errors or omissions result from negligence, accident, or any other cause, or from any usage of the information herein.

Manning Publications Co.

Development editor: Dustin Archibald

20 Baldwin Road

Technical editor: Dimitris Polychronopoulos

PO Box 761

Review editor: Radmila Ercegovac

Shelter Island, NY 11964

Production editor: Kathy Rossland

Copy editor: Tiffany Taylor

Development copy editor: Frances Buran

Proofreader: Olga Milanko

Technical proofreader: Sachin Panemangalore

Typesetter and Cover designer: Marija Tudor

To Aurora, Filippo, and Flavia —Alessandro

To my family—and especially my parents—for your unwavering love, support, and patience. To my friends and mentors, for walking with me, inspiring me, nudging me forward, and having the courage to be honest when it matters most. —Vlastimil

To Debora—my constant since we collided on the cosmic graph of life—thank you for   
walking beside me through every node and edge of this journey. I'm deeply grateful to my   
parents, Marieta and Cosimo, for your unwavering support—and for only occasionally asking what on earth a knowledge graph is. And to my brother, Dante—your perfectly timed reality checks have kept me grounded (and reasonably sane). Giuseppe

To my wife Fiorella and my children Giulio, Azzurra, and Arianna, who patiently endured countless evenings of “I'm almost done writing.” —Fabio

### brief contents

PART 1 FOUNDATIONS OF HYBRID INTELLIGENT SYSTEMS 1   
1 ■ Knowledge graphs and LLMs: A killer combination 3   
2 ■ Intelligent systems: A hybrid approach 17   
PART 2 BUILDING KNOWLEDGE GRAPHS FROM STRUCTURED   
DATA SOURCES 37   
3 ■ Create your first knowledge graph from ontologies 39   
4 ■ From simple networks to multisource integration 65   
PART 3 BUILDING KNOWLEDGE GRAPHS FROM TEXT 95   
5 ■ Extracting domain-specific knowledge from unstructured   
data 97   
6 Building knowledge graphs with large language models 115   
Named entity disambiguation 129   
8 NED with open LLMs and domain ontologies 180   
PART 4 MACHINE LEARNING ON KNOWLEDGE GRAPHS 207   
9 Machine learning on knowledge graphs: A primer approach 209   
10 Graph feature engineering: Manual and semiautomated   
approaches 233   
11 Graph representation learning and graph neural networks 272   
12 ■ Node classification and link prediction with GNNs 302   
PART 5 INFORMATION RETRIEVAL WITH KNOWLEDGE   
GRAPHS AND LLMS 335   
13 ■ Knowledge graph–powered retrieval-augmented   
generation 337   
14 Asking a KG questions with natural language 356   
15 ■ Building a QA agent with LangGraph 397   
appendix A Introduction to graphs 435   
appendix B Neo4j 447   
appendix C Building knowledge graphs from structured sources 461   
references 493   
index 505

### contents

forewords xv   
preface xvii   
acknowledgments xix   
about this book xxi   
about the authors xxv   
about the cover illustration xxvii

### PART 1 FOUNDATIONS OF HYBRID INTELLIGENT SYSTEMS

1.5 Building data-driven applications using KGs and LLMs 12 Example use case: Drug discovery and development 13 ■ Example use case: Conversational AI for customer support 13 ■ Deciding whether to use a KG 14

1.6 Knowledge graph technologies 14 Taxonomies and ontologies 15

### CONTENTS

![](images/3df1884ccbfdb166aba4507ad89eb018a3cc675d9fa995ef440bccd55112ca87.jpg)

1.7 How do we teach KGs and LLMs? 16   
2 Intelligent systems: A hybrid approach 17   
2.1 What is intelligence? 18   
2.2 Designing an intelligent system 19   
What is an intelligent system? 20 ■ Categories of intelligent   
systems 20 ■ Characteristics of an intelligent system 23   
2.3 Knowledge acquisition and representation 24   
2.4 Reasoning 27   
2.5 Reasoning engines 30   
Limitations of a pure deductive reasoning engine 31 ■ Using   
inductive reasoning and ML 32 ■ The role of LLMs in the   
reasoning engine 33   
2.6 A KG approach to IASs 33   
2 BUILDING KNOWLEDGE GRAPHS FROM   
STRUCTURED DATA SOURCES 37   
3 Create your first knowledge graph from ontologies 39   
3.1 Knowledge graph building: Warmup 41   
Business and domain understanding 41 ■ Data   
understanding 43   
3.2 Understanding knowledge graph technologies 46   
RDF or LPG? A goal-driven discussion 47 ■ Representing edge   
properties with RDF and LPG 49   
3.3 Building a knowledge graph 52   
Ontology ingestion and processing with neosemantics 52   
Annotation ingestion and processing 55   
3.4 Querying the data 59   
3.5 Reasoning over the KG 62   
4 From simple networks to multisource integration 65   
4.1 Biomedical knowledge graphs and applications 66   
4.2 Multi-omic applications of KGs 67   
Creating a KG from the PPI and protein-disease networks 69   
High-level analysis of the resulting KGs 73 ■ Domain-specific   
analysis of the PPI and disease KG 76

![](images/d00ff192c6782e8fd2e3d392267302c6eae5ae707a003cf890599bdfd54893e9.jpg)

![](images/b6e34c9f2bb28e33aaf19e106c94f06cb1e5cf15c06efc6332082ad727f7fb9c.jpg)

### 4.3 Pharmaceutical applications of KGs 80

Deep analysis of the Hetionet knowledge graph 84 ■ LLM-assisted interpretation of pathway analysis results 88

4.4 Clinical applications of KGs 90

LLM-guided clinical decision support analysis 93

#### PART 3 BUILDING KNOWLEDGE GRAPHS FROM TEXT ... 95

![](images/3d79b3342c86671bdee996d37d6e980b44f7eb815750c10c4eed301c06219a2d.jpg)

5 Extracting domain-specific knowledge from unstructured data 97 5.1 The archives challenge 98 5.2 Key concepts of knowledge extraction 99 Recognizing named entities 100 ■ Extracting relations 101 5.3 Building KGs with large language models 101 Using LLMs 102 ■ Prompt engineering examples 104 Prompt engineering guidelines 109 ■ KG building: Traditional NLP or LLMs? 112 Building knowledge graphs with large language models 115 6.1 Transforming an archive to a KG 116 Graph modeling 118 ■ Creating a metagraph 119 Normalization and cleansing 119 ■ Graph-based entity resolution 120 6.2 Intellectual network analysis: The value of graphs 122 6.3 Next steps in the Rockefeller Archive Center project 126 6.4 The value of knowledge graphs in the LLM era 127   
7 Named entity disambiguation 129 7.1 From recognition to disambiguation 129 7.2 Understanding named entity disambiguation 132 7.3 Domain-based NED and LLMs 136 7.4 Business and domain understanding 138 Context 138 ■ Use case definition 140 7.5 Understanding the data 141 Unstructured data 141 ■ Domain ontologies 142 7.6 Building a SoHO knowledge graph 146 Defining the schema 147 ■ Processing and ingesting documents 148 ■ Disambiguating and ingesting medical

entities 149 ■ Processing, loading, and mapping ontologies 152   
Generating entity co-occurrences 157   
7.7 KG-based use cases 158   
Conceptual search 159 ■ Structured knowledge-based search 162   
KG-based interpretability and discovery 166 ■ Uncovering new   
knowledge 174   
8 NED with open LLMs and domain ontologies 180   
8.1 Understanding limitations of traditional NED systems 180   
8.2 Ingesting the domain ontology 182   
8.3 Setting up the model with Ollama and Llama 3.1 8B 186   
8.4 End-to-end NED process 187   
Named entity recognition 188 ■ Candidate selection 192   
Candidate disambiguation 194   
8.5 Conclusions 205   
ART 4 MACHINE LEARNING ON KNOWLEDGE   
GRAPHS 207   
Machine learning on knowledge graphs: A primer   
approach 209   
9.1 Machine learning on graphs: Why? 210   
9.2 Machine learning on graphs: What? 211   
Node classification 211 ■ Link prediction (a.k.a. relationship   
prediction) 214 ■ Clustering and community detection 216   
Graph classification 217   
9.3 Machine learning on graphs: How? 219   
Node classification and link prediction 220 ■ Graph   
classification 228 ■ Graph clustering 229   
10 Graph feature engineering: Manual and semiautomated   
approaches 233   
10.1 Manual node features 235   
Degree 237 ■ Triangles 239 ■ Density 241 ■ Geodesic (or   
shortest) path 242 ■ Closeness 244 ■ Betweenness 247   
PageRank 249 ■ Prediction 250   
10.2 Manual relationship features 254   
Node-based representation 255 ■ Path-based features 256

#### CONTENTS

![](images/2d5958ed5b0f209d835cebeab85bbb61918b385c6704524836bc44d22aad5bbf.jpg)

### 10.3 Semiautomated feature extraction 263

Performing ReFeX manually 266 ■ Performing ReFeX automatically with code 268

## 11 Graph representation learning and graph neural

networks 272

11.1 Embeddings in graph representation learning 273 Understanding graph embeddings: From discrete to continuous 274 Real-world applications and examples 278

11.2 The encoder–decoder model 279

The encoder: Converting graph structure to vectors 279

The decoder: Reconstructing graph properties 280 ■ The power of the framework 280 ■ Node2Vec: An example of an encoder–decoder framework 280

11.3 Shallow embeddings: A first approach to graph representation 283 Understanding shallow embeddings 283 ■ Limitations of shallow embeddings 284

11.4 Embeddings in knowledge graphs 285 Loss function 285 ■ Multirelationship decoder 288

11.5 Message passing and graph neural networks 289

The message-passing framework: A neural conversation 289 ■ Motivation and intuition: Why message passing works 290 ■ The basic GNN model 291 ■ Message passing with self-loops 291

11.6 Generalized aggregation and update methods 292 Neighborhood normalization 293 ■ Neighborhood attention 294 ■ Multihead attention and transformer connections 294 ■ Generalized update methods 297

11.7 The synergy of GNNs and LLMs 299

12 Node classification and link prediction with GNNs 302

12.1 Node classification for anti-money laundering applications 303

Input data 304 ■ Graph processor: Data preparation 305 Graph processor: Homogeneous PyG graph 307 Encoder–decoder architecture 310 ■ Evaluation and analysis 313

### 12.2 Link prediction for movie recommendations 317

Input data 318 ■ Graph processor: Data preparation 319 Graph processor: Heterogeneous PyG graph 321 Encoder–decoder architecture 326 ■ Evaluation and analysis 330

#### PART 5 INFORMATION RETRIEVAL WITH KNOWLEDGE GRAPHS AND LLMS 335

## 13 Knowledge graph–powered retrieval-augmented generation 337

13.1 AI agents 338

13.2 Chatting with the LLM 339

13.3 Challenges in the production environment 341

13.4 Chatting with the AI about private data 342 Retrieval-augmented generation 343 ■ Vector-based RAG limitations 345 ■ Graph RAG 347 ■ Reasoning agents 351 Let’s chat with our KG 352

## 14 Asking a KG questions with natural language 356

14.1 Querying a knowledge graph in the policing domain 357 Enabling domain experts with knowledge graphs 357

14.2 RAG for KG querying: Capabilities and challenges 358 RAG effectiveness with complete context 359 ■ RAG fragility with incomplete retrieval 361

14.3 Schema-based approach for querying KGs 363 Understanding and using graph schemas 364

14.4 Think like an expert: Using metadata for enhanced querying 366

14.5 Intent detection: Understanding user expectations 367 Classifying by visualization type 368 ■ Is it data, documentation, or just complaining? 372

14.6 From schema to LLM-ready context 376 Schema extraction and representation 377 ■ Enriching schemas with descriptive annotations 380 ■ A practical approach to schema representation 382

14.7 It’s time to think: Understanding LLM reasoning 383 The order matters: Answer first vs. reasoning first 384 ■ Thinking in queries: From text to Cypher 386 ■ Structuring output for reliable query generation 391

14.8 Response summarization: From results to insights 392

## 15 Building a QA agent with LangGraph 397

15.1 Building the LangGraph pipeline 398

System architecture overview 399 ■ Configuring pipeline components 401 ■ Schema translation service 404 State management design 408 ■ Pipeline agent implementation 409 ■ Pipeline integration layer 415

15.2 Streamlit application 417 Application overview 418 ■ LangGraph integration 420

15.3 Expert-emulating investigation 422 Identifying the initial case 423 ■ Spatial analysis of surveillance coverage 425 ■ Vehicle pattern detection 427 ■ Context-aware request refinement 428 ■ Historical record analysis 430

15.4 Future directions and enhancements 432 Learning from use 432 ■ Enhancing core capabilities 433 Advanced evolution paths 433

appendix A Introduction to graphs 435

appendix B Neo4j 447

appendix C Building knowledge graphs from structured sources 461

references 493

index 505

### forewords

Working with graph neural networks and large language models over the years has taught me that each technology has profound strengths and equally profound limitations. Graph neural networks excel at understanding structured relationships but struggle with natural language interfaces. Large language models can engage in sophisticated conversations but frequently hallucinate facts and lack reliable grounding in structured knowledge.

Knowledge Graphs and LLMs in Action tackles an important challenge in AI: how do we combine these technologies to build systems that are both intelligent and trustworthy? Alessandro Negro, Giuseppe Futia, Vlastimil Kůs, and Fabio Montagna don't just theorize about this convergence; they provide practical recipes for making it work. Their approach bridges the gap between the precision of knowledge graphs and the accessibility of natural language, creating systems that can reason over complex data and explain their conclusions.

What impressed me most about this work is its rare emphasis on real-world implementation. The authors walk you through building knowledge graphs from messy, unstructured data and then show how to integrate them with language models for applications in healthcare, law enforcement, and beyond. The examples are concrete and the code is production-ready, making this both a learning resource and a practical guide.

The technical depth here is substantial, covering everything from graph construction to advanced retrieval systems, but the authors never lose sight of the practical goal: building AI systems that can serve as reliable advisors in critical decisions. This hybrid approach addresses the reliability and explainability challenges that have limited AI deployment in high-stakes environments.

If you're working on AI systems that need to be both powerful and trustworthy, Knowledge Graphs and LLMs in Action provides a clear framework for achieving it. The combination of knowledge graphs and language models represents a significant step toward AI that can handle complexity while maintaining the transparency and reliability that real-world applications demand.

—MAXIME LABONNE

HEAD OF POST-TRAINING, LIQUID AI

As a data science leader and passionate advocate for knowledge graphs, I’m thrilled to recommend Knowledge Graphs and LLMs in Action. We are witnessing a transformative moment in AI, shaped by the rise of generative AI and large language models (LLMs). Systems like Gemini and ChatGPT have opened the doors to natural language interaction at scale, offering a glimpse of intelligent machines. Yet we know these models are not without flaws. Hallucinations, outdated knowledge, limited transparency, and a lack of contextual grounding remain real challenges.

Addressing concerns like these is where knowledge graphs (KGs) shine, not just as a complement to LLMs, but as a necessary foundation for building accurate, explainable, and context-aware systems. This book demonstrates how the convergence of KGs and LLMs creates a powerful synergy, mitigating each other’s weaknesses while unlocking their full potential.

The authors—Alessandro Negro, Vlastimil Kůs, Giuseppe Futia, and Fabio Montagna—bring years of hands-on experience and consulting expertise. Their work moves beyond theory to deliver actionable, production-ready insights grounded in real-world applications.

This book is more than a reference for knowledge graphs and LLMs. It’s a practical toolkit for developing intelligent systems that enhance, not replace, human decision-making across domains like healthcare, finance, and law enforcement.

In an age where AI must be transparent, contextual, and trustworthy, this book is both timely and essential. It belongs on the shelf of every data scientist, engineer, architect, and knowledge-driven professional ready to build the next generation of intelligent systems.

Thank you, Alessandro, Vlastimil, Giuseppe, and Fabio, for this insightful and practical book!

—KHALIFEH ALJADDA

When I was nearing completion of my previous book, Graph-Powered Machine Learning, I reached out to my acquisitions editor, Mike Stephens, with a proposal for a natural continuation. That earlier work introduced knowledge graphs and demonstrated how they could be built using natural language processing, but many readers pointed out that graph neural networks were a significant missing piece. My proposed book would fill that gap while extending the knowledge graph story further, including detailed analysis and building techniques.

Mike accepted the proposal, and I embarked on a new adventure with the working title Knowledge Graphs Applied. Recognizing the scope of the challenge, I invited three colleagues from GraphAware—Fabio, Giuseppe, and Vlastimil—to join the effort, confident that their combined expertise would be invaluable. I naively thought that if one author could write a book in four years, four authors could complete a book in just a year. That assumption proved as flawed as expecting nine women to deliver a baby in one month.

Reality had other plans. Over the past years, significant changes swept through the technology landscape. Large language models (LLMs) and generative AI disrupted the field entirely, and knowledge graph practitioners suddenly found themselves with unprecedented opportunities to use this established technology in revolutionary ways. We initially planned to build on existing natural language processing (NLP) tools like BERT, but these were rapidly being superseded by LLM capabilities that opened new possibilities for building, querying, and analyzing knowledge graphs.

This was precisely where many practitioners, ourselves included, were struggling. Rather than resist this transformation, we decided, together with Mike and Dustin Archibald (our development editor), to embrace it. We adjusted our title to Knowledge Graphs and LLMs in Action and substantially revised the content to position LLMs as an integral component of our ultimate goal: intelligent advisor systems that empower humans in performing complex decision-making tasks. This pivot required extensive refactoring and a fundamental shift in our approach, but the result exceeded our expectations.

The book you are reading has evolved into a manifesto for the power of hybrid sys tems. It demonstrates how combining these technologies—knowledge graphs, which are well established, and LLMs, which are newly emerged—creates a flywheel effect that delivers remarkable long-term results. Knowledge graph practitioners will discover how to use LLM capabilities for greater impact, and LLM practitioners will learn techniques that address some of the major limitations of language models.

We invite you to join us on this journey toward more intelligent, more reliable, and more human-centered AI systems.

## acknowledgments

This book took almost five years to complete, and during that time, many things changed around us, both professionally and personally. The technology landscape has transformed dramatically since we began writing: LLMs have fundamentally shaped our professions.

Writing a book requires dedication and countless hours, usually outside normal working hours, late at night, and on weekends and holidays. So first and foremost, we need to thank our families and all the people who somehow received a “no” or suffered a delay because of this book.

To my co-authors—Fabio, Giuseppe, and Vlastimil—thank you for embarking on this adventure with me. Each of you brought unique expertise and perspectives that made this book infinitely better than what I could have accomplished alone. Your dedication to excellence and willingness to adapt as the technology landscape shifted around us were nothing short of remarkable.

We owe an enormous debt of gratitude to the team at Manning Publications. In particular, Mike Stephens, Manning's associate publisher, not only accepted our book proposal but also provided invaluable guidance during the pivotal transformation from our original direction (Knowledge Graphs Applied) to what you are now reading (Knowledge Graphs and LLMs in Action). A special thank you to Dustin Archibald, our development editor, who followed us step by step with enormous patience and consistently provided excellent advice to make this a better book; your commitment to quality and your understanding during the extensive refactoring process made all the difference. We also want to thank the production and marketing teams at Manning— there are so many talented individuals that it's impossible to mention them all, but they are the reason Manning books are such high quality and so well presented.

Our sincere appreciation goes to Dimitris Polychronopoulos, our technical editor, whose meticulous attention to detail and expert feedback significantly improved the technical accuracy and clarity of this work. Dimitris is an R&D scientist and entrepreneur specializing in genomics and data-driven drug discovery. With roles across biotech and big pharma, he has led innovative work on applying knowledge graphs and AI to uncover novel targets in oncology and chronic liver diseases.

We also extend our gratitude to all the reviewers who provided valuable feedback throughout the development process, including those who shared comments and suggestions online: Alexey Ott, Angelo Simone Scotto, Ayush Tomar, Avinash Tiwari, Chalamayya Batchu, Charles Ivie, Chris Viner, Dan McCreary, David Cronkite, David Meza, Floris Bouchot, Gajendra Babu Thokala, Gourav Sengupta, Guillaume Alleon, James J Byleckie, Jeremy Chen, Kristof Leroux, Kumar Abhishek, Lawrence Nderu, Maria Ana, Nicolas Bievre, Or Golan, Ozan Evkaya, Pethuru Raj, R. P. Shrivastava, Richard Vaughan, Robert Wardenga, S. S. Narendran, Sachin Panemangalore, Samantha Berk, Shailja Gupta, Simeon Leyzerzon, Sophia Shvets, Sumit Pal, and Suvarsha Rai. Your insights and constructive criticism helped shape this book into its final form.

A special acknowledgment goes to Khalifeh and Maxime, who kindly agreed to write forewords for this book. They received an almost-final copy when the book was ready for production, requiring them to work under tight deadlines to provide the thoughtful forewords you'll find at the beginning—no small feat for professionals as busy as they are. Khalifeh and Maxime are among the most knowledgeable experts we know in their fields. Their endorsement carries particular weight because of their extensive experience in bringing these technologies to real-world applications, and their ongoing work continues to inspire us and the broader community.

We also want to acknowledge the companies that provided us with the knowledge, experience, and opportunities we needed to create this book. GraphAware, in particular, has been instrumental in shaping our understanding of real-world graph applications and the challenges organizations face when implementing these technologies at scale.

Finally, we extend our appreciation to the broader community of researchers, practitioners, and open source contributors whose work made many of our examples possible. The datasets, tools, and frameworks that power the demonstrations in this book represent countless hours of effort from dedicated individuals who chose to share their knowledge with the world.

This book exists because of all of you. Thank you for making this journey possible.

### about this book

Knowledge Graphs and LLMs in Action is a comprehensive guide to building hybrid intelligent systems that combine the structured reasoning capabilities of knowledge graphs (KGs) with the natural language understanding of large language models (LLMs). This book demonstrates how these complementary technologies can work together to create more powerful, reliable, and explainable AI solutions that address real-world challenges across various domains.

### Who should read this book

This book is designed for machine learning engineers, data scientists, graph experts, and AI engineers who want to harness the synergistic power of KGs and LLMs. Whether you're working with structured enterprise data, building recommendation systems, developing fraud detection algorithms, or creating question-answering applications, this book will show you how to use both technologies to achieve better results than either could deliver alone.

If you're a data scientist looking to enhance your models with structured knowledge, a machine learning engineer seeking to reduce hallucinations in LLM applications, or an AI practitioner interested in building explainable and verifiable systems, this book provides the practical guidance you need. Although some familiarity with machine learning concepts and graph databases is helpful, the book introduces all necessary concepts and builds complexity gradually.

### How this book is organized: A roadmap

The book has 15 chapters organized into 5 parts, progressing from foundational concepts to advanced implementations.

Part 1 establishes the theoretical and practical foundations for hybrid intelligent systems:

Chapter 1 introduces the powerful combination of KGs and LLMs, demonstrating their complementary nature through concrete examples and use cases.

Chapter 2 explores fundamental concepts of intelligent systems, diving deep into knowledge representation and reasoning strategies, and illustrating how KGs and LLMs work together in practice.

Part 2 focuses on building KGs from structured data sources:

Chapter 3 demonstrates KG construction through a healthcare example, showing how to help clinicians diagnose rare diseases using the Human Phenotype Ontology.

Chapter 4 expands on these foundations with advanced analysis methodologies, including community detection algorithms and multisource integration across biomedical applications.

Part 3 tackles the challenging realm of extracting knowledge from unstructured text:

Chapter 5 demonstrates the fundamental pipeline for converting text to KGs using both traditional natural language processing (NLP) and modern LLMbased methods through a case study at the Rockefeller Archive Center.

Chapter 6 expands on document processing workflows, from OCR scanning to sophisticated graph analytics for identifying research networks and influence patterns.

 Chapter 7 explores named entity disambiguation in healthcare regulation, showing how to link entities to structured knowledge bases like the Unified Medical Language System (UMLS) and Systematized Nomenclature of Medicine (SNOMED).

 Chapter 8 introduces an innovative approach to disambiguation that combines open LLMs with domain ontologies for enhanced accuracy.

Part 4 explores machine learning applications on KGs:

 Chapter 9 introduces fundamental concepts for applying machine learning to KGs and establishes the theoretical foundation for learnable representations.

Chapter 10 illustrates feature engineering approaches through practical examples in fraud detection and drug repurposing.

 Chapter 11 advances into graph neural networks, showing how these architectures automatically learn optimal representations from graph structures.

Chapter 12 demonstrates real-world applications through anti-money laundering and movie recommendation systems.

Part 5 brings everything together in practical information retrieval systems:

Chapter 13 explores integrating KGs with LLMs through retrieval augmented generation, demonstrating graph RAG systems.

Chapter 14 shows how to build sophisticated question-answering systems that emulate domain expert reasoning through a law enforcement example.

Chapter 15 provides a complete implementation using LangGraph and Streamlit, demonstrating how to build production-ready systems.

The book is designed to be read sequentially for comprehensive understanding, but experienced practitioners can focus on specific parts based on their immediate needs. Beginners should start with parts 1 and 2 to establish foundational knowledge before exploring specialized applications in later parts.

### About the code

This book contains extensive source code examples demonstrating practical implementations of KG and LLM integration. Code appears both in numbered listings and in line with explanatory text, formatted in a fixed-width font like this to distinguish it from regular content. Sometimes code is also in bold to highlight relevant pieces.

In many cases, the original source code has been reformatted; we've added line breaks and reworked indentation to accommodate the available page space in the book. In rare cases, even this was not enough, and listings include line-continuation markers (➥). Additionally, comments in the source code have often been removed from the listings when the code is described in the text. Code annotations accompany many of the listings, highlighting important concepts.

All source code examples are available for download from the book's GitHub repository at https://github.com/alenegro81/knowledge-graphs-and-llms-in-action. The repository includes complete implementations for each chapter's examples, along with setup instructions and data files needed to run the code.

The examples require Python 3.8 or higher and various libraries, including Neo4j, NetworkX, transformers, LangChain, and Streamlit. Specific requirements and installation instructions are provided in each chapter and the repository's documentation. Some examples also require access to OpenAI APIs or other language model services, with instructions for setting up the necessary credentials.

You can get executable snippets of code from the liveBook (online) version of this book at https://livebook.manning.com/book/knowledge-graphs-and-llms-in-action. The complete code for the examples in the book is also available for download from the Manning website at https://www.manning.com/books/knowledge-graphs-and -llms-in-action.

### liveBook discussion forum

Purchase of Knowledge Graphs and LLMs in Action includes free access to liveBook, Manning’s online reading platform. Using liveBook’s exclusive discussion features, you can attach comments to the book globally or to specific sections or paragraphs. It’s a snap to make notes for yourself, ask and answer technical questions, and receive help from the authors and other users. To access the forum, go to https://livebook .manning.com/book/knowledge-graphs-and-llms-in-action/discussion.

Manning’s commitment to our readers is to provide a venue where a meaningful dialogue among individual readers and between readers and the authors can take place. It is not a commitment to any specific amount of participation on the part of the authors, whose contribution to the forum remains voluntary (and unpaid). We suggest you try asking the authors some challenging questions lest their interest stray! The forum and the archives of previous discussions will be accessible from the publisher’s website as long as the book is in print.

![](images/85bc5cc19298711b55482a9ee909fbb0d55c07c73e5cca4b8e7402d193aa77fd.jpg)

### about the authors

ALESSANDRO NEGRO is the chief scientist at GraphAware, where he supervises the science and technology area responsible for delivering Hume, a mission-critical analytics platform that uses knowledge graphs (KGs) at its core. He holds a Ph.D. in computer science and has successfully deployed machine learning systems combined with graphs for numerous organizations across various industries. Dr.

Negro is the author of Graph-Powered Machine Learning (Manning, 2021). His recent work focuses on integrating LLMs with KGs to create more reliable and explainable AI systems at scale. Beyond his role at GraphAware, Alessandro actively mentors and advises startups, helping organizations in specialized domains create custom models tailored to their unique requirements.

![](images/b11c90bcfc7268f75e15987d884a1771b5459b29eca674b7669ef43e80ab4155.jpg)

GIUSEPPE FUTIA is a data scientist and research fellow with expertise in knowledge graphs, large language models, and graph neural networks. He holds a Ph.D. in computer engineering and bridges academia and industry through extensive experience in research and applied innovation. His work focuses on developing knowledgedriven intelligent systems that integrate symbolic and statistical AI to

enable advanced decision-making across diverse domains.

![](images/50da90070983efeb24220ed487434c9ab1137079ec02bdbaf8418715c5e7d338.jpg)

VLASTIMIL KŮS is a lead AI engineer and data science leader specializing in knowledge graphs, Agentic AI, NLP, and unstructured data. With a background in subnuclear physics research and more than a decade of experience in advanced data science and machine learning, he combines deep technical expertise with a strong focus on delivering business impact.

![](images/caf662f83179d1358d0d60b1ae4b672ceb87f1d3caf292a4189490be3053e66b.jpg)

FABIO MONTAGNA is a senior graph and AI engineer at GraphAware, with more than 15 years of experience in research-driven software engineering across neuroscience, operational oceanography, and natural language processing. As a pioneer in bridging scientific discovery and commercial application, he specializes in transforming complex research concepts into scalable production systems that deliver busi-

ness value. His dual role as product strategist and technical expert enables him to translate theoretical breakthroughs into practical solutions across healthcare, earth sciences, and commercial applications. Beyond GraphAware, Fabio has been collaborating with neurologists in pharmaceutical research for Parkinson's disease treatment evaluation.

### about the cover illustration

The image featured on the cover of this book is called “Mařaťanka” by Joža Uprka (1861–1940), a Czech painter renowned for his vivid portrayals of folk life in southern Moravia, a historic wine-growing region in the Czech Republic. The painting depicts a young woman from the village of Mařatice, whose name echoes the ancient designation “sons of Maria” for its inhabitants. Dressed in a traditional local costume, she carries a tray of grapes and fruit aloft, embodying the spirit of harvest, abundance, and the enduring rhythm of historical rural life. It is a celebration of shared European traditions rooted in the land, the seasons, and community rituals. Like wine itself, these themes transcend borders and bring people together, mirroring the Czecho-Italian nature of the co-authoring team.