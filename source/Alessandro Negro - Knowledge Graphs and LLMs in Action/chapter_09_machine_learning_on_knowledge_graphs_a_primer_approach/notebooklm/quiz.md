# Graph Quiz

## Question 1
Why do traditional machine learning algorithms often struggle with raw graph data without specific adaptations?

- [x] Traditional algorithms assume that data points are independent and identically distributed (i.i.d.).
- [ ] Graphs cannot be represented in matrix or tensor formats used by traditional algorithms.
- [ ] Traditional algorithms are unable to handle categorical data types typically found in graphs.
- [ ] Graph data is strictly unsupervised, whereas traditional algorithms are strictly supervised.

**Hint:** Consider how traditional models view the relationship between individual rows in a dataset.

## Question 2
In the context of link prediction training, why is it necessary to sample 'negative cases'?

- [x] To provide the model with examples of pairs of nodes that do not have an existing connection.
- [ ] To reduce the computational load of processing the entire graph adjacency matrix.
- [ ] To identify nodes that should be removed from the knowledge graph due to low connectivity.
- [ ] To ensure that only nodes with high homophily are considered for future relationships.

**Hint:** Think about what a binary classifier needs to see to understand what a 'missing' relationship looks like.

## Question 3
Which of the following defines a 'graph-focused' machine learning task according to the source material?

- [x] The data consists of a set of multiple graphs, where each data point is an entire graph.
- [ ] The entire dataset is represented as one single large graph with nodes as data points.
- [ ] The algorithm focuses exclusively on the weights of relationships rather than node attributes.
- [ ] The goal is to predict the internal structure of a node based on the overall graph topology.

**Hint:** Distinguish between analyzing elements within one graph versus analyzing many different graphs.

## Question 4
When converting relationships into vectors for link prediction, which operator is commonly used to combine the feature vectors of two nodes?

- [x] Hadamard product
- [ ] Linear Regression
- [ ] Adjacency Multiplication
- [ ] Euclidean Aggregation

**Hint:** This operator performs element-wise multiplication of vectors.

## Question 5
Why is feature scaling (standardization) critical when preparing graph-based features for machine learning?

- [x] To prevent features with larger numerical ranges from dominating distance-based calculations.
- [ ] To convert non-linear graph patterns into a linear format for logistic regression.
- [ ] To ensure that every node in the graph has the same number of features.
- [ ] To remove negative values from the feature matrix to satisfy Bayesian algorithms.

**Hint:** Consider the impact of using node degree (integers) alongside normalized centrality scores (decimals) in a geometric model.

## Question 6
What is a distinguishing characteristic of graph clustering (community detection) compared to node classification?

- [x] It is typically an unsupervised task that does not require a training phase with labeled examples.
- [ ] It requires the conversion of the graph into a feature matrix before any processing can occur.
- [ ] It focuses exclusively on node attributes rather than the network's topology.
- [ ] It always produces a single definitive classification that remains consistent across all runs.

**Hint:** Think about the role of ground-truth labels and the training process in these two tasks.

## Question 7
In the Zachary Karate Club experiment, what did the nodes' final shades (colors) in the visualization represent?

- [x] Ground-truth labels indicating which faction each member joined after the club split.
- [ ] The degree centrality of each member within the club's social network.
- [ ] The output of the Louvain community detection algorithm.
- [ ] The frequency of communication between members before the instructor resigned.

**Hint:** This information serves as the target variable for the classification task.

## Question 8
Which statement best describes the 'homophily' pattern in social networks?

- [x] Nodes tend to connect with others who share similar interests, attributes, or behaviors.
- [ ] Nodes preferentially connect to those with vastly different characteristics.
- [ ] Nodes with similar neighborhood structures share functional properties regardless of their distance.
- [ ] The tendency for highly connected nodes to become even more connected over time.

**Hint:** Think about the phrase 'birds of a feather flock together' applied to graph nodes.

## Question 9
According to the primer, what is the output of the 'training phase' in the node classification flow?

- [x] A machine learning model.
- [ ] A set of community mappings.
- [ ] A vector representation of the entire graph.
- [ ] The link probability for missing relationships.

**Hint:** The training phase creates the 'actor' that performs the subsequent classification.

## Question 10
What is a key requirement for the featurization process used during the 'prediction phase' of a graph ML task?

- [x] It must be exactly the same as the featurization approach used during the training phase.
- [ ] It must use only node attributes to avoid leaking structural information into the model.
- [ ] It must be performed using a manual approach, even if training was automated.
- [ ] It should only include nodes that were already present in the training set graph.

**Hint:** Consistency is vital for the model to recognize the patterns it was taught.

## Question 11
How does 'Node2Vec' differ from simple degree-based featurization?

- [x] It learns structural embeddings by processing random walk sequences through the graph.
- [ ] It relies exclusively on ground-truth labels to generate node representations.
- [ ] It is a manual feature engineering technique that requires domain expertise.
- [ ] It provides a single integer value representing the node's rank in the network.

**Hint:** Recall the parallel mentioned between representation learning and modern language models.

## Question 12
In the provided example of molecule classification, what serve as the 'nodes' and 'edges' in the graph representation?

- [x] Atoms serve as nodes and chemical bonds serve as edges.
- [ ] Molecules serve as nodes and reactions serve as edges.
- [ ] Protons serve as nodes and electrons serve as edges.
- [ ] Solubility serves as nodes and toxicity serves as edges.

**Hint:** Think about the basic components of a chemical structure diagram.
