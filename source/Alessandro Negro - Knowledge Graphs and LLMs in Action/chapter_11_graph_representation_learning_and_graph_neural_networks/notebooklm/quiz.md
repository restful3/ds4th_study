# Graph Quiz

## Question 1
In the context of graph representation learning (GRL), what is the primary purpose of transforming discrete graph structures into continuous vector embeddings?

- [x] To convert complex graph relationships into a format easily consumable by standard machine learning algorithms.
- [ ] To reduce the computational storage of the graph by eliminating node attributes.
- [ ] To ensure that every node in the graph has exactly the same number of connections.
- [ ] To replace automated feature engineering with manual heuristic-based feature design.

**Hint:** Think about how standard ML models require numerical inputs compared to the raw structure of an adjacency matrix.

## Question 2
Why might hyperbolic space be preferred over Euclidean space for embedding certain real-world graphs like biological taxonomies or corporate hierarchies?

- [x] Hyperbolic space allows the available space to grow exponentially as you move outward from the center.
- [ ] Euclidean space is too complex for calculating standard distances between nodes in a tree.
- [ ] Hyperbolic embeddings eliminate the need for an encoder-decoder framework.
- [ ] Most machine learning models naturally assume a non-Euclidean geometry by default.

**Hint:** Consider the relationship between the branching nature of a tree and the volume of space available as you increase the distance from the root.

## Question 3
Which type of embedding would be most useful for identifying nodes that play similar functional roles (e.g., 'bridge-builders' or 'organizers') regardless of their physical distance in the network?

- [x] Structural embeddings
- [ ] Positional embeddings
- [ ] Discrete embeddings
- [ ] Shallow embeddings

**Hint:** Recall the distinction between local connectivity patterns and global network coordinates.

## Question 4
A recommendation system needs to generate embeddings for new users who join the platform daily. Which learning strategy is required for this dynamic scenario?

- [x] Inductive learning
- [ ] Transductive learning
- [ ] Manual feature engineering
- [ ] Matrix factorization

**Hint:** Focus on the difference between memorizing a fixed set of nodes and learning a general rule for any node.

## Question 5
What is a primary limitation of shallow embedding methods like those that use a simple lookup table?

- [x] They cannot incorporate node features or attributes during the embedding process.
- [ ] They are too computationally complex for small, static graphs.
- [ ] They require the use of complex numbers like those in the ComplEx model.
- [ ] They prioritize global structure over local connectivity by default.

**Hint:** Consider how a dictionary-style lookup table handles information outside of the unique identifier of the entry.

## Question 6
In the loss function for Knowledge Graph (KG) embeddings, what is the role of the $\gamma$ (gamma) parameter?

- [x] It acts as a balancing parameter that controls the importance of negative samples.
- [ ] It defines the learning rate for the cross-entropy optimization.
- [ ] It determines the dimensionality of the vector space for the head and tail nodes.
- [ ] It normalizes the sigmoid function outputs to ensure they sum to one.

**Hint:** Look at the negative part of the loss equation and think about the imbalance between existing and non-existing edges.

## Question 7
How does a Graph Neural Network (GNN) capture information from a node's broader neighborhood beyond its immediate friends?

- [x] By performing multiple iterations (rounds) of message passing.
- [ ] By increasing the dimensionality of the node's initial feature vector.
- [ ] By using a lookup table to store the entire graph adjacency matrix.
- [ ] By applying the softmax function to the entire node set simultaneously.

**Hint:** Think of the process as a conversation where information spreads from person to person over time.

## Question 8
What is the primary advantage of 'symmetric normalization' in Graph Convolutional Networks (GCNs) compared to simple mean aggregation?

- [x] It accounts for the degrees of both the source and the target nodes, preventing high-degree nodes from dominating the signal.
- [ ] It ensures that the sum of all neighbor features always equals exactly one.
- [ ] It allows the model to ignore the node's own features during the update step.
- [ ] It eliminates the need for non-linear activation functions like ReLU.

**Hint:** Consider how a very popular node (high degree) might influence a smaller node if the normalization only looked at the target.

## Question 9
In a Graph Attention Network (GAT), what determines the weight $\alpha_{u,v}$ assigned to a neighboring node $v$ during aggregation?

- [x] A learnable mechanism that calculates relevance based on the features of both nodes $u$ and $v$.
- [ ] The shortest path distance between node $u$ and node $v$ in the global graph.
- [ ] The ratio of node $v$'s degree to the total number of edges in the graph.
- [ ] A fixed constant assigned based on the relationship type in a knowledge graph.

**Hint:** Think about how 'relevance' might change depending on the specific characteristics of the nodes involved in the interaction.

## Question 10
What problem do 'skip connections' (residual connections) primarily solve in deep GNN architectures?

- [x] Over-smoothing, where node representations become indistinguishable after many layers.
- [ ] The inability of GNNs to process graphs with more than 1,000 nodes.
- [ ] The requirement for manual feature engineering in real-world knowledge graphs.
- [ ] The lack of non-Euclidean geometric properties in hierarchical data.

**Hint:** Consider what happens to information as it passes through many successive averaging operations.

## Question 11
Which architecture combines representations from all previous layers adaptively to capture multi-scale structural information?

- [x] Jumping Knowledge Networks
- [ ] GraphSAGE
- [ ] TransE
- [ ] Vanilla RNN

**Hint:** This method is named after the idea of 'skipping' or 'leaping' through the hierarchy of network layers.

## Question 12
How can Large Language Models (LLMs) be used as 'encoders' in a GNN-based system?

- [x] By processing textual node attributes to create rich feature vectors that the GNN then uses for structural processing.
- [ ] By replacing the GNN's adjacency matrix with a sequence of word embeddings.
- [ ] By acting as the final classifier that predicts links based only on the graph topology.
- [ ] By aligning the outputs of two separate models using contrastive learning.

**Hint:** Think about where the textual information of a paper abstract or user profile enters the pipeline.
