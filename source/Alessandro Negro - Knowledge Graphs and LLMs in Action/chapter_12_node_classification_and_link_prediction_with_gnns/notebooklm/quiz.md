# GNN Quiz

## Question 1
Which general framework is utilized in Chapter 12 to handle both node classification and link prediction tasks?

- [x] An encoder-decoder architecture
- [ ] A recurrent feedback loop
- [ ] A standard multi-layer perceptron
- [ ] A generative adversarial network

**Hint:** Think about the common two-phase design used to translate graph data into task-specific predictions.

## Question 2
In the context of Anti-Money Laundering (AML) applications, what do the nodes and edges represent in the financial transaction graph?

- [x] Nodes are accounts/entities; edges are transaction flows
- [ ] Nodes are time intervals; edges are payment types
- [ ] Nodes represent illicit transactions only; edges represent bank locations
- [ ] Nodes are currency types; edges are conversion rates

**Hint:** Consider the fundamental components of a graph in the context of people or businesses sending money to one another.

## Question 3
According to the analysis of the Elliptic dataset, what is the significance of the 'Unknown' class for nodes?

- [x] They represent $77.15\%$ of the dataset and must be masked during supervised training
- [ ] They are used as the primary training set to ensure model generalization
- [ ] They represent a third classification category used for final model evaluation
- [ ] They are automatically assigned to the illicit class to be conservative

**Hint:** Reflect on how models handle data that lacks specific ground-truth labels during the supervision phase.

## Question 4
When comparing GCN, GAT, and GraphSAGE (SAGE) for node classification in AML, which model was noted for having the highest precision?

- [x] GraphSAGE (SAGE)
- [ ] Graph Convolutional Network (GCN)
- [ ] Graph Attention Network (GAT)
- [ ] Log Softmax Network

**Hint:** One specific model is highlighted for its ability to avoid misclassifying licit transactions as illicit.

## Question 5
What is the primary architectural difference when moving from node classification to link prediction in the MovieLens scenario?

- [x] The use of HeteroData to represent different types of nodes and edges
- [ ] Replacing the GNN encoder with a simple lookup table
- [ ] Removing the need for a decoder phase entirely
- [ ] Transitioning from semi-structured data to a homogeneous graph

**Hint:** Consider how the data structure changes when you have to distinguish between a 'user' entity and a 'movie' entity.

## Question 6
In the link prediction decoder, which operation is performed to estimate the likelihood of a relationship between a user and a movie?

- [x] A dot product between user and movie embeddings
- [ ] A cross-entropy calculation on the adjacency matrix
- [ ] A concatenation followed by a max-pooling operation
- [ ] A linear regression on the genre feature vector

**Hint:** Think of a common vector operation used to determine how 'similar' or 'compatible' two vectors are.

## Question 7
During the data preparation for link prediction, why is the `disjoint_train_ratio` parameter used in the `RandomLinkSplit` function?

- [x] To separate edges used for message passing from those used as training labels
- [ ] To increase the number of negative examples in the validation set
- [ ] To convert directed edges into undirected ones for the GNN
- [ ] To balance the number of user nodes and movie nodes

**Hint:** This relates to preventing the model from having access to the 'answers' while it is still performing message passing.

## Question 8
What is the role of reverse edges, such as `rev_rates`, in a heterogeneous GNN for link prediction?

- [x] They allow bidirectional message passing so movies can also aggregate user information
- [ ] They serve as negative labels for the training phase
- [ ] They are used to reduce the memory footprint of the graph
- [ ] They prevent the model from overfitting on movie genres

**Hint:** Think about how information moves between different node types in a graph during the aggregation step.

## Question 9
How are the initial embeddings for 'user' nodes generated in the MovieLens system since they lack intrinsic features?

- [x] They are initialized as learnable parameters and updated by the model
- [ ] They are derived from a random hash of the user's ID
- [ ] They are copied from the genre features of the movies they watched
- [ ] They are set to a constant vector of ones to maintain neutrality

**Hint:** Consider how a model handles missing input features for an entity it needs to represent in a latent space.

## Question 10
Which specific loss function is paired with the dot-product decoder for the link prediction task?

- [x] Binary cross-entropy with logits
- [ ] Mean Squared Error (MSE)
- [ ] Categorical cross-entropy
- [ ] Hinge loss

**Hint:** Look for a loss function commonly used when the outcome is a binary 'yes/no' probability.
