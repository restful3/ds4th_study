# Graph Quiz

## Question 1
According to the source material, what is the primary trade-off when moving from manual feature engineering to fully automated features?

- [ ] Efficiency decreases while the accuracy of the vectors significantly increases.
- [x] Features become easier to generate but tend to lose their interpretability.
- [ ] Domain expertise becomes more critical as the process becomes more automated.
- [ ] The numerical vectors produced by automated methods cannot be processed by traditional algorithms.

**Hint:** Consider the relationship between how a feature is created and our ability to explain why it works.

## Question 2
In a fraud detection network, a node has a total degree of 6. If it is connected to 2 fraudulent nodes and 4 legitimate nodes, how are these specific metrics defined in the manual featurization process?

- [x] The 'fraud degree' is 2, the 'legit degree' is 4, and the 'total degree' is 6.
- [ ] The 'fraud degree' is $2/6$, representing the ratio of fraudulent connections.
- [ ] The 'fraud degree' is 6, as it represents the total potential influence of the node.
- [ ] The 'legit degree' is 2, representing the distance to the nearest legitimate node.

**Hint:** These metrics are based on the direct count of different types of neighbors.

## Question 3
When calculating the density of a node's egonet, what is the denominator used in the formula if the egonet contains $N$ nodes?

- [ ] $N^{2}$
- [x] $N(N - 1) / 2$
- [ ] $N - 1$
- [ ] $M / N$

**Hint:** Think about the total number of unique pairs that can be formed among $N$ nodes.

## Question 4
Which centrality metric measures the frequency with which a node appears on the shortest paths between all other pairs of nodes in the graph?

- [ ] Closeness Centrality
- [x] Betweenness Centrality
- [ ] PageRank
- [ ] Degree Centrality

**Hint:** This metric is often associated with controlling information flow.

## Question 5
How does 'fraud-weighted PageRank' differ from standard PageRank in a manual feature engineering context?

- [x] It uses a personalization dictionary to give higher weight to connections from known fraudulent nodes.
- [ ] It ignores all legitimate nodes and only calculates paths between fraudulent nodes.
- [ ] It is calculated by multiplying the base PageRank by the total number of fraudulent neighbors.
- [ ] It is a manual adjustment made by domain experts after the standard algorithm finishes.

**Hint:** Focus on how the algorithm is 'primed' to care more about certain types of starting points.

## Question 6
In the context of relationship featurization, if a node vector $u = [2, 4]$ and $v = [4, 8]$, what would be the resulting vector using the L1 (Manhattan distance) operator?

- [ ] $[3, 6]$
- [x] $[2, 4]$
- [ ] $[6, 12]$
- [ ] $[8, 32]$

**Hint:** Think about finding the absolute difference between corresponding elements.

## Question 7
What is the primary function of the ReFeX (Recursive Feature Extraction) algorithm described in the text?

- [ ] To manually prune irrelevant nodes from a fraud network based on their color or type.
- [x] To convert nodes into vectors by aggregating topological features from increasingly distant neighborhoods.
- [ ] To replace LLMs in the process of generating Cypher queries for metapaths.
- [ ] To calculate the geodesic distance between a compound and a disease in Hetionet.

**Hint:** Consider how information is 'passed' between nodes over multiple iterations.

## Question 8
If ReFeX is in its first iteration using a SUM operator for node $A$, and node $A$ has neighbors with degrees of 3, 4, 2, 4, 3, and 1, what is the value of this recursive feature?

- [x] 17
- [ ] 2.83
- [ ] 6
- [ ] 69

**Hint:** Simply add the individual degree values of all adjacent nodes.

## Question 9
How can Large Language Models (LLMs) assist in the manual feature engineering process for graphs, specifically in tasks like drug repurposing?

- [ ] By automatically calculating PageRank for billions of nodes without using graph databases.
- [x] By translating high-level metapath descriptions into optimized Cypher queries and suggesting relevant patterns.
- [ ] By serving as the primary classifier model that replaces Random Forests and Logistic Regression.
- [ ] By visually rendering the graph schema to identify errors in relationship directions.

**Hint:** Think about the bridge between domain-specific concepts (like metapaths) and technical implementation (like Cypher).

## Question 10
In the study of Hetionet, researchers created representations for node pairs using metapaths of what lengths?

- [ ] Length 1 only
- [x] Lengths 2 to 4
- [ ] Exactly length 10
- [ ] Only even-numbered lengths

**Hint:** The range covers paths slightly beyond direct neighbors but avoids extremely long traversals.
