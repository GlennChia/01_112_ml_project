# 01_112_ml_project
Project for SUTD 01.112 Machine Learning Fall 2019

# 1. Instructions on running the code 

## 1.1 Part 2

## 1.2 Part 3

## 1.3 Part 4

## 1.4 Part 5



# 2. Explaining the code

## 2.1 Part 2

### 2.1.1 Part 2 Question 1 Emission probabilities

### 2.2.2 Part 2 Question 2: Smoothing

### 2.2.3 Part 2 Question 3: Simple Sentiment Analysis System



## 2.3 Part 4: Implement an algorithm to find the 7-th best output sequences. Clearly describe the steps of your algorithm in your report.

### 2.3.1 High level description

Our approach for finding the 7th best path involves finding and storing the paths to the best 7 scores as we traverse between the layers. Hence, for a node `i` in the `j` layer, it will store the best 7 scores from all nodes in the previous layer `j-1` and the paths from `START` to itself that traverses through the best 7 nodes in layer `j-1`. For example, this means that a layer with 10 nodes will have 10*7=70 best paths and scores.

We do this from the START node (The best path 7 paths at this point ) all the way to the end STOP node. At the STOP node, we have all the scores from the previous layer multiplied by the transition probabilities between nodes from the previous layer to the STOP node $a_{u,STOP}$. We sort all the scores and find the 7th best score. This score guarantees that the path from the START node to the STOP node is the 7th best path. Since we are keeping track of the path from the START node to every node in every layer, we can fetch the path corresponding to the 7th best path.  

### 2.3.2 Low level description

**<u>Notations</u>**

We modify the scores equation to be an array $\pi(j,u)[k]$ where `j` is the current layer (Note that START starts from layer 0), `u` is the current node and k is the index of the array. if k=0 we are finding the best path. If k=6, we are finding the 7th best path to that node (since index starts from 0).

We let PATH(j,u)[k] where `j` is the current layer and `u` is the current node in that layer. `k` represents the index in the array that contains the path to that node.

<u>**Step 1: Initializing the tree**</u>

To initialise the tree, we instantiate a (n, t, 7) 3D-array with a custom NODE datatype, where n = # of words in the sentence and t = # of tags. Each (n, t) represents a node in the traditional viterbi tree, and each node u is expanded to an array of size 7 to store the 7 best nodes (score, parent, tag) from START to node u. The node datatype in each position stores the score from START to u, as well as the parent node that produced this score, and its corresponding output tag. This is done so backtracking can be done efficiently to find the full path eventually. 

**Step 1a: Consider the start layer.**

We initialize the best score of the start layer to be 1.0

$\pi(0, START)[0] = 1.0$

All other scores will be set to 0.0 since there will only be one path from START node to the next layer and we don't need to track the other scores besides the best one for the transition from START node to layer 1.

 $\pi(0, START)[h] = 0.0$ where $1 \leq h \leq k-1$

The paths stored at this stage are PATH(0,START)[0] = START and the rest of the paths are PATH(0,START)[0] = NULL

**Step 1b: All other nodes in the tree besides the START node will be initialized to a starting score of 0.0**

<u>**Step 2: Consider the nodes between START and STOP**</u> 

For each node in layer `j+1` we find all the scores from the previous layer to this node by taking $\pi(j+1,u)[h]*a_{v,u}*b_{u}(x_{j+1})$ for every node `v` in the previous layer `j` and every score in node `v`'s score array. We then have all the scores.  We then find the best 7 scores and store the indexes of these 7 scores. The indexes give us useful information of which score in which node produced that particular score. In order words, this tells us the node's parent and corresponding tag. We store the 7 best nodes in $\pi(j+1,u)[h]$ for h=0 to h=6. 


PATH(j+1, u)[h] = PATH(j, v))[Index which produced one of the 7 best scores] + (v -> u) 

where `v` is the previous node in layer `j` with its respective index that produced one of the best 7 scores. 

<u>**Step 3: Consider the STOP node at layer n+1**</u>

We compute all the scores from the previous layer n to this node by doing $\pi(n,u)[h]*a_{u,STOP}$ for all nodes u in the previous layer and the 7 scores stored in each node. We now have all the scores. We now find the 7 best scores and store corresponding parent and tag information into the a Node() object. 

Assume node q's third score is the overall 7th best score, we have

PATH(n+1, STOP)[7] = PATH(n, q))[3] + (q -> STOP)

<u>**Step 4: Get 7th best path**</u>

At this stage, we have a completed (n, t, 7) ndarray of Node() objects. We use this array and the final array of 7 best scores from the last STOP layer to find the 7th best path. Take note that each Node() object contains a pointer to its parent and corresponding tag. 

Using the 7th best score from the last layer, we implement a simple backtrack to get the tags of each node's parent. We append the tag of last node's parent (already stored in the node from previous iteration of modified viterbi) to a path. We then find this node's parent and do to same by inserting its tag at the front of this path. This is done from layer n+1 to layer 1, and the 7th best path is generated. 


