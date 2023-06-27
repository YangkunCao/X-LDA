# X-LDA: an Interpretable and Heterogeneous Graph Learning Framework for LncRNA-Disease Association Prediction
The code for paper "X-LDA: an Interpretable and Heterogeneous Graph Learning Framework for LncRNA-Disease Association Prediction". 

<img width="1310" alt="f1" src="https://github.com/YangkunCao/X-LDA/assets/127037183/13f2be1c-497f-41ea-8c23-75fba662f6e9">

### Key Points
- To capture domain knowledge of lncRNA nodes, we collect lncRNA sequences and construct lncRNA features for various length sequence, which are essential for cold start problem of lncRNAs. Moreover, we calculate lncRNA sequence similarities and disease semantic similarities to draw the relationships between lncRNA or disease nodes, respectively.
- We construct a weighted adjacency matrix of the heterogeneous graph rather than a boolean matrix to integrate information of heterogeneous nodes and reduce the noise caused by truncating a similarity threshold. Inspired by patches in compute vision, we define nine types of graph patches in the heterogeneous graph based on diverse biological premises and construct graph patch features for a lncRNA-disease node pair.
- We propose the interpretable and heterogeneous graph learning method. We design graph patch convolution to learn common semantic information in the same type of graph patches and multiply views of a graph patch. The context convolution is exploited to fuse different semantic of graph patches.
- Based on the graph patch features and the integrated gradients, our method can give the post-hoc explanations of LDA predictions and can be provided to biologists in a visual way.
- Experiment results and analysis indicate that X-LDA outperforms nine state-of-the-art methods of three categories, and demonstrate the capability of discovering potential associations. In addition, we explore how the trained model predicts a known and an unknown LDA in the dataset by their post-hoc explanations.


### Dependencies
X-LDA is built in Python 3.8.5 and Pytorch 1.8.0. Please use the following command to install the requirements:

`pip install -r requirements.txt`


### 2.Construction of the weighted adjacency matrix and graph patch features.
<img width="1791" alt="f2" src="https://github.com/YangkunCao/X-LDA/assets/127037183/f1ebbe5e-8525-4548-8fd2-a6f27a3de4d5">


### 3.Explainable Graphs.
<img width="1832" alt="f3" src="https://github.com/YangkunCao/X-LDA/assets/127037183/435efd1a-2bbd-468c-9734-50b8abbcf422">


