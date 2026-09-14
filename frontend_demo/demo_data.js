window.DEMO_DATA = {
  "project": {
    "papers": 12,
    "retrieval_chunks": 446,
    "graph_nodes": 1204,
    "graph_edges": 1761,
    "benchmark": "Core80 → Core69",
    "generation_set": "Core32",
    "global_scores": {
      "Structured Vector": 0.8734,
      "Vanilla Hybrid": 0.7969,
      "Hybrid + Rerank": 0.825
    },
    "global_claim_support_rate": 0.799,
    "verified_claims": 199,
    "supported_claims": 159,
    "verification_success": "27/32"
  },
  "case": {
    "qid": "C01",
    "qtype": "comparison",
    "question": "GCN 和 GraphSAGE 在归纳能力上有什么区别？",
    "scores": {
      "vector": 1.0,
      "hybrid": 0.5,
      "rerank": 1.0
    },
    "vanilla_evidence": [
      {
        "evidence_id": "V1",
        "source_type": "vector",
        "paper_id": "cluster_gcn_2019",
        "title": "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks",
        "section": "Experiments",
        "retrieval_score": 0.6806147694587708,
        "rerank_score": 0.0,
        "content": "GraphSAGE is slower than VRGCN and our method, the curves for\nGraphSAGE only appear for PPI and Reddit datasets. We can see that our method is the fastest for both PPI and Reddit datasets for\nGCNs with different numbers of layers. For Amazon data, since nodes’ features are not available, an identity matrix is used as the feature matrix X . Under this setting, the shape of parameter matrix W (0) becomes 334863x128. Therefore,\nthe computation is dominated by sparse matrix operations such as\nA W(0). Our method is still faster than VRGCN for 3-layer case, but slower for 2-layer and 4-layer ones. The reason may come from the speed of sparse matrix operations from different frameworks. VRGCN is implemented in TensorFlow, while Cluster-GCN is implemented in PyTorch whose sparse tensor support are still in its very early stage. In Table 6, we show the time for TensorFlow and\nPyTorch to do forward/backward operations on Amazon data, and a simple two-layer network are used for benchmarking both frameworks. We can clearly see that TensorFlow is faster than PyTorch. The difference is more significant when the number of hidden units increases. This may explain why Cluster-GCN has longer training time in Amazon dataset. Memory usage comparison: For training large-scale GCNs,\nbesides training time, memory usage needed for training is of-ten more important and will directly restrict the scalability. The memory usage includes the memory needed for training the GCN for many epochs. As discussed in Section 3, to speedup training,\nVRGCN needs to save historical embeddings during training, so it needs much more memory for training than Cluster-GCN. Graph-\nSAGE also has higher memory requirement than Cluster-GCN due to the exponential neighborhood growing problem.",
        "paper_ids": [
          "cluster_gcn_2019"
        ]
      },
      {
        "evidence_id": "V2",
        "source_type": "vector",
        "paper_id": "diffpool_2018",
        "title": "Hierarchical Graph Representation Learning with Differentiable Pooling",
        "section": "Experiments",
        "retrieval_score": 0.659974217414856,
        "rerank_score": 0.0,
        "content": "Other GNN variants such as those proposed in [22]\nare omitted as empirically GraphSAGE obtained higher performance in the task.\n• STRUCTURE 2V EC (S2V ) [7] is a state-of-the-art graph representation learning algorithm that combines a latent variable model with GNNs. It uses global mean pooling.\n• Edge-conditioned filters in CNN for graphs (ECC ) [36] incorporates edge information into the\nGCN model and performs pooling using a graph coarsening algorithm. Table 1: Classification accuracies in percent. The far-right column gives the relative increase in accuracy compared to the baseline GRAPH SAGE approach. Method Data Set\nENZYMES D&D R EDDIT-MULTI-12K COLLAB PROTEINS Gain\nKernel\nGRAPHLET 41.03 74.85 21.73 64.66 72.91\nSHORTEST-PATH 42.32 78.86 36.93 59.10 76.43\n1-WL 53.43 74.02 39.03 78.61 73.76\nWL-OA 60.13 79.04 44.38 80.74 75.26\nPATCHYSAN – 76.27 41.32 72.60 75.00 4.17\nGNN\nGRAPHSAGE 54.25 75.42 42.24 68.25 70.48 –\nECC 53.50 74.10 41.73 67.79 72.65 0.11\nSET2SET 60.15 78.12 43.49 71.75 74.29 3.32\nSORTPOOL 57.12 79.37 41.82 73.76 75.54 3.39\nDIFFPOOL-DET 58.33 75.47 46.18 82.13 75.62 5.42\nDIFFPOOL-NOLP 61.95 79.98 46.65 75.58 76.22 5.95\nDIFFPOOL 62.53 80.64 47.08 75.48 76.25 6.27\n• PATCHY SAN [30] defines a receptive field (neighborhood) for each node, and using a canonical node ordering, applies convolutions on linear sequences of node embeddings.\n• SET2SET replaces the global mean-pooling in the traditional GNN architectures by the aggregation used in SET2SET [39]. Set2Set aggregation has been shown to perform better than mean pooling in previous work [15]. We use GRAPH SAGE as the base GNN model.\n• SORTPOOL [41] applies a GNN architecture and then performs a single layer of soft pooling followed by 1D convolution on sorted node embeddings.",
        "paper_ids": [
          "diffpool_2018"
        ]
      },
      {
        "evidence_id": "V3",
        "source_type": "vector",
        "paper_id": "graphsage_2017",
        "title": "Inductive Representation Learning on Large Graphs",
        "section": "Experiments",
        "retrieval_score": 0.6435948014259338,
        "rerank_score": 0.0,
        "content": "We test the performance of GraphSAGE on three benchmark tasks: (i) classifying academic papers into different subjects using the Web of Science citation dataset, (ii) classifying Reddit posts as belonging to different communities, and (iii) classifying protein functions across various biological protein-protein interaction (PPI) graphs. Sections 4.1 and 4.2 summarize the datasets, and the supplementary material contains additional information. In all these experiments, we perform predictions on nodes that are not seen during training, and, in the case of the PPI dataset, we test on entirely unseen graphs. Experimental set-up. To contextualize the empirical results on our inductive benchmarks, we compare against four baselines: a random classifer, a logistic regression feature-based classifier\n(that ignores graph structure), the DeepWalk algorithm [28] as a representative factorization-based approach, and a concatenation of the raw features and DeepWalk embeddings. We also compare four variants of GraphSAGE that use the different aggregator functions (Section 3.3). Since, the “convolutional” variant of GraphSAGE is an extended, inductive version of Kipf et al’s semi-supervised\nGCN [17], we term this variant GraphSAGE-GCN. We test unsupervised variants of GraphSAGE trained according to the loss in Equation (1), as well as supervised variants that are trained directly on classification cross-entropy loss. For all the GraphSAGE variants we used rectified linear units as the non-linearity and setK = 2 with neighborhood sample sizesS1 = 25 andS2 = 10 (see Section\n4.4 for sensitivity analyses). For the Reddit and citation datasets, we use “online” training for DeepWalk as described in Perozzi et al.",
        "paper_ids": [
          "graphsage_2017"
        ]
      },
      {
        "evidence_id": "G1",
        "source_type": "triple",
        "paper_id": "graphsage_2017",
        "title": "Inductive Representation Learning on Large Graphs",
        "section": "",
        "retrieval_score": 2.0,
        "rerank_score": 0.0,
        "content": "GraphSAGE --EXTENDS--> GCN",
        "paper_ids": [
          "graphsage_2017"
        ]
      },
      {
        "evidence_id": "G2",
        "source_type": "triple",
        "paper_id": "diffpool_2018",
        "title": "Hierarchical Graph Representation Learning with Differentiable Pooling",
        "section": "",
        "retrieval_score": 2.0,
        "rerank_score": 0.0,
        "content": "GraphSAGE --RELATED_TO--> GCN",
        "paper_ids": [
          "diffpool_2018"
        ]
      },
      {
        "evidence_id": "G3",
        "source_type": "triple",
        "paper_id": "graphsage_2017",
        "title": "Inductive Representation Learning on Large Graphs",
        "section": "",
        "retrieval_score": 2.0,
        "rerank_score": 0.0,
        "content": "GraphSAGE --RELATED_TO--> GCN",
        "paper_ids": [
          "graphsage_2017"
        ]
      }
    ],
    "rerank_evidence": [
      {
        "evidence_id": "V1",
        "source_type": "vector",
        "paper_id": "cluster_gcn_2019",
        "title": "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks",
        "section": "Experiments",
        "retrieval_score": 0.6806147694587708,
        "rerank_score": 0.7101536804437638,
        "content": "GraphSAGE is slower than VRGCN and our method, the curves for\nGraphSAGE only appear for PPI and Reddit datasets. We can see that our method is the fastest for both PPI and Reddit datasets for\nGCNs with different numbers of layers. For Amazon data, since nodes’ features are not available, an identity matrix is used as the feature matrix X . Under this setting, the shape of parameter matrix W (0) becomes 334863x128. Therefore,\nthe computation is dominated by sparse matrix operations such as\nA W(0). Our method is still faster than VRGCN for 3-layer case, but slower for 2-layer and 4-layer ones. The reason may come from the speed of sparse matrix operations from different frameworks. VRGCN is implemented in TensorFlow, while Cluster-GCN is implemented in PyTorch whose sparse tensor support are still in its very early stage. In Table 6, we show the time for TensorFlow and\nPyTorch to do forward/backward operations on Amazon data, and a simple two-layer network are used for benchmarking both frameworks. We can clearly see that TensorFlow is faster than PyTorch. The difference is more significant when the number of hidden units increases. This may explain why Cluster-GCN has longer training time in Amazon dataset. Memory usage comparison: For training large-scale GCNs,\nbesides training time, memory usage needed for training is of-ten more important and will directly restrict the scalability. The memory usage includes the memory needed for training the GCN for many epochs. As discussed in Section 3, to speedup training,\nVRGCN needs to save historical embeddings during training, so it needs much more memory for training than Cluster-GCN. Graph-\nSAGE also has higher memory requirement than Cluster-GCN due to the exponential neighborhood growing problem.",
        "paper_ids": [
          "cluster_gcn_2019"
        ]
      },
      {
        "evidence_id": "V2",
        "source_type": "vector",
        "paper_id": "diffpool_2018",
        "title": "Hierarchical Graph Representation Learning with Differentiable Pooling",
        "section": "Experiments",
        "retrieval_score": 0.659974217414856,
        "rerank_score": 0.704993578195572,
        "content": "Other GNN variants such as those proposed in [22]\nare omitted as empirically GraphSAGE obtained higher performance in the task.\n• STRUCTURE 2V EC (S2V ) [7] is a state-of-the-art graph representation learning algorithm that combines a latent variable model with GNNs. It uses global mean pooling.\n• Edge-conditioned filters in CNN for graphs (ECC ) [36] incorporates edge information into the\nGCN model and performs pooling using a graph coarsening algorithm. Table 1: Classification accuracies in percent. The far-right column gives the relative increase in accuracy compared to the baseline GRAPH SAGE approach. Method Data Set\nENZYMES D&D R EDDIT-MULTI-12K COLLAB PROTEINS Gain\nKernel\nGRAPHLET 41.03 74.85 21.73 64.66 72.91\nSHORTEST-PATH 42.32 78.86 36.93 59.10 76.43\n1-WL 53.43 74.02 39.03 78.61 73.76\nWL-OA 60.13 79.04 44.38 80.74 75.26\nPATCHYSAN – 76.27 41.32 72.60 75.00 4.17\nGNN\nGRAPHSAGE 54.25 75.42 42.24 68.25 70.48 –\nECC 53.50 74.10 41.73 67.79 72.65 0.11\nSET2SET 60.15 78.12 43.49 71.75 74.29 3.32\nSORTPOOL 57.12 79.37 41.82 73.76 75.54 3.39\nDIFFPOOL-DET 58.33 75.47 46.18 82.13 75.62 5.42\nDIFFPOOL-NOLP 61.95 79.98 46.65 75.58 76.22 5.95\nDIFFPOOL 62.53 80.64 47.08 75.48 76.25 6.27\n• PATCHY SAN [30] defines a receptive field (neighborhood) for each node, and using a canonical node ordering, applies convolutions on linear sequences of node embeddings.\n• SET2SET replaces the global mean-pooling in the traditional GNN architectures by the aggregation used in SET2SET [39]. Set2Set aggregation has been shown to perform better than mean pooling in previous work [15]. We use GRAPH SAGE as the base GNN model.\n• SORTPOOL [41] applies a GNN architecture and then performs a single layer of soft pooling followed by 1D convolution on sorted node embeddings.",
        "paper_ids": [
          "diffpool_2018"
        ]
      },
      {
        "evidence_id": "V3",
        "source_type": "vector",
        "paper_id": "graphsage_2017",
        "title": "Inductive Representation Learning on Large Graphs",
        "section": "Experiments",
        "retrieval_score": 0.6435948014259338,
        "rerank_score": 0.7008987122774124,
        "content": "We test the performance of GraphSAGE on three benchmark tasks: (i) classifying academic papers into different subjects using the Web of Science citation dataset, (ii) classifying Reddit posts as belonging to different communities, and (iii) classifying protein functions across various biological protein-protein interaction (PPI) graphs. Sections 4.1 and 4.2 summarize the datasets, and the supplementary material contains additional information. In all these experiments, we perform predictions on nodes that are not seen during training, and, in the case of the PPI dataset, we test on entirely unseen graphs. Experimental set-up. To contextualize the empirical results on our inductive benchmarks, we compare against four baselines: a random classifer, a logistic regression feature-based classifier\n(that ignores graph structure), the DeepWalk algorithm [28] as a representative factorization-based approach, and a concatenation of the raw features and DeepWalk embeddings. We also compare four variants of GraphSAGE that use the different aggregator functions (Section 3.3). Since, the “convolutional” variant of GraphSAGE is an extended, inductive version of Kipf et al’s semi-supervised\nGCN [17], we term this variant GraphSAGE-GCN. We test unsupervised variants of GraphSAGE trained according to the loss in Equation (1), as well as supervised variants that are trained directly on classification cross-entropy loss. For all the GraphSAGE variants we used rectified linear units as the non-linearity and setK = 2 with neighborhood sample sizesS1 = 25 andS2 = 10 (see Section\n4.4 for sensitivity analyses). For the Reddit and citation datasets, we use “online” training for DeepWalk as described in Perozzi et al.",
        "paper_ids": [
          "graphsage_2017"
        ]
      },
      {
        "evidence_id": "V4",
        "source_type": "vector",
        "paper_id": "gat_2018",
        "title": "Graph Attention Networks",
        "section": "Results",
        "retrieval_score": 0.6366592645645142,
        "rerank_score": 0.6991648161411286,
        "content": "GCN-64∗ corresponds to the best GCN result computing 64 hidden features (using ReLU or ELU). Transductive\nMethod Cora Citeseer Pubmed\nMLP 55.1% 46.5% 71.4%\nManiReg (Belkin et al., 2006) 59.5% 60.1% 70.7%\nSemiEmb (Weston et al., 2012) 59.0% 59.6% 71.7%\nLP (Zhu et al., 2003) 68.0% 45.3% 63.0%\nDeepWalk (Perozzi et al., 2014) 67.2% 43.2% 65.3%\nICA (Lu & Getoor, 2003) 75.1% 69.1% 73.9%\nPlanetoid (Yang et al., 2016) 75.7% 64.7% 77.2%\nChebyshev (Defferrard et al., 2016) 81.2% 69.8% 74.4%\nGCN (Kipf & Welling, 2017) 81.5% 70.3% 79.0%\nMoNet (Monti et al., 2016) 81.7 ± 0.5% — 78.8 ± 0.3%\nGCN-64∗ 81.4± 0.5% 70.9 ± 0.5% 79.0± 0.3%\nGAT (ours) 83.0± 0.7% 72.5± 0.7% 79.0± 0.3%\nTable 3: Summary of results in terms of micro-averaged F1 scores, for the PPI dataset. GraphSAGE∗\ncorresponds to the best GraphSAGE result we were able to obtain by just modifying its architecture. Const-GAT corresponds to a model with the same architecture as GAT, but with a constant attention mechanism (assigning same importance to each neighbor; GCN-like inductive operator). Inductive\nMethod PPI\nRandom 0.396\nMLP 0.422\nGraphSAGE-GCN (Hamilton et al., 2017) 0.500\nGraphSAGE-mean (Hamilton et al., 2017) 0.598\nGraphSAGE-LSTM (Hamilton et al., 2017) 0.612\nGraphSAGE-pool (Hamilton et al., 2017) 0.600\nGraphSAGE∗ 0.768\nConst-GAT (ours) 0.934 ± 0.006\nGAT (ours) 0.973± 0.002\nthe other techniques. Specifically, as our setup is supervised, we compare against the supervised\nGraphSAGE approaches.",
        "paper_ids": [
          "gat_2018"
        ]
      },
      {
        "evidence_id": "V5",
        "source_type": "vector",
        "paper_id": "graphsage_2017",
        "title": "Inductive Representation Learning on Large Graphs",
        "section": "Experiments",
        "retrieval_score": 0.6365313529968262,
        "rerank_score": 0.6991329336166382,
        "content": "to the GCN approach. For example, the unsupervised variant GraphSAGE-pool outperforms the concatenation of the DeepWalk embeddings and the raw features by 13.8% on the citation data and 29.1% on the Reddit data, while the supervised version provides a gain of 19.7% and 37.2%,\nrespectively. Interestingly, the LSTM based aggregator shows strong performance, despite the fact that it is designed for sequential data and not unordered sets. Lastly, we see that the performance of unsupervised GraphSAGE is reasonably competitive with the fully supervised version, indicating that our framework can achieve strong performance without task-specific fine-tuning.\n4.2 Generalizing across graphs: Protein-protein interactions\nWe now consider the task of generalizing across graphs, which requires learning about node roles rather than community structure. We classify protein roles—in terms of their cellular functions from gene ontology—in various protein-protein interaction (PPI) graphs, with each graph corresponding to a different human tissue [41]. We use positional gene sets, motif gene sets and immunological signatures as features and gene ontology sets as labels (121 in total), collected from the Molecular\nSignatures Database [34]. The average graph contains 2373 nodes, with an average degree of 28.8.\nWe train all algorithms on 20 graphs and then average prediction F1 scores on two test graphs (with two other graphs used for validation).\nThe final two columns of Table 1 summarize the accuracies of the various approaches on this data. Again we see that GraphSAGE significantly outperforms the baseline approaches, with the\nLSTM- and pooling-based aggregators providing substantial gains over the mean- and GCN-based aggregators.6",
        "paper_ids": [
          "graphsage_2017"
        ]
      },
      {
        "evidence_id": "V6",
        "source_type": "vector",
        "paper_id": "gin_2019",
        "title": "How Powerful are Graph Neural Networks?",
        "section": "Method",
        "retrieval_score": 0.6242356300354004,
        "rerank_score": 0.680058931350708,
        "content": "Between the two graphs, nodes v and v′ get the same embedding even though their corresponding graph structures differ. Figure 2 gives reasoning about how different aggregators “compress” different multisets and thus fail to distinguish them. Networks (Xu et al., 2018), where we replace Eq. 2.4 with graph representations concatenated across all iterations/layers of GIN:\nhG = CONCAT\n(\nREADOUT\n( {\nh(k)\nv |v∈ G\n} )⏐⏐ k = 0, 1, . . . , K\n)\n. (4.2)\nBy Theorem 3 and Corollary 6, if GIN replaces READOUT in Eq. 4.2 with summing all node features from the same iterations (we do not need an extra MLP before summation for the same reason as in Eq. 4.1), it provably generalizes the WL test and the WL subtree kernel. 5 L ESS POWERFUL BUT STILL INTERESTING GNN S\nNext, we study GNNs that do not satisfy the conditions in Theorem 3, including GCN (Kipf &\nWelling, 2017) and GraphSAGE (Hamilton et al., 2017a). We conduct ablation studies on two aspects of the aggregator in Eq. 4.1: (1) 1-layer perceptrons instead of MLPs and (2) mean or max-pooling instead of the sum. We will see that these GNN variants get confused by surprisingly simple graphs and are less powerful than the WL test. Nonetheless, models with mean aggregators like GCN perform well for node classification tasks. To better understand this, we precisely characterize what different GNN variants can and cannot capture about a graph and discuss the implications for learning with graphs. 5.1 1- LAYER PERCEPTRONS ARE NOT SUFFICIENT\nThe function f in Lemma 5 helps map distinct multisets to unique embeddings. It can be parameterized by an MLP by the universal approximation theorem (Hornik, 1991).",
        "paper_ids": [
          "gin_2019"
        ]
      }
    ],
    "answer": "根据给定证据，GCN 与 GraphSAGE 在归纳能力上的区别主要是：GraphSAGE 被用于归纳式设置，而 GCN 在对比中被列为直推式方法。\n\n- GCN 在 GAT 的对比表中被归入 **Transductive** 方法 [gat_2018]。\n- GraphSAGE 的卷积变体 GraphSAGE-GCN 被明确定义为 Kipf 等人半监督 GCN 的“扩展的、归纳的版本”（extended, inductive version） [graphsage_2017]。\n- GraphSAGE 实验中对训练时未见的节点进行预测；在 PPI 数据上还测试完全未见的图 [graphsage_2017]。\n- GraphSAGE 的跨图泛化实验在 20 个图上训练，并在两个测试图上评估预测 F1 [graphsage_2017]。\n- 因此，在给定证据范围内：GCN 是直推式对比基线，GraphSAGE 是归纳式框架或其归纳扩展，支持对未见节点和未见图进行预测 [gat_2018][graphsage_2017]。\n\n但关于 GCN 与 GraphSAGE 在聚合机制层面为何导致归纳能力差异，所给证据没有详细说明；E6 只指出 GCN 和 GraphSAGE 弱于 WL test，并未专门比较二者归纳能力 [gin_2019]。",
    "answer_sha256": "489335f83fcc47f9d2c0bc7f9b206e511a561b19c377d4393393755d79c971ff",
    "answer_regenerated": false,
    "claims": [
      {
        "claim_id": "C1",
        "claim": "根据给定证据，GCN 与 GraphSAGE 在归纳能力上的区别主要是：GraphSAGE 被用于归纳式设置，而 GCN 在对比中被列为直推式方法",
        "supported": true,
        "evidence_ids": [
          "E3",
          "E4"
        ]
      },
      {
        "claim_id": "C2",
        "claim": "GCN 在 GAT 的对比表中被归入 **Transductive** 方法",
        "supported": true,
        "evidence_ids": [
          "E4"
        ]
      },
      {
        "claim_id": "C3",
        "claim": "GraphSAGE 的卷积变体 GraphSAGE-GCN 被明确定义为 Kipf 等人半监督 GCN 的“扩展的、归纳的版本”（extended, inductive version）",
        "supported": true,
        "evidence_ids": [
          "E3",
          "E4"
        ]
      },
      {
        "claim_id": "C4",
        "claim": "GraphSAGE 实验中对训练时未见的节点进行预测",
        "supported": true,
        "evidence_ids": [
          "E3"
        ]
      },
      {
        "claim_id": "C5",
        "claim": "在 PPI 数据上还测试完全未见的图",
        "supported": true,
        "evidence_ids": [
          "E3"
        ]
      },
      {
        "claim_id": "C6",
        "claim": "GraphSAGE 的跨图泛化实验在 20 个图上训练，并在两个测试图上评估预测 F1",
        "supported": false,
        "evidence_ids": []
      },
      {
        "claim_id": "C7",
        "claim": "因此，在给定证据范围内：GCN 是直推式对比基线，GraphSAGE 是归纳式框架或其归纳扩展，支持对未见节点和未见图进行预测",
        "supported": true,
        "evidence_ids": [
          "E3",
          "E4"
        ]
      },
      {
        "claim_id": "C8",
        "claim": "但关于 GCN 与 GraphSAGE 在聚合机制层面为何导致归纳能力差异，所给证据没有详细说明",
        "supported": false,
        "evidence_ids": []
      },
      {
        "claim_id": "C9",
        "claim": "E6 只指出 GCN 和 GraphSAGE 弱于 WL test，并未专门比较二者归纳能力",
        "supported": true,
        "evidence_ids": [
          "E6"
        ]
      }
    ],
    "claim_support_rate": 0.7777777777777778,
    "supported_claims": 7,
    "total_claims": 9
  }
};
