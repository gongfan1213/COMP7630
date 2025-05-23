### 基于模型的协同过滤（Model - based CF）
1. **与基于内存的方法对比**
    - **基于内存的方法（In memory - based methods）**：基于用户或物品之间的相似度来预测缺失的评分。例如，找到与目标用户相似的其他用户，根据这些相似用户对物品的评分来预测目标用户对该物品的评分 。
    - **基于模型的协同过滤（In model - based collaborative filtering）**：假设存在一个潜在的模型来控制用户的评分行为。通过学习这个模型，并使用它来预测缺失的评分。这里重点介绍基于奇异值分解（SVD）的成熟技术。
2. **基于SVD的模型实现**
    - **奇异值分解应用**：对评分矩阵应用奇异值分解（SVD），并取用户 - 物品矩阵$X$的最佳秩 - $k$近似$X_k$ 。原始评分矩阵$X$和近似矩阵$X_k$具有相同的$m×n$形状，但$X_k$的秩为$k$，并且比$X$更稠密（零元素少得多）。其数学表达式为$X = U\sum V^T$（原始矩阵SVD分解）和$X_k = U_k\sum_k V_k^T$（秩 - $k$近似矩阵） 。其中，$U_k$（用户矩阵）的任意一行在“去噪”的$k$维空间中编码一个用户；$V_k$（物品矩阵）的任意一列在“去噪”的$k$维空间中编码一个物品 。
    - **处理缺失值**：在应用截断奇异值分解（Truncated SVD）之前，对于缺失的条目，可以将其设置为零，然后截断SVD会隐式地为它们学习一个有语义意义的值；或者，也可以调整截断SVD的内部优化过程，使其仅考虑非缺失值 。截断SVD实际上是最小化$\|X - X_k\|_F$（$F$范数），其中$X_k$的秩为$k$ 。
3. **示例**
    - 给出一个用户 - 物品矩阵示例，包含用户（John、Joe、Jill、Jorge）对物品（Lion King、Aladdin、Mulan ）的评分。考虑秩为2的近似（$k = 2$），对分解得到的三个矩阵$U$、$\sum$、$V^T$进行截断，得到$U_k$、$\sum_k$、$V_k^T$ 。通过这个示例可以直观看到SVD分解和截断操作后的矩阵形式。
4. **潜在空间（Latent Space）**
    - **投影与特征**：将用户和物品的偏好投影到一个低维空间，这个低维空间由潜在/隐藏特征形成，这些特征捕捉了偏好的相关方面。虽然这些特征难以解释，但新的矩阵比原来更稠密且具有语义意义 。需要注意的是，SVD和矩阵分解并不是基于模型的协同过滤的唯一方法，例如还有非负矩阵分解（Non - Negative Matrix Factorization） 。
5. **潜在空间的其他用途**
    - **语义表示**：$U_k$的行（可能乘以奇异值）是用户的语义表示；$V_k$的行（即$V_k^T$的列，可能乘以奇异值）是物品的语义表示 。
    - **相似度计算与聚类**：可以在这些语义表示上计算余弦相似度，以衡量用户或物品对之间的语义相似度；还可以在这些表示上执行聚类算法，将相似的用户或物品聚类在一起，这有助于提高推荐系统的可扩展性和多样性，例如可以考虑用户聚类而不是单个用户来提供推荐 。此外，也可以在这些语义表示上训练和执行分类算法 。 
