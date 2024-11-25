import torch
print(torch.cuda.is_available())
from tensorboardX import SummaryWriter
from torch_geometric.utils import dense_to_sparse
from Pytorch_GCQ_DuelingDistributional import torch_GCQ_DuelingDistributional

# 初始化图强化学习网络
GCQ = torch_GCQ_DuelingDistributional(40, 8, 0, 0, 3, 51, -10, 10)

# 定义特征长度
N = 40
F = 8
A = 3

# 节点特征
X_in = torch.ones(N, F)
X_in = X_in.float()

# 邻接矩阵
# 定义稠密邻接矩阵
A_in_Dense = torch.ones(N, N)
A_in_Dense = A_in_Dense.long()
# 计算稀疏邻接矩阵COO
A_in_Sparse, _ = dense_to_sparse(A_in_Dense)

# print(A_in_Dense.type())
# print(A_in_Sparse.type())

# 强化学习智能体索引
RL_indice = torch.ones(N)
RL_indice = RL_indice.long()

# print(X_in)
# print(A_in_Dense)
# print(A_in_Sparse)
# print(RL_indice)

# 网络测试
output = GCQ(X_in, A_in_Dense, RL_indice)

# 网络结构可视化
# with SummaryWriter(comment="GCQ") as w:
#     w.add_graph(GCQ, [X_in, A_in_Sparse, RL_indice])

# 打印测试结果
print("output:", output)
