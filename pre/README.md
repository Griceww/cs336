## 工具类
### Vocab
token 与 id 的映射
### TextDataset
给一个很长的 (?,) 的 Tensr，将其切割成很多块，会用 TextDataset[idx] 调用第 idx 块
块是定长的 `seq_len`，启示位置是 `idx * stride`

Dataset 负责"取单个样本"，DataLoader 负责"组装成 batch 并高效迭代"。
collate_fn 是将多个样本（由 __getitem__ 返回的）拼接成一个 batch。
#### torch.utils.data.DataLoader
构造参数：
- 一个 Dataset，需要包含 `def __len__(self)` 和 `def __getitem__(self, idx)`
- batch_size: 每个 batch 的样本数。  
- shuffle: 是否打乱样本顺序（训练时通常 True）。  
- num_workers: 子进程数量（Linux 上可 >0，提高 IO 性能）。  
- drop_last: 是否丢弃最后不足一个 batch 的数据。  
- pin_memory: 若使用 GPU，把数据固定到页内存可加速拷贝。  
- collate_fn: 控制如何把多个 __getitem__ 返回的样本合并为 batch（默认把 tensor 列表 stack）。可用于可变长序列的 padding。

**什么时候不需要自己写 collate_fn**:
- getitem 返回固定形状的 torch.Tensor（例如 3×224×224 的图像）或固定格式的元组 (input, target)，default_collate 能自动把它们 stack 成 tensor。
- 返回值是数值、tensor、或相同结构的嵌套 list/tuple/dict，且每个样本对应项的 shape 相同。

**什么时候需要自己写 collate_fn**：
- 可变长度序列（文本、可变长度音频、点云等）需要 padding 或生成 attention mask。
- getitem 返回 dict，且不同样本可能缺少某些键或某键对应 shape 可变。

#### 改造成 DataLoader 可用
先介绍一下 **DataLoader**：
- DataLoader 会调用 dataset.len() 和 dataset.getitem(idx)。
- 常见参数：batch_size, shuffle, drop_last, num_workers, collate_fn。
- 默认 collate_fn 会把返回的 tensor 列表 stack 成 batch。若样本长度可变，需要自定义 collate_fn 做 padding。

## torch 的用法
### torch.Tensor
将序列转化成 Tensor
```py
torch.as_tensor(tokens, dtype = torch.long) # 返回 Tensor
```
### torch.nn.Module
#### torch.nn.Linear
```py
nn.Linear(in_features, # 输入的神经元个数
          out_features, # 输出神经元个数
          bias=True # 是否包含偏置
         )
```
用法：
```py
model = nn.Linear(2, 1) # 输入特征数为2，输出特征数为1

input = torch.Tensor([1, 2]) # 给一个样本，该样本有2个特征（这两个特征的值分别为1和2）
output = model(input)
```
#### torch.nn.Embedding
```py
nn.Embedding(
    num_embeddings: int,          # 必选：离散特征的总数量（如词汇表大小、类别数）
    embedding_dim: int,           # 必选：嵌入向量的维度（如 128、256 维）
    padding_idx Optional[int] = None,  # 可选：填充索引（该索引的嵌入向量恒为0，且不参与训练）
)
```
作用：把整数 id 查表转成 (embedding_dim,) 的向量，padding_idx 的向量恒为 0，不参与训练
内部参数：一个 (num_embeddings, embedding_dim) 的矩阵，

#### torch.nn.Droput
随机把一些输出置零，即随机把几个神经元置 0
> 一个神经元 = 隐藏层的一个输出值
```tex
正常：   [0.5, 0.3, 0.8, 0.2, 0.6]
Dropout: [0.5, 0.0, 0.8, 0.0, 0.6]  ← 随机把一些变成 0
              ↑        ↑
            被"丢弃"了
```
**目的**：防止过拟合
> 过拟合：模型把训练数据"背下来"了，但在新数据上表现差。

**Dropout 通常加在哪里？**：
```tex
输入
  ↓
Linear(隐藏层1)
  ↓
ReLU
  ↓
Dropout  ← 👈 这里
  ↓
Linear(隐藏层2)
  ↓
ReLU
  ↓
Dropout  ← 👈 这里
  ↓
Linear(输出层)  ← 这里不加 Dropout！
  ↓
输出 logits
```

#### torch.nn.BatchNorm1d
在做的事情：
```py
x_norm = (x - mean) / sqrt(var + eps)   # 归一化
output = gamma * x_norm + beta          # 缩放和平移
```
这个 gamma, beta 是学习的参数

#### torch.no_grad()
PyTorch 会在前向传播时记录计算图

在我们 `with torch.no_grad()` 时，只计算结果，不记录任何东西，会省空间
> 不要觉得记录占的空间很小，大模型推理时，计算图可能占用几个 GB 的显存！

#### torch.nn.Module.eval
禁用 `dropout`，不丢弃全部保留

对于 `BatchNorm`，用之前训练的全局 mean/var

### torch.autograd
PyTorch 采用「动态计算图」（Dynamic Computational Graph）—— 前向传播时逐行执行代码，每执行一个 Tensor 张量运算（如 mean, sum, 或者 nn.Linear 中设计到的运算），就会在内存中构建一个「节点（Node）」和「边（Edge）」组成的有向无环图（DAG）

- 节点（Node）：对应一个运算，由 torch.autograd.Function 类封装；
- 边（Edge）：对应张量的依赖关系（如 fc1 的输出是 ReLU 的输入，ReLU 的输出是 fc2 的输入）；
- 特点：图随前向传播动态构建，每次前向都会重建（灵活支持分支、循环等动态逻辑）。
#### torch.autograd.Function
每个张量运算（如 nn.Linear、torch.relu、nn.MSELoss）都对应一个继承自 torch.autograd.Function 的类，这个类必须实现两个方法：
1. forward(ctx, *inputs)：前向计算逻辑，ctx 用于存储反向计算需要的中间结果；
2. backward(ctx, *grad_outputs)：反向梯度计算逻辑，接收 “上游梯度”（下游节点传递的梯度），返回 “下游梯度”（对当前节点输入的梯度），核心是实现链式法则。

#### 反向传播
在计算完 loss 之后，应该手动调用 `loss.backward()`，因为一旦这个触发，Autograd 会执行以下步骤，**自动推导梯度计算顺序**：

**步骤 1：从 loss 出发，构建「反向依赖链」**
loss 是标量（如 MSE 损失），它的 grad_fn 指向最后一个运算的 Function 节点（如 MseLossBackward）。Autograd 会从这个节点开始，通过 grad_fn.next_functions 递归遍历所有上游节点，构建完整的反向依赖链。

**步骤 2：对依赖链做「反向拓扑排序」**
计算图是有向无环图（DAG），Autograd 会对反向依赖链做拓扑排序（保证：处理一个节点前，它的所有下游节点都已处理完毕）。

**步骤 3：按拓扑序调用 backward 方法，链式计算梯度**

**步骤 4：终止于叶子节点**
当遍历到「叶子节点」（模型参数，grad_fn=None）时，梯度计算终止，梯度值会被累积到 .grad 属性中（需注意：model.zero_grad() 需手动调用，否则梯度会累加）。