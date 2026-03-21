# 核技巧图解

## 什么是核技巧？

```mermaid
graph TB
    subgraph "输入空间"
        A["原始数据点 x₁, x₂"] --> B["核函数 K(x₁,x₂)"]
        C["原始数据点 x₃, x₄"] --> B
    end

    subgraph "特征空间 (希尔伯特空间)"
        D["映射 φ(x₁)"] --> E["内积 ⟨φ(x₁),φ(x₂)⟩"]
        F["映射 φ(x₂)"] --> E
        G["映射 φ(x₃)"] --> H["内积 ⟨φ(x₃),φ(x₄)⟩"]
        I["映射 φ(x₄)"] --> H
    end

    B -.-> D
    B -.-> F
    C -.-> G
    C -.-> I

    E --> J["等价"]
    H --> J
    J --> K["直接计算核函数"]

    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style F fill:#f3e5f5
    style K fill:#e8f5e8
```

## 核技巧的工作原理

```mermaid
sequenceDiagram
    participant Input as 输入空间
    participant Kernel as 核函数
    participant Feature as 特征空间
    participant SVM as 非线性SVM

    Input->>Kernel: 提供原始数据点
    Kernel->>Feature: 隐式映射
    Feature->>Kernel: 计算内积
    Kernel->>SVM: 等价于高维空间中的线性SVM
    SVM->>Input: 返回分类决策边界
```

## 关键概念解释

| 概念 | 定义 | 说明 |
|------|------|------|
| 输入空间 | 欧式空间或离散集合 | 原始数据所在的空间 |
| 特征空间 | 希尔伯特空间 | 高维映射后的空间 |
| 核函数 | K(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩ | 避免显式映射的计算技巧 |
| 映射函数 | φ(·) | 将输入映射到高维空间 |

## 常见核函数

```mermaid
graph LR
    A['核函数家族'] --> B["线性核 K(x,x')=⟨x,x'⟩"]
    A --> C["多项式核 K(x,x')=(⟨x,x'⟩+c)^d"]
    A --> D["RBF/高斯核 K(x,x')=exp(-γ||x-x'||²)"]
    A --> E["Sigmoid核 K(x,x')=tanh(κ⟨x,x'⟩+θ)"]
```

## 实际应用流程

```mermaid
flowchart LR
    subgraph "训练阶段"
        A["原始训练数据"] --> B["选择核函数"]
        B --> C[隐式映射到高维空间]
        C --> D[在高维空间训练线性SVM]
        D --> E[学习决策边界]
    end

    subgraph "预测阶段"
        F[新测试样本] --> G[应用相同核函数]
        G --> H[在高维空间进行分类]
        H --> I[输出预测结果]
    end

    E --> J[模型参数]
    J --> G
```