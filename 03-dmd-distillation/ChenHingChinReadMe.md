# 陈庆展 (Chen Hing Chin) — 个人工作看板

> **项目:** PVTT (Product Video Template Transfer) | IP-2026-Spring
> **所属任务组:** Task 3 — DMD 蒸馏加速（4人组）
> **小组 Leader:** 李志颖、Jacky
> **联系方式:** hcchenab@connect.ust.hk
> **开发分支:** `Task3_dev_ChenHingChin`
> **独立仓库:** [`global-optima-research/ChenQingzhan-DMD-distillation`](https://github.com/global-optima-research/ChenQingzhan-DMD-distillation)
> **工作目录:** `03-dmd-distillation/`

---

## 我的角色

我是**陈庆展**，负责 PVTT 项目 **Task 3（DMD 蒸馏加速）** 中的**渐进式蒸馏 & Discriminator** 方向。我的核心使命是：通过 Distribution Matching Distillation 将 50 步 Teacher Model 渐进压缩至 **4 步 Student Model**，实现 4-8× 推理加速，质量损失控制在 5% 以内。

---

## 项目全局背景

PVTT 是面向电商场景的产品视频智能编辑系统，整体流程为：

```
Task 1 (数据集构建 · 20K+ 训练对)
  → Task 2 (Teacher Model 训练 · Wan2.1/2.2 + LoRA)
    → Task 3 (DMD 蒸馏加速 · 50步→4步)  ← 我在这里
      → 最终交付：4 步快速推理的 PVTT Student Model
```

**关键交接节点：**
- Week 8-9：Task 1 → Task 2（训练数据交付）
- **Week 12：Task 2 → Task 3（Teacher Model 交付 — 我需要关注此节点）**
- Week 16：Task 3 产出最终 4 步推理模型

---

## 我的具体任务清单

### Phase 0：现有方案复现（Week 1-6，全组共同参与）

> **核心原则：先复现，再改进。**

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| FastGen 环境搭建 | 搭建 NVIDIA FastGen 框架运行环境，配置 GPU 服务器依赖 | FastGen | 可运行的开发环境 |
| DMD2 复现 | 使用 FastGen 框架复现 DMD2 方法在 Wan2.1 上的蒸馏（50→4 步 T2V） | FastGen + Wan2.1 | 蒸馏模型 + 训练日志 |
| ECT 复现 | 复现 FastGen 中的 ECT (Enhanced Consistency Training) 方法 | FastGen | ECT 蒸馏结果 |
| Consistency 复现 | 复现 Consistency Distillation 方法作为对比基线 | FastGen | CD 蒸馏结果 |
| 多方法对比 | 横向对比 DMD2 vs ECT vs Consistency 的收敛速度、生成质量、显存占用 | FastGen | 方法对比报告 |
| 复现报告 | 提交完整复现报告（环境配置/训练配置/Loss曲线/生成质量/踩坑记录/框架评估） | 文档 | 复现报告 |

### Phase 1+：深入研发（Week 7-16）

> **基于 Phase 0 选定的最优 codebase，进入针对 PVTT 视频编辑任务的深入研发。**

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| 3D Video Discriminator 设计 | 基于 FastVideo/FastGen 已有 Discriminator，适配视频编辑场景；采用**潜在空间判别器**（参考 SDXL-Lightning 设计，比像素空间 DINOv2 判别器更省内存）；需区分商品区域 vs 背景区域 | PyTorch | Discriminator 架构 |
| 3D Video Discriminator 训练 | 训练判别器，用于后续蒸馏阶段的对抗损失 | PyTorch + DeepSpeed | 训练好的 Discriminator |
| 蒸馏阶段 1：50→16 步 | 损失：L_DMD + L_regression + L_temporal；~20K 训练步；学习率 1e-4 | FastVideo + DeepSpeed | 16 步 Student |
| 蒸馏阶段 2：16→8 步 | 损失：阶段 1 + L_bg + L_id；~20K 训练步；学习率 5e-5 | FastVideo + DeepSpeed | 8 步 Student |
| 蒸馏阶段 3：8→4 步 | 损失：全部损失 + 可选 L_adv；~20K 训练步；学习率 1e-5 | FastVideo + DeepSpeed | 4 步 Student |
| 质量门槛控制 | 每阶段设定质量门槛（CLIP-I / FVD / BG-PSNR），不达标则延长训练或调整策略 | 评测体系 | 质量达标报告 |
| EMA 策略 | 设计 Student 模型的 EMA 更新策略，提升训练稳定性 | PyTorch | EMA 配置 |

---

## 关键技术要点（摘自 Task 3 技术综述）

### DMD2 核心原理

DMD 通过分布级别的匹配（而非样本级别）将扩散模型转为少步生成器：

- **分布匹配损失**: `L_DMD = KL(p_data || p_student)`，梯度表示为两个 score function 的差异
- **DMD2 关键改进**：消除回归损失（降低训练成本）、集成 GAN 损失、支持多步采样、修正输入不匹配
- **性能基线**: ImageNet-64 FID 1.28，推理成本降低 500×

### 渐进式蒸馏策略

采用 Progressive Distillation (Salimans & Ho, ICLR 2022) 范式：
- 每阶段学生用 1 步模拟教师的 2+ 步
- **PVTT 路径**: 50→16（3× 减少）→ 8（2× 减少）→ 4（2× 减少）
- 每阶段充分训练至质量稳定后再进入下一阶段

### 损失函数组合

```
L_total = α · L_DMD           (分布匹配，主损失，α=1.0)
        + β · L_regression    (回归损失，β=0.5-1.0，DMD2可设为0)
        + γ · L_temporal      (时序一致性，γ=0.1-0.5，视频关键)
        + δ · L_bg            (背景保持，δ=0.1-0.3，电商场景重要)
        + ε · L_id            (身份保持，ε=0.1-0.3，产品一致性)
        + ζ · L_adv           (对抗损失，ζ=0.01-0.1，可选)
```

### Discriminator 设计参考

- **SDXL-Lightning 方案**: 潜在空间判别器（用 Diffusion UNet 编码器作主干），比像素空间方案（DINOv2）内存更低
- **视频编辑适配**: 需通过 mask 区分商品区域和背景区域，对两者施加不同的判别权重

---

## 关键里程碑

| 时间 | 里程碑 | 验收标准 | 状态 |
|------|--------|---------|------|
| Week 1-2 | 环境搭建，阅读 FastGen 代码，跑通推理 | FastGen 推理 pipeline 可运行 | ✅ 完成 (2026-03-05) |
| Week 2-3 | 训练数据准备 (OpenVid-1M → WebDataset) | 21K 样本 WebDataset shards | ✅ 完成 (2026-03-07) |
| Week 3-4 | ECT/CD 蒸馏训练验证 + CausVid 预训练对比 | Loss 收敛验证 | ✅ 完成 (2026-03-09) |
| Week 5-6 | Phase 0 报告完成：预训练模型推理对比 (Teacher/CausVid/rCM) | 报告提交 + 方案确定 | ✅ 完成 (2026-03-12) |
| Week 8-10 | 3D Video Discriminator 训练完成 | Discriminator loss 稳定 | ⬜ 待开始 |
| Week 10-12 | 50→16→8 步蒸馏完成 | FVD < Teacher × 1.15 | ⬜ 待开始 |
| Week 12-14 | 8→4 步蒸馏完成 | CLIP-I > 0.80, FVD < 120 | ⬜ 待开始 |
| Week 16 | 最终 4 步 Student Model 交付 | 4-8× 加速, 质量损失 < 5% | ⬜ 待开始 |

---

## 工作进度追踪 (To-Do List)

### Phase 0：复现阶段 (Week 1-6)

- [x] GitHub 分支环境配置 (2026-02-23)
- [x] 阅读 NVIDIA FastGen 源码和文档
- [x] 阅读 Task 3 技术综述 (`03-dmd-distillation/README.md`)
- [ ] 学习 DMD / DMD2 核心论文 (CVPR 2024 / NeurIPS 2024)
- [ ] 学习 Progressive Distillation 论文 (ICLR 2022)
- [x] 搭建 FastGen 运行环境（GPU 服务器配置、依赖安装）(2026-03-05)
- [x] 下载 Wan2.1 模型权重 (2026-03-05)
- [x] 跑通 FastGen 推理 pipeline (2026-03-05, 50-step Teacher, 5 videos)
- [x] 编写五种蒸馏方法的配置和脚本 (2026-03-06, DMD2/ECT/CD/f-distill/LADD)
- [x] 组会讨论确定训练数据方案：统一使用 OpenVid-1M (2026-03-06)
- [x] 编写 OpenVid-1M 下载 + WebDataset 转换脚本 (2026-03-06)
- [x] 创建独立仓库 ChenQingzhan-DMD-distillation (2026-03-06)
- [x] 上传脚本到服务器，完成 OpenVid-1M 数据下载 (2026-03-07)
- [x] 转换 21K 样本为 WebDataset 格式 (2026-03-07)
- [x] DMD2 训练验证 → OOM (4 networks exceed 32GB)，改用 CausVid 预训练 (2026-03-08)
- [x] CausVid 预训练推理：3-step, 28.5s/video, 6.4x speedup (2026-03-08)
- [x] ECT 自训练完成：6000 iter, 36.5h (质量差，作为学习用) (2026-03-09)
- [x] CD 自训练：停在 1500 iter (质量差，转用预训练模型) (2026-03-10)
- [x] rCM 预训练推理：4-step, 37.6s/video, 4.9x speedup (2026-03-12)
- [x] 三模型对比实验完成：Teacher + CausVid + rCM (15 videos) (2026-03-12)
- [x] 撰写 Phase0 Report（含架构分析/推理对比/能力边界/Phase 1 建议）(2026-03-12)
- [x] 脚本整理 + 文档编写 (scripts/ 分 5 类 + README) (2026-03-12)
- [x] 代码提交并推送到远程 (commit 5f5d649) (2026-03-12)
- [ ] 参加全组讨论，选定最优 codebase

### Phase 1+：深入研发阶段 (Week 7-16)

- [ ] 研究 SDXL-Lightning 潜在空间判别器架构
- [ ] 设计 3D Video Discriminator（适配视频编辑场景，区分商品/背景区域）
- [ ] 实现 Discriminator 训练代码
- [ ] 训练 3D Video Discriminator
- [ ] 配置蒸馏阶段 1 损失函数：L_DMD + L_regression + L_temporal
- [ ] 执行蒸馏阶段 1：50 步 → 16 步（~20K 训练步，lr=1e-4）
- [ ] 质量门槛检查：FVD < Teacher × 1.15, 速度提升 ~3×
- [ ] 配置蒸馏阶段 2 损失函数：+ L_bg + L_id
- [ ] 执行蒸馏阶段 2：16 步 → 8 步（~20K 训练步，lr=5e-5）
- [ ] 质量门槛检查：FVD < Stage1 × 1.10, BG-PSNR > 35dB
- [ ] 配置蒸馏阶段 3 损失函数：全部损失 + 可选 L_adv
- [ ] 执行蒸馏阶段 3：8 步 → 4 步（~20K 训练步，lr=1e-5）
- [ ] 质量门槛检查：CLIP-I > 0.80, FVD < 120, 速度提升 ~12.5×
- [ ] 设计和调优 EMA 更新策略
- [ ] 最终 4 步 Student Model 交付和文档整理

---

## 核心参考 Codebase

| 项目 | GitHub | 优先级 | 备注 |
|------|--------|--------|------|
| **NVIDIA FastGen** | [NVlabs/FastGen](https://github.com/NVlabs/FastGen) | ⭐ 最高（我的主要复现目标） | 统一蒸馏框架，支持 DMD2/ECT/MeanFlow/CausVid 等 10+ 方法；原生支持 Wan2.1 14B；64×H100 上 16h 收敛 |
| **FastVideo** | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) | ⭐ 最高 | DMD + Sparse Attention 联合训练；原生支持 Wan2.1/2.2；H200 上 5s 生成 480p 5s 视频 |
| **distill_wan2.1** | [azuresky03/distill_wan2.1](https://github.com/azuresky03/distill_wan2.1) | ⭐ 高 | 支持 DMD2 + Consistency + RL；代码清晰，适合学习 |
| **LightX2V** | [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) | ⭐ 高 | 4 步无 CFG 推理，~20× 加速 |
| **CausVid** | [tianweiy/CausVid](https://github.com/tianweiy/CausVid) | 参考 | CVPR 2025，DMD 扩展到自回归流式生成，VBench 第一名 (84.27) |
| **TurboDiffusion** | [thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) | 参考 | rCM + SLA + W8A8，100-200× 加速 |

---

## 核心论文阅读清单

| 论文 | 会议 | 与我的工作关系 |
|------|------|---------------|
| DMD: One-step Diffusion with Distribution Matching Distillation | CVPR 2024 | 蒸馏方法核心 |
| DMD2: Improved Distribution Matching Distillation | NeurIPS 2024 Oral | 蒸馏方法改进版（消除回归损失 + GAN 损失） |
| Progressive Distillation for Fast Sampling | ICLR 2022 | 渐进式蒸馏策略（50→16→8→4） |
| Consistency Models | ICML 2023 | 一致性模型（对比基线） |
| SDXL-Lightning: Progressive Adversarial Diffusion Distillation | ByteDance | 潜在空间判别器设计参考 |
| CausVid | CVPR 2025 | DMD 的视频自回归扩展 |

---

## 质量目标

| 指标 | Teacher (基线) | Stage 1 (16步) | Stage 2 (8步) | Stage 3 (4步) |
|------|---------------|----------------|---------------|---------------|
| CLIP-I | > 0.85 | > 0.83 | > 0.82 | > 0.80 |
| FVD | < 100 | < 115 | < 118 | < 120 |
| BG-PSNR | > 30dB | > 30dB | > 35dB | > 30dB |
| 推理步数 | 50 步 | 16 步 | 8 步 | 4 步 |
| 加速比 | 1× | ~3× | ~6× | 4-8× |

---

## 团队协作

**我的 Task 3 队友：**
| 成员 | 方向 | Phase 0 复现任务 |
|------|------|-----------------|
| **李一佳** | DMD 蒸馏核心 & Loss 设计 | FastVideo — DMD 蒸馏 on Wan2.1 |
| **陈庆展（我）** | 渐进式蒸馏 & Discriminator | FastGen — DMD2 on Wan2.1 |
| **Sze Long** | 工程加速 & 量化部署 | distill_wan2.1 + LightX2V |
| **邱张西子** | Mean Flow 方法探索 | FastGen — MeanFlow / rCM on Wan2.1 |

**上游依赖：**
- Task 2（钟蕊、肖泓锐、方子维）→ 产出 Teacher Model（Week 12 交付）

---

> **最后更新:** 2026-03-12
