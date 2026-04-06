# 周报 | Weekly Report

## 基本信息

- **姓名：** 王洁怡
- **日期：** 2026-04-06

---

## 1. 研究领域

电商产品视频编辑（Product Video Template Transfer）。核心场景：商家复用优质视频模板，将模板中的产品替换为自己的产品，自动生成新的产品展示视频。研究重点在于构建该任务的 Benchmark 数据集与评测协议。

## 2. 领域核心问题

- 电商产品视频人工制作成本高，AI 视频编辑是潜在解决方案
- 现有视频编辑 Benchmark（DAVIS、TGVE）均面向通用视频，缺乏针对产品视频编辑的公开 Benchmark
- 产品视频存在独特挑战：产品形状多样、展示方式多样（白底展示 vs 模特佩戴）、产品-人体交互等
- **需要解决的问题：** 如何构建一个系统性的 Benchmark 来评估现有视频编辑模型在产品视频场景下的表现

## 3. 技术方案

构建 **PVTT-Bench**（Product Video Template Transfer Benchmark），包括：

1. **多源数据采集**：Amazon 爬虫 + Shopify API，覆盖 8 个品类
2. **多维度视频分类体系（Taxonomy）**：按编辑难度和视频特征进行分类标注
3. **评测协议设计**：定义任务、评估指标、baseline 模型选择
4. 以论文格式撰写技术报告，后续与团队合作扩展为完整论文

### Report 方向与内容框架

**标题（草案）：** PVTT-Bench: A Multi-Source Benchmark for Product Video Template Transfer

**内容框架：**

```
Abstract

1. Introduction
   1.1 电商产品视频编辑的实际需求
   1.2 现有 Benchmark 的不足（DAVIS/TGVE/MOSE 均为通用视频）
   1.3 贡献总结（数据集 + Taxonomy + 评测协议）

2. Related Work
   2.1 视频编辑模型（TokenFlow, AnyV2V, FFGO 等）
   2.2 现有视频 Benchmark（DAVIS, MOSE, TGVE）
   2.3 电商产品视频相关工作

3. PVTT-Bench Dataset                    ← 核心章节
   3.1 数据采集 Pipeline
       - Amazon 爬虫：关键词搜索 → ASIN → 产品页 → 视频/图片下载
       - Shopify API：/products.json → 产品媒体提取
       - 多平台尝试与选择理由
   3.2 数据集统计与分析
       - 规模、品类分布、视频时长/分辨率分布
       - 与现有数据集对比
   3.3 多维度视频分类体系（Taxonomy）     ← 核心贡献
       主分类（按编辑难度，互斥）：
         L1 - Product Only：产品独立展示，无人，简单背景
         L2 - Styled Scene：产品在场景中，无人，复杂背景
         L3 - Human Interaction：有人参与（佩戴/手持/试用）
       副标签（多选）：
         [static-camera] / [camera-motion] / [occlusion] /
         [multi-shot] / [360-rotation]
   3.4 数据处理（标准化、镜头分割、产品提取）

4. Benchmark Protocol                    ← 核心章节
   4.1 任务定义：V_o = f(V_t, I_p)
   4.2 评估指标（FID, FVD, CLIP-Score, 帧间一致性, 人工评估）
   4.3 评测子集设计

5. Baseline Experiments（待补充，与队友合作）

6. Conclusion & Future Work

Appendix
```

## 4. 本周工作

- [x] **确定 Report 方向与框架**：确定以 Benchmark Paper 格式撰写技术报告，明确标题、核心贡献、论文结构
- [x] **确定视频分类体系方案**：采用多维度标注（3 级主分类 + 5 种副标签），主分类按编辑难度划分（Product Only / Styled Scene / Human Interaction），而非按商品种类
- [x] **数据集 v4 扩展**：运行 Amazon v4（51 个新关键词）和 Shopify v4（37 家新店铺）爬虫
  - Amazon：handbag 155→235（+80），cosmetics 21→68（+47）
  - Shopify：cosmetics +154，watch +159，sunglasses +82，handbag +59，necklace +68，earring +50
  - 部分关键词/店铺超时（Amazon 反爬加强、Shopify 大店铺超时）
- [x] **数据上传服务器**：已上传约 2980 个文件（2.7GB），后因 SSH 连接中断未完成，需重跑

### 当前数据集规模

| 品类 | Amazon | Shopify | 合计 |
|------|--------|---------|------|
| bracelet | 328 | 457 | 785 |
| cosmetics | 68 | 919 | 987 |
| earring | 206 | 930 | 1,136 |
| handbag | 235 | 557 | 792 |
| necklace | 165 | 1,006 | 1,171 |
| ring | 186 | 509 | 695 |
| sunglasses | 192 | 606 | 798 |
| watch | 268 | 802 | 1,070 |
| **合计** | **1,648** | **5,786** | **7,434 产品** |

## 5. 结论与发现

- **数据规模已充足**（7434 产品），后续重心应转向 Taxonomy 标注和 Report 撰写
- **按编辑难度分类优于按商品种类分类**：同一品类内编辑难度差异巨大（如白底手链 vs 模特佩戴手链），按展示方式/编辑挑战分类更能反映 Benchmark 的评测价值
- **Amazon 反爬持续加强**：v4 中大量关键词超时（尤其是 cosmetics），未来不建议继续扩展 Amazon 数据
- **Shopify 仍是最高效的数据源**：API 公开、视频丰富、品类覆盖好

## 6. 下周计划

- [ ] **P0：开始写 Report Section 3（Dataset）**：数据采集 pipeline 描述 + 统计分析图表
- [ ] **P0：Taxonomy 标注试点**：从数据集中抽样 100+ 视频，按主分类 + 副标签进行试标注，验证分类体系的可行性
- [ ] **P1：完成数据上传服务器**：重跑上传脚本，将全部数据同步到服务器
- [ ] **P1：写 Report Section 1（Introduction）初稿**
- [ ] **P2：阅读 MOSE（ICCV 2023）论文**，学习 Benchmark 论文的写作方式

---

## 附录

### Report 写作优先级

| 优先级 | Section | 内容 | 状态 |
|--------|---------|------|------|
| P0 | 3.1-3.2 | 数据采集 + 统计分析 | 数据已有，可直接写 |
| P0 | 3.3 | Taxonomy 定义与标注 | 需看视频确定分类标准 |
| P1 | 1 | Introduction | 框架已定，可以写 |
| P1 | 4.1-4.2 | 任务定义 + 评估指标 | 可写框架 |
| P2 | 2 | Related Work | 需读论文 |
| P3 | 5 | Experiments | 待队友模型就绪 |
