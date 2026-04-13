# Weekly Report - w8

- **姓名：** 钟蕊
- **日期：** 2026-04-13

---

## 1. 研究领域

基于FFGO（CVPR 2026）首帧引导范式，构建完整的产品视频模板迁移（PVTT）研究框架：给定源视频A + 参考商品图B，生成外观替换为目标商品B、运动与场景保持一致的新视频。研究路线包含4个阶段——数据集构建、重量级Baseline（FFGO源模型）、轻量级Baseline对比（VACE 1.3B）、以及后续优化方法。

## 2. 领域核心问题

- 快速验证VACE模型在新类别产品上的有效性
- 对比FFGO原论文Baseline
- 优化方法：
  - **方向A — 跨帧身份持续显式注入**：利用冻结DINOv2编码器提取产品特征，通过cross-attention在每步denoising中持续注入
  - **方向B — LoRA微调优化**：在小规模数据集上对VACE 1.3B进行LoRA微调，增强产品身份保持能力
  - **方向C — 双条件CFG训练**：训练时同时使用"完美擦除首帧"和"未擦除原始首帧"作为条件

## 3. 技术方案

采用VACE + FFGo组合方案：
- **FFGo（Frame-level Foreground Object Removal）**：移除原始产品，生成干净的首帧
- **VACE（Video Conditioned Generation）**：基于FFGo帧和目标产品图进行视频生成
- **PoC验证流程**：准备FFGo帧 → VACE推理 → 输出视频

## 4. 本周工作

- [x] 完成PVTT基准测试VACE PoC环境搭建
- [x] 准备5个PoC任务的FFGo帧（0002-handfan2, 0012-handbag2, 0022-bracelet4, 0029-earring1, 0041-watch1）
- [x] 完成2/5个PoC任务的VACE推理验证：
  - ✅ 0002-handfan2_to_handfan1（团扇）
  - ✅ 0041-watch1_to_watch2（手表）
- [ ] 0012-handbag2_to_handbag1（手提包）- 待完成
- [ ] 0022-bracelet4_to_bracelet1（手镯）- 待完成
- [ ] 0029-earring1_to_earring2（耳环）- OOM问题待解决

**代码产出：**
- `ffgo_prepare_frame_v2.py`：FFGo帧准备脚本
- `vace_v2_poc.py`：VACE PoC推理脚本
- 任务目录：`/data/zhongrui/PVTT_Workspace/pvtt_benchmark/results/baseline2_vace_poc/`

## 5. 结论与发现

**踩坑记录：**

- SSH连接超时问题：服务器负载高时需要等待恢复
- Conda环境路径：diffs环境位于`/data/zhongrui/miniconda3/envs/diffs/bin/python`
- VACE脚本参数：不同版本脚本参数不同，需通过`--help`确认

## 6. 下周计划

- [ ] SSH服务器恢复后，完成剩余3个PoC任务（0012、0022、0029）
- [ ] 解决0029耳环任务的OOM问题（可能需要调整参数或分步处理）
- [ ] 收集完整5个任务的VACE输出视频
- [ ] 进行人工质量评估
- [ ] 扩展到完整数据集测试

---

## 附录（可选）

**服务器信息：**
- 地址：111.17.197.107
- 用户：zhongrui
- GPU：8卡配置（需确认具体型号）

**已完成输出示例：**
```
/data/zhongrui/PVTT_Workspace/pvtt_benchmark/results/baseline2_vace_poc/
├── 0002-handfan2_to_handfan1/
│   ├── ffgo_frame.png      # FFGo处理后的首帧
│   ├── vace_full.mp4       # VACE生成的完整视频
│   └── vace.log            # 推理日志
├── 0041-watch1_to_watch2/
│   ├── ffgo_frame.png
│   ├── vace_full.mp4
│   └── vace.log
...
```

**参考脚本：**
- FFGo准备：`/data/zhongrui/PVTT_Workspace/pvtt_benchmark/scripts/ffgo_prepare_frame_v2.py`
- VACE推理：`/data/zhongrui/PVTT_Workspace/pvtt_benchmark/baselines/vace_ffgo/vace_v2_poc.py`
