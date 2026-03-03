# Week 2 进度报告 — 王洁怡
**日期**：2026-02-23 ~ 2026-03-01  
**角色**：数据采集 & 预处理 Pipeline  
**项目**：PVTT (Product Video Template Transfer)  
**服务器**：RTX 5090（已配置 PySceneDetect / FFmpeg / rembg / Conda）

---

## 本周完成内容

### 1. 技术文档阅读与理解
- 精读师兄整理的《Video Editing 技术调研》，掌握项目整体技术背景：Text-Guided Video Editing、Video Inpainting、DiT 架构演进等
- 精读《PVTT 数据集构建技术综述》，理解完整四阶段数据管线：

```
阶段1 预处理        →  阶段2 交叉配对       →  阶段3 视频合成         →  阶段4 质量过滤
PySceneDetect          模板视频 ×               VideoAnyDoor /             DINO-I / CLIP-I
TransNetV2             电商产品图像             InsertAnywhere             MUSIQ / DOVER
Grounded-SAM2          → 训练对                → 合成 Ground Truth         Qwen2-VL 语义评分
SAM2 / VideoPainter
```

- 明确自己负责的工具链：**Grounded-SAM2 → SAM2 → VideoPainter → PySceneDetect**

### 2. 电商视频采集环境搭建

注册并配置三个采集 API 平台：

| 平台 | API 服务 | 负责采集内容 | 状态 |
|------|---------|------------|------|
| 淘宝/天猫 | TMAPI | 商品图片 + 主图视频 | ✅ 账号已注册 |
| Amazon / Etsy | Apify | 商品多视角图片 | ✅ 账号已注册 |
| TikTok | ScrapeCreators | 商品视频 + 推广视频 | ✅ 账号已注册 |
| 小红书 | Apify | 笔记图片 + 视频 | ✅ 账号已注册 |

### 3. 采集框架代码开发

搭建 Python 五平台统一采集框架，共 7 个模块：

```
ecommerce_scraper/
├── config.py                        # API Keys + 品类关键词配置
├── main.py                          # 主调度（metadata / download / report）
├── media_downloader.py              # 异步并发批量下载
└── scrapers/
    ├── taobao_scraper.py            # 淘宝/天猫（TMAPI）
    ├── amazon_etsy_scraper.py       # Amazon + Etsy（Apify）
    └── tiktok_xhs_scraper.py        # TikTok + 小红书（ScrapeCreators）
```

### 4. API 调试与问题修复

调试过程中发现并修复以下问题：

| 平台 | 问题 | 原因 | 解决方案 |
|------|------|------|----------|
| 淘宝 | 404 错误 | TMAPI 无关键词搜索接口 | 改为按品牌店铺 URL 采集 |
| Amazon | Actor 404 | Apify ID 中 `/` 应为 `~` | 修正格式：`junglee~amazon-crawler` |
| Etsy | Actor 404 | Actor ID 错误 | 改为 `epctex~etsy-scraper` |
| TikTok | 脚本崩溃 | `title=None` 调用 `.replace()` | 添加 `str(... or "product")` 保护 |
| TikTok | 采集结果 0 条 | 31 个商品 `product_id` 全为 None | 待解决（见下周计划）|
| Amazon | 403 付费错误 | Apify Actor 免费试用已过期 | 待解决（见下周计划）|

### 5. 服务器环境确认
- SSH 成功连接 5090 服务器
- 确认已安装：PySceneDetect、FFmpeg、rembg、Conda/Python 环境
- 服务器已就绪，等待视频数据传入后可立即开始镜头切割

---

## 遇到的主要问题

**API 免费额度限制**：ScrapeCreators 和 Apify 的免费额度已接近耗尽，TikTok Shop 返回的 product_id 全部为 None，Amazon Actor 需要付费租用。本周未能采集到实际视频文件。

---

## 下周计划（Week 3）

### 优先任务 1：换用免费采集方案，获取第一批真实视频

放弃付费 API，改用 `yt-dlp` 直接下载 TikTok 产品视频：

```bash
pip install yt-dlp
yt-dlp "https://www.tiktok.com/tag/watchreview" --max-downloads 50 -o "data/tiktok/手表/%(id)s.mp4"
yt-dlp "https://www.tiktok.com/tag/jewelryshowcase" --max-downloads 50 -o "data/tiktok/珠宝/%(id)s.mp4"
```

目标：本地下载 **50+ 产品展示视频**，覆盖手表、珠宝至少两个品类。

### 优先任务 2：将视频传到服务器，跑 PySceneDetect 镜头切割

```bash
# 传输视频到服务器
scp -r data/tiktok/ username@服务器IP:~/data/raw_videos/

# 服务器上批量切割镜头（目标：每视频切出 3-8 个 1.5-5s 片段）
cd ~/data/raw_videos
for f in *.mp4; do
    scenedetect -i "$f" detect-adaptive split-video -o ~/data/shots/
done
```

目标：产出 **200+ 单镜头片段**，去除转场和无效帧。

### 优先任务 3：视频标准化

用 FFmpeg 统一格式（720p / 24fps / 2-4s）：

```bash
for f in ~/data/shots/*.mp4; do
    ffmpeg -i "$f" -vf scale=1280:720 -r 24 -t 4 ~/data/standardized/$(basename "$f")
done
```

### 次要任务
- 整理产品多视角图片采集方案（rembg 去背景）
- 阅读 Grounded-SAM2 部署文档，为产品分割做准备
- 建立 JSON 数据索引（产品ID、品类、视角、分辨率）

---

## 里程碑进度

| 里程碑 | 截止 | 状态 | 备注 |
|--------|------|------|------|
| 完成 50+ 产品视频采集和镜头切割 | Week 1-2 | 🟡 延迟至 Week 3 | API 付费限制，改用 yt-dlp |
| 完成视频标准化和产品图片采集 | Week 3-4 | ⬜ 未开始 | |
| 完成数据索引系统 | Week 5 | ⬜ 未开始 | |

---

*最后更新：2026-03-01*
