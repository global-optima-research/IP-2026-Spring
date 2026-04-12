# 团队 AI 工具配置指南 & 学术免费额度申请

> 帮助团队成员快速获取 AI 编程助手，加速科研效率。

---

## 一、免费学术额度一览

| 平台 | 方式 | 额度 | 申请条件 |
|---|---|---|---|
| **GitHub Copilot** | GitHub Education 学生包 | **完全免费** | .edu 邮箱 + 学生证明 |
| **Cursor Pro** | 学生认证 | 免费 Pro（有额度） | .edu 邮箱 |
| **Google Cloud** | Google for Education | $50–$300 credits | 教师/学生均可 |
| **Azure** | Azure for Students | $100 credits/年 | .edu 邮箱，无需信用卡 |
| **AWS** | AWS Educate | 有限 credits | 学生认证 |
| **Cohere** | For AI 研究者 | 免费 API 额度 | 研究用途说明 |

---

## 二、推荐每人必做（10 分钟，零成本）

### 1. GitHub Copilot（必申）

- 访问 [github.com/education](https://github.com/education)，用 .edu 邮箱申请学生包
- 审核通过后（秒批或几天内），在 VS Code / JetBrains 中安装 GitHub Copilot 插件即可使用
- 写代码时自动补全，日常最实用

### 2. Cursor（推荐）

- 下载 [cursor.com](https://cursor.com)
- 用 .edu 邮箱注册，申请学生认证，获得免费 Pro
- 内置 AI 对话 + 代码上下文引用，科研写代码体验优于 Copilot

### 3. Claude / ChatGPT 免费层

- 每人注册自己的账号，免费层够日常问答和文献调研

---

## 三、进阶方案：团队共享 API

### 方案 A：API + 开源前端（性价比最高）

1. 购买 Anthropic API 或 OpenAI API 账号
2. 在服务器上部署开源 Web UI（推荐 **Open WebUI** / **LibreChat** / **LobeChat**）
3. 团队成员通过浏览器访问，后端统一走 API

**优点**：按用量计费、可自托管保隐私、支持多模型切换、可查看每人用量  
**缺点**：需要人维护部署

```bash
# Open WebUI 一键部署示例（Docker）
docker run -d -p 3000:8080 --name open-webui ghcr.io/open-webui/open-webui:main
```

### 方案 B：Claude Team / ChatGPT Team

| 平台 | 价格 | 特点 |
|---|---|---|
| Claude Team | $30/人/月 | 管理后台、对话不用于训练 |
| ChatGPT Team | $25/人/月 | 生态丰富、插件多 |

适合不想折腾、预算充足的团队。

---

## 四、导师出面申请研究额度

如果团队需要更大量的 API 调用（如跑实验、批量推理），可由导师以课题组名义申请：

| 平台 | 申请方式 | 备注 |
|---|---|---|
| **OpenAI Researcher Access Program** | 提交研究计划 + 发表记录 | 可获得大量 API credits |
| **Anthropic Research Access** | 官网联系说明研究用途 | 针对学术研究者 |
| **Google Gemini API** | 免费层本身额度不低（15 次/分钟） | 轻度使用直接免费 |

---

## 五、快速行动清单

- [ ] 每人用 .edu 邮箱申请 GitHub Education 学生包
- [ ] 每人下载 Cursor 并完成学生认证
- [ ] 注册 Claude / ChatGPT 免费账号
- [ ] （可选）团队讨论是否部署共享 API 前端
- [ ] （可选）导师申请 OpenAI / Anthropic 研究额度
