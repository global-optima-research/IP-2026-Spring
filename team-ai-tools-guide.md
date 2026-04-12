# 团队 AI 工具配置指南 & 学术免费额度申请

> 本文档提供详细的分步操作，帮助团队每位成员在 10–15 分钟内完成 AI 编程助手的配置，零成本加速科研。

---

## 免费学术额度总览

| 平台 | 方式 | 额度 | 申请条件 |
|---|---|---|---|
| **GitHub Copilot** | GitHub Education 学生包 | **完全免费**，无限使用 | .edu 邮箱 + 学生证明 |
| **Cursor Pro** | 学生认证 | 免费 Pro（每月 500 次高级模型调用） | .edu 邮箱 |
| **Claude** | 免费注册 | 免费层（有对话次数限制） | 邮箱注册即可 |
| **ChatGPT** | 免费注册 | 免费层（GPT-4o 有次数限制） | 邮箱注册即可 |
| **Google Cloud** | Google for Education | $50–$300 credits | 教师/学生均可 |
| **Azure** | Azure for Students | $100 credits/年 | .edu 邮箱，无需信用卡 |

---

## 第一步：申请 GitHub Copilot（必做，优先级最高）

GitHub Copilot 是目前最实用的 AI 编程助手，学生完全免费。

### 1.1 申请 GitHub Education 学生包

1. 打开 https://github.com/signup ，如果还没有 GitHub 账号，先注册一个
   - 建议使用你的 **.edu.cn / .edu** 学校邮箱注册，或注册后在 Settings → Emails 中添加学校邮箱
2. 打开 https://education.github.com/benefits
3. 点击 **"Sign up for Student Developer Pack"**（或 "Get student benefits"）
4. 选择身份 **Student**
5. 选择你的学校邮箱（如果列表里没有，手动输入学校名称）
6. 上传学生身份证明，以下任一即可：
   - 学生证照片
   - 在读证明截图
   - 学信网截图（https://www.chsi.com.cn/ 登录后截图在读状态）
   - 校园门户截图（能看到你的姓名和在读状态）
7. 点击提交，等待审核
   - 通常 **几分钟到几天** 内通过
   - 审核状态在 https://education.github.com/discount_requests/application 查看

### 1.2 在 VS Code 中安装 Copilot

审核通过后：

1. 打开 VS Code
2. 点击左侧 **Extensions（扩展）** 图标（或按 `Ctrl+Shift+X`）
3. 搜索 `GitHub Copilot`，安装以下两个插件：
   - **GitHub Copilot** — 代码自动补全
   - **GitHub Copilot Chat** — AI 对话（可以问问题、解释代码）
4. 安装后右下角会弹出登录提示，点击 **Sign in to GitHub**
5. 浏览器跳转授权页面，点击 **Authorize**
6. 回到 VS Code，右下角出现 Copilot 图标即表示激活成功

### 1.3 在 JetBrains（PyCharm）中安装 Copilot

1. 打开 PyCharm → `File` → `Settings` → `Plugins`
2. 搜索 `GitHub Copilot`，点击 **Install**
3. 重启 PyCharm
4. 右下角点击 Copilot 图标 → **Login to GitHub** → 浏览器授权

### 1.4 验证是否生效

新建一个 Python 文件，输入以下内容：

```python
def fibonacci(n):
```

如果 Copilot 自动给出灰色的补全建议，按 `Tab` 接受，说明已经正常工作。

---

## 第二步：安装 Cursor（推荐，AI 对话 + 编码一体化）

Cursor 是一个内置 AI 的代码编辑器（基于 VS Code），学生免费获得 Pro 版。相比 Copilot 的优势是可以**对话式**操作：选中代码让 AI 解释、修改、重构。

### 2.1 下载安装

1. 打开 https://www.cursor.com/
2. 点击 **Download**，下载对应系统版本（Windows / Mac / Linux）
3. 安装并打开

### 2.2 注册 + 学生认证

1. 首次打开 Cursor 会提示登录，点击 **Sign Up**
2. **使用 .edu 邮箱注册**（这一步很关键，决定是否能拿到学生 Pro）
3. 注册完成后，打开 https://www.cursor.com/settings
4. 在 Account 页面查看是否显示 **Pro (Student)** 标识
5. 如果没有自动识别，在设置页面找到学生认证入口，提交 .edu 邮箱验证

### 2.3 基本使用

| 操作 | 快捷键 | 说明 |
|---|---|---|
| AI 对话 | `Ctrl+L` | 打开右侧 AI Chat 面板，可以提问、贴代码 |
| 行内编辑 | `Ctrl+K` | 选中代码后按此键，输入指令让 AI 修改 |
| 自动补全 | 输入时自动触发 | 和 Copilot 类似的代码补全 |
| @ 引用文件 | 在对话中输入 `@filename` | AI 会读取该文件内容作为上下文 |

### 2.4 实用技巧

- 选中一段报错日志，按 `Ctrl+L`，直接问 "这个报错怎么修"
- 选中一个函数，按 `Ctrl+K`，输入 "加上类型注解"
- 在对话中输入 `@base.py` 可以让 AI 理解项目中某个文件的上下文

---

## 第三步：注册 Claude 和 ChatGPT（免费层）

这两个作为日常问答、文献调研、写作辅助使用。

### 3.1 注册 Claude

1. 打开 https://claude.ai/
2. 点击 **Sign up**，用邮箱注册（任何邮箱均可）
3. 验证邮箱后即可使用
4. 免费层可使用基础模型，有每日对话次数限制

**适合场景**：长文档分析、论文写作润色、代码解释

### 3.2 注册 ChatGPT

1. 打开 https://chat.openai.com/
2. 点击 **Sign up**，用邮箱或 Google 账号注册
3. 免费层可使用 GPT-4o（有次数限制，超出后降级到 GPT-4o-mini）

**适合场景**：通用问答、数据分析（可上传文件）、画图（DALL-E）

---

## 第四步（可选）：申请云平台学生 Credits

如果你需要 GPU 云资源或 API 调用额度：

### 4.1 Azure for Students（推荐，最简单）

1. 打开 https://azure.microsoft.com/en-us/free/students/
2. 点击 **Start free**
3. 用 .edu 邮箱登录 Microsoft 账号（没有就注册一个）
4. 验证学生身份（通常自动识别 .edu 邮箱）
5. 获得 **$100 credits**，无需信用卡
6. 可用于：Azure OpenAI Service（调用 GPT-4 API）、虚拟机、存储等

### 4.2 Google Cloud for Education

1. 打开 https://cloud.google.com/edu
2. 通常需要通过学校或课程申请
3. 也可以直接申请 Google Cloud Free Trial：https://cloud.google.com/free
   - 新用户 $300 credits（90 天有效），需要信用卡但不会自动扣费
4. 可用于：Vertex AI（Gemini API）、Compute Engine（GPU 实例）

---

## 进阶：团队共享 API 方案

如果团队需要更深度使用 AI（大量 API 调用、批量处理），可以搭建共享服务。

### 方案 A：Open WebUI + API Key（推荐）

在团队服务器上部署一个 Web 界面，所有人通过浏览器访问，后端调用 API。

**前提**：服务器上已安装 Docker。

```bash
# 1. 一键启动 Open WebUI
docker run -d \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# 2. 浏览器打开 http://服务器IP:3000
# 3. 首次访问会创建管理员账号
# 4. 进入 Settings → Connections，填入 API Key：
#    - OpenAI API Key（从 https://platform.openai.com/api-keys 获取）
#    - 或 Anthropic API Key（从 https://console.anthropic.com/ 获取）
# 5. 为每位团队成员创建账号
```

每位成员在浏览器中打开 `http://服务器IP:3000`，登录后即可使用，体验和 ChatGPT 网页版类似。

### 方案 B：直接购买 Team Plan

| 平台 | 价格 | 操作 |
|---|---|---|
| Claude Team | $30/人/月 | https://claude.ai/ → 左下角 → Upgrade → Team |
| ChatGPT Team | $25/人/月 | https://chat.openai.com/ → Settings → Upgrade → Team |

管理员付费后邀请成员加入，零运维。

---

## 导师申请研究额度（适合大规模实验）

由导师以课题组名义申请，可获得远超个人免费层的额度：

### OpenAI Researcher Access Program

1. 打开 https://openai.com/form/researcher-access-program/
2. 填写：研究机构、研究方向、预期用途、已发表论文
3. 提交后等待审核（通常 1–2 周）
4. 通过后获得 API credits 直接充入账户

### Anthropic Research Access

1. 打开 https://www.anthropic.com/ ，找到 Contact / Research 入口
2. 发邮件说明研究用途、团队规模、预期调用量
3. 附上论文发表记录

### Google Gemini API

- 免费层：15 requests/min，足够轻度研究使用
- 申请地址：https://aistudio.google.com/
- 直接用 Google 账号登录即可拿到 API Key

---

## 快速行动清单

按优先级排序，每人逐项完成：

- [ ] **【5 分钟】** 注册 GitHub 账号（用 .edu 邮箱），申请 Education 学生包
- [ ] **【3 分钟】** 在 VS Code / PyCharm 中安装 GitHub Copilot 插件并登录
- [ ] **【5 分钟】** 下载安装 Cursor，用 .edu 邮箱注册并获取学生 Pro
- [ ] **【2 分钟】** 注册 Claude 账号（https://claude.ai/）
- [ ] **【2 分钟】** 注册 ChatGPT 账号（https://chat.openai.com/）
- [ ] **【可选】** 申请 Azure for Students $100 credits
- [ ] **【可选】** 和导师沟通申请 OpenAI / Anthropic 研究额度

---

## 常见问题

**Q: GitHub Education 审核不通过怎么办？**  
A: 确保上传的证明文件清晰可读，包含姓名和在读状态。推荐使用学信网在线验证报告截图。被拒后可以重新提交。

**Q: Copilot 和 Cursor 可以同时用吗？**  
A: 可以。Copilot 在 VS Code 里用，Cursor 是独立编辑器。根据场景切换：Copilot 适合纯写代码，Cursor 适合需要和 AI 对话讨论的场景。

**Q: 免费层额度用完了怎么办？**  
A: Claude 和 ChatGPT 免费层每天/每几小时重置。如果频繁用完，考虑升级 Pro（~$20/月）或走团队共享 API 方案。

**Q: 我用的是 .edu.cn 邮箱，可以吗？**  
A: 可以。GitHub Education 和 Cursor 都支持 .edu.cn 后缀的邮箱。

**Q: 这些工具会泄露我们的代码吗？**  
A: GitHub Copilot Business/Education 版本明确不使用你的代码训练模型。Cursor Pro 同样承诺不训练。如果特别敏感，可以在设置中关闭 telemetry。
