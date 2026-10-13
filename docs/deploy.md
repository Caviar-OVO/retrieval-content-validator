# 部署指南（Deploy）

把「检索内容校验台」发布成网页，让别人也能打开体验。**默认全部离线可跑，
不配置任何密钥也能完整体验**（deterministic critic、hash 检索、heuristic NLI、
judge=local）；只有用到 LLM / ragas / trulens 轨时才需要 `OPENAI_API_KEY`。

仓库里已备好云端构建需要的产物：

- `requirements.txt` —— Streamlit Cloud 的构建/运行依赖（已注释掉可选 judge 依赖）。
- `.streamlit/config.toml` —— headless、关闭用量上报、基础主题。
- `app/streamlit_app.py` —— 主入口文件（部署时填这个路径）。
- `run_ui.py` —— 本地/局域网启动器（自动使用项目 `.venv`，多余参数透传给 `streamlit run`）。

---

## 方案 A · 局域网 / 临时分享（零账号，最快）

同一 Wi-Fi / 内网的人就能打开，适合内部评审、当面演示。

```bash
python run_ui.py --server.address 0.0.0.0
```

- 启动后终端会打印 `Local URL` 和 `Network URL`，把 **Network URL**（形如
  `http://192.168.x.x:8501`）发给对方即可。
- 首次可能在 Windows 弹窗询问是否允许防火墙放行；勾选「专用网络」。
- 对方需与你在同一网络；公网访问不适用（见方案 B）。
- 想换端口：`python run_ui.py --server.address 0.0.0.0 --server.port 8600`。

> 注意：局域网模式下终端 `Ctrl+C` 即关闭服务；对方刷新会断开。

---

## 方案 B · 公网（Streamlit Community Cloud，免费）

得到形如 `https://<app-name>.streamlit.app` 的公开链接，任何人随时访问。

**前置**：一个 GitHub 账号。

### 1. 把代码推到 GitHub

```bash
git init
git add .
git commit -m "RVA validation console"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

- `.gitignore` 已排除 `.env`、`.venv/`、`__pycache__/`，密钥不会被推上去。
- 私有仓库也可部署；Cloud 免费版支持（需要用 GitHub 授权访问私有库）。

### 2. 在 share.streamlit.io 部署

1. 打开 https://share.streamlit.io ，用 GitHub 登录。
2. **Create app** → **Deploy a public app from GitHub**。
3. 选择你的仓库与分支（`main`）。
4. **Main file path** 填：`app/streamlit_app.py`
5. 点 **Deploy**。首次构建约 1–3 分钟，之后每次 `git push` 自动重新部署。

### 3. （可选）配置密钥以启用 LLM 轨

部署页 → **Settings → Secrets**，按 TOML 填入，例如：

```toml
OPENAI_API_KEY = "sk-..."
# 可选：
# OPENAI_BASE_URL = "https://..."   # 走代理或兼容端点
# RVA_JUDGE_MODEL = "gpt-4o-mini"
# RVA_CRITIC = "auto"               # 有 key 且有 sources 才走 LLM critic
```

保存后应用会自动重启，也可以随时在页面侧边栏「🔌 模型 / 端点」临时切换端点与模型（仅影响当前会话，不写回 Secrets）。云端没有 `.env` 文件，配置会从 Secrets 读取
（`rva/config.py` 会把 `st.secrets` 自动桥接进环境变量；真实环境变量与本地
`.env` 优先级更高，本地行为不变）。

> 安全提示：Secrets 里的值不会出现在公开页面，但请勿把 key 写进代码或提交到仓库。

---

## 部署前自检

```bash
python run_ui.py                 # 本地能打开 → 云端基本没问题
.venv/Scripts/python -m unittest discover -s tests   # 250 条应全绿
```

## 常见问题

- **构建失败：装不上 ragas/trulens** —— 它们默认**不在** `requirements.txt` 里。
  只有需要 grounding judge 时才取消注释；这几个包版本敏感、构建慢，免费额度下
  容易超时。日常体验不需要它们。
- **页面能开但「运行校验」报 key 缺失** —— 你选了 LLM 轨（`critic_backend=llm`、
  `reviser_backend=llm`、`judge=ragas/trulens`、或 `retriever=openai`）。改回
  deterministic / local 即可离线跑，或按上面第 3 步配密钥。
- **休眠** —— Cloud 免费应用一段时间无人访问会休眠，下次访问自动唤醒（首次稍慢）。
