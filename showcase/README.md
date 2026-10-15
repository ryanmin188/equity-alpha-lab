# 交互式研究展示

在线页面：[Equity Alpha Lab 研究展示](https://equity-alpha-lab-research.brinycharm64.chatgpt.site)。

从仓库根目录运行 `python -m http.server 8000`，打开 `http://localhost:8000/showcase/`。页面使用原生 HTML/CSS/JS，不需要前端构建或外部图表库。直接双击 HTML 可能被浏览器限制读取本地 JSON，因此请通过本地 HTTP 服务预览。

三个视角分别呈现冻结账户的累计净值、回撤、年度和主要证券贡献；28项因子的 IC、ICIR、分组高低差与相关矩阵；以及样本时间切分、同样本 CH-4 回归和交易假设敏感性。页面仅读取 `data/research.json`，没有策略参数输入或历史重算功能。

数据文件从本地已保存的**聚合摘要**生成，不含原始行情、逐笔交易、完整持仓或外部因子工作簿。若本地有原始研究产出的 `reports/summary/`，可执行：

```bash
python scripts/equity_alpha_v2/build_showcase_data.py --summary-dir reports/summary
```

生成程序校验冻结观察区间、期末净值、484日同样本回归与28项因子数量，并在 JSON 中保留源摘要的 SHA-256。公开数据属于历史模拟和事后分析，不是新样本或实盘记录。
