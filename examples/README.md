# 无行情 Demo

使用 Python 3.12，从仓库根目录运行：

```bash
python -m pip install -r examples/requirements.txt
python examples/demo.py
```

脚本构造10只**合成证券**、110个交易日和4次月度排名，直接调用项目原有的 `execution_v16.run_account`，展示前一日收盘生成信号、缓冲选股、次日开盘模拟成交、费用与账户净值。输出在 `examples/demo_output/`，包含逐日账户、订单事件和调仓目标。

这只是执行流程的人工数据示例，使用简化的三持仓缓冲组合；它不使用真实沪深300行情，也不复现 README 的37.71%历史模拟收益。要核对冻结研究配置，请阅读 `configs/v16_protocol.json` 及主研究脚本。

要在同一组合与合成行情上比较原引擎和执行控制，可分别运行 `python examples/demo.py --regime E0`、`--regime E1`、`--regime E2`；每次指定不同的 `--output` 路径以保留各自结果。E1 对最小卖出手数做合法目标投影，E2 继续清理未卖出的非目标持仓并检查实际权重。E1/E2 额外输出 `synthetic_exposures.csv`，对应目标、合法目标、成交后与日末权重。这是软件示例，不是历史策略收益。
