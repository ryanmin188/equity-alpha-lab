# 公开仓库文件结构

```text
equity-alpha-lab/
├── README.md                         # 面试官入口：问题、方法、结果、局限
├── FILE_STRUCTURE.md
├── requirements.txt
├── environment_versions.json
├── .gitignore
├── configs/
│   ├── v16_protocol.json             # 最终测试前的策略与选择规则
│   ├── v17_attribution_protocol.json # 收益口径与市场归因
│   ├── v18_attribution_protocol.json # 逐股与CH-4事后归因口径
│   └── forward_observation_protocol.json # 未来信号登记边界
├── scripts/equity_alpha_v2/
│   ├── research_v16_improvements.py  # 开发、验证、冻结最终测试
│   ├── execution_v16.py              # 日线模拟账户
│   ├── attribute_v18_holdings.py     # V1.8逐股账本/集中度/技术风险/因子IC
│   ├── attribute_v18_ch4.py          # 外部A股CH-4日收益回归
│   ├── diagnose_v19_ch4.py           # 固定样本的相关、VIF和年度稳定性
│   ├── validate_v19_external.py      # 独立报价、分红和历史权重输入门槛
│   ├── record_forward_signal.py      # 未来Equal8分数只追加登记
│   ├── audit_v17_attribution.py      # V1.7收益时间戳与市场风险
│   ├── build_v16_report.py
│   ├── paper_journal.py
│   └── ...                            # V1.0—V1.5历史研究脚本
├── reports/
│   ├── V18_ATTRIBUTION_REPORT.md     # 当前主要经济解释
│   ├── EXTERNAL_VALIDATION_AND_FORWARD_PROTOCOL.md # 后续外部验证状态
│   ├── V17_RETURN_AND_MARKET_ATTRIBUTION.md
│   ├── V16_RESEARCH_REPORT.md
│   ├── INTERVIEW_GUIDE.md
│   ├── figures/                       # PNG，可在GitHub直接预览，含V1.8年度归因图
│   ├── summary/                       # 不含完整私有账本的聚合CSV
│   │   ├── v18_yearly_active_decomposition.csv
│   │   ├── v18_contribution_concentration.csv
│   │   ├── v18_top_security_drivers.csv
│   │   ├── v18_stale_mark_summary.csv
│   │   ├── v18_technical_style_exposures.csv
│   │   ├── v18_factor_significance.csv
│   │   ├── v18_ch4_regression.csv
│   │   ├── v18_corporate_action_case.csv
│   │   └── v19_*                   # 外部单日核对、因子诊断及缺口清单
│   ├── research_dashboard.html       # 早期探索性可下载仪表盘
│   └── ...                            # V1.0—V1.6原报告和其他汇总
└── tests/
    ├── test_v18_attribution.py
    ├── test_execution_v16.py
    ├── test_paper_journal.py
    ├── test_v19_gates.py
    └── test_synthetic.py
```

`scripts/equity_alpha_v2/outputs/`、Qlib行情、明汯原始因子工作簿、完整每日持仓/订单/现金账、预测缓存和环境密钥均不在公开仓库。`sector_attribution.csv`与含规模/价值的`style_exposures.csv`未生成，因为必需的历史时点数据尚未齐备；已完成的是单独标明的技术风格暴露。实际提交文件可用 `git ls-files` 核对。
