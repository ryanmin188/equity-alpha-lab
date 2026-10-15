# 发布目录（V1.7归因审计）

共63个文件，含V1.6策略归档与V1.7事后分析。V1.7不包含逐股预测、交易明细或原始行情。

```text
equity-alpha-lab/
├── .gitignore
├── FILE_STRUCTURE.md
├── README.md
├── configs/
│   ├── v16_protocol.json
│   └── v17_attribution_protocol.json
├── environment_versions.json
├── reports/
│   ├── ACCOUNT_AUDIT.md
│   ├── FACTOR_RESEARCH.md
│   ├── FINAL_VALIDATION_REPORT.md
│   ├── FORWARD_PAPER_PLAN.md
│   ├── INTERVIEW_GUIDE.md
│   ├── METHODOLOGY_REVIEW.md
│   ├── REPRODUCTION_CHECKS.json
│   ├── RESEARCH_REPORT.md
│   ├── RISK_RESEARCH.md
│   ├── V16_AUDIT.json
│   ├── V16_FINAL_RUN.json
│   ├── V16_FROZEN_SELECTION.json
│   ├── V16_PROTOCOL_NOTES.md
│   ├── V16_RELEASE_CHECKS.json
│   ├── V16_RESEARCH_REPORT.md
│   ├── V16_RISK_STATISTICS.json
│   ├── V17_RETURN_AND_MARKET_ATTRIBUTION.md
│   ├── figures/
│   │   ├── cost_sensitivity.png
│   │   ├── equity_vs_benchmark.png
│   │   ├── factor_correlation.png
│   │   ├── factor_deciles.png
│   │   ├── v16_all_candidates.png
│   │   ├── v16_execution_stress.png
│   │   ├── v16_final_equity.png
│   │   ├── v17_log_relative_decomposition.png
│   │   └── v17_rolling_market_risk.png
│   ├── research_dashboard.html
│   └── summary/
│       ├── cost_sensitivity.csv
│       ├── market_attribution.csv
│       ├── model_ic_summary.csv
│       ├── return_convention_anomalies.csv
│       ├── return_convention_audit.csv
│       ├── risk_attribution.csv
│       ├── v16_before_lot_fix_development.csv
│       ├── v16_before_lot_fix_validation.csv
│       ├── v16_development.csv
│       ├── v16_execution_stress.csv
│       ├── v16_final.csv
│       ├── v16_validation.csv
│       ├── v17_calendar_segments.csv
│       ├── v17_market_regressions.csv
│       └── v17_rolling_market_risk.csv
├── requirements.txt
├── scripts/
│   └── equity_alpha_v2/
│       ├── audit_v11_accounting.py
│       ├── audit_v17_attribution.py
│       ├── build_v14_report.py
│       ├── build_v16_report.py
│       ├── execution_v16.py
│       ├── finalize_v15_release.py
│       ├── paper_journal.py
│       ├── research_v12_factors.py
│       ├── research_v13_risk.py
│       ├── research_v16_improvements.py
│       └── run_research.py
└── tests/
    ├── test_execution_v16.py
    ├── test_paper_journal.py
    └── test_synthetic.py
```

公开的收益口径审计CSV不含逐日现金、持仓价值与成本金额；完整账户核对表保存在本地被Git忽略的输出目录。
