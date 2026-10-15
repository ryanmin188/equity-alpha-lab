# 发布目录（V1.6）

共52个文件。此清单仅列Git发布内容，不含本地行情、虚拟环境、缓存、预测或成交明细。

```text
equity-alpha-lab/
├── .gitignore
├── FILE_STRUCTURE.md
├── README.md
├── configs/
│   └── v16_protocol.json
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
│   ├── figures/
│   │   ├── cost_sensitivity.png
│   │   ├── equity_vs_benchmark.png
│   │   ├── factor_correlation.png
│   │   ├── factor_deciles.png
│   │   ├── v16_all_candidates.png
│   │   ├── v16_execution_stress.png
│   │   └── v16_final_equity.png
│   ├── research_dashboard.html
│   └── summary/
│       ├── cost_sensitivity.csv
│       ├── model_ic_summary.csv
│       ├── risk_attribution.csv
│       ├── v16_before_lot_fix_development.csv
│       ├── v16_before_lot_fix_validation.csv
│       ├── v16_development.csv
│       ├── v16_execution_stress.csv
│       ├── v16_final.csv
│       └── v16_validation.csv
├── requirements.txt
├── scripts/
│   └── equity_alpha_v2/
│       ├── audit_v11_accounting.py
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

源码、报告、图表和汇总保持各自目录层级。原V1.0—V1.5报告保留并标注归档；V1.6见V16_RESEARCH_REPORT.md。
