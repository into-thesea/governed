"""tools.data_cleaner —— 数据清洗工具（工业级）。

根据体检结果对数据做可追溯清洗，产出"干净数据集 + 清洗报告"。每一步记录动作与影响，
默认策略保守且可被显式 rules 覆盖。

处理角度（项目计划 7.2 全量）：
- 重复值：整行 / 按主键子集去重；
- 列处理：删全空/常量列、显式删除/保留、重命名；
- 文本规范：空串转缺失、strip、空白折叠、全半角 NFKC、大小写、类别同义词归一；
- 数值文本：货币符号/千分位/百分比清洗并转数值；
- 类型修正：数值/日期/类别/字符串转换，时间列自动识别；
- 缺失填充：均值/中位数/众数/前后向/线性插值/常量，按列缺失率给策略建议；
- 异常值：IQR / Z-score 边界，截断 clip / Winsorize 缩尾 / 标记 / 删除。
产物写 /workspace，Markdown 清洗报告写 /reports。
"""

from __future__ import annotations

import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from harness.models import ToolDef
from packages.data_analysis.tools.common import (
    ToolDataError,
    infer_semantic_type,
    load_table,
    numeric_text_profile,
    parse_numeric_text,
    save_dataframe,
    save_text_report,
    to_native,
    truncate,
)

TOOL_DEF = ToolDef(
    name="data_cleaner",
    description=(
        "数据清洗工具，在 data_inspector 体检之后、eda 之前调用。可执行：去重（整行/按主键）、"
        "删除全空列与常量列、列重命名/选择、文本规范化（去空格/折叠空白/全半角统一/大小写/"
        "类别同义词归一/空字符串转缺失）、把含货币符号或千分位或百分号的文本列转成数值、"
        "类型转换、缺失值填充（均值/中位数/众数/前后向/线性插值/常量）、IQR 或 Z-score 异常值"
        "处理（截断/Winsorize 缩尾/标记/删除）。输出清洗后 CSV 与可追溯清洗报告（每步影响行数、"
        "前后缺失率对比），并给出后续清洗建议。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "待清洗数据文件路径"},
            "output_name": {"type": "string", "description": "输出文件名（可不含扩展名），默认 原名_cleaned"},
            "rules": {
                "type": "object",
                "description": "清洗规则，全部可选，缺省采用保守默认",
                "properties": {
                    "drop_duplicates": {"description": "true 或 {subset:[列],keep:first|last}"},
                    "drop_all_null_columns": {"type": "boolean"},
                    "drop_constant_columns": {"type": "boolean"},
                    "drop_columns": {"type": "array", "items": {"type": "string"}},
                    "keep_columns": {"type": "array", "items": {"type": "string"}},
                    "rename": {"type": "object"},
                    "nfkc_normalize": {"type": "boolean", "description": "全角字母数字转半角，默认 true"},
                    "empty_string_to_na": {"type": "boolean"},
                    "strip_text": {"type": "boolean"},
                    "normalize_whitespace": {"type": "boolean"},
                    "text_case": {"type": "object", "description": "列名 -> lower|upper"},
                    "category_map": {"type": "object", "description": "列名 -> {同义词: 标准名}"},
                    "clean_numeric_text": {
                        "type": "object",
                        "description": "货币/千分位/百分比文本转数值：{columns:'auto'或[列], percent_to_fraction:true}",
                    },
                    "fillna": {"type": "object", "description": "列名 -> mean|median|mode|ffill|bfill|interpolate|常量"},
                    "drop_na_rows": {"description": "true 或 {subset:[列]}"},
                    "astype": {"type": "object", "description": "列名 -> int|float|str|datetime|category"},
                    "outliers": {
                        "type": "object",
                        "description": "列名 -> {method:iqr|zscore, k:1.5或3, action:clip|winsorize|mark|drop, quantile:0.01}",
                    },
                },
            },
        },
        "required": ["file_path"],
    },
    required_role="analyst",
    rate_limit_per_min=20,
    requires_approval=False,
    run_in_sandbox=False,
)

DEFAULT_RULES: dict[str, Any] = {
    "drop_duplicates": True,
    "drop_all_null_columns": True,
    "drop_constant_columns": False,
    "nfkc_normalize": True,
    "empty_string_to_na": True,
    "strip_text": True,
    "normalize_whitespace": True,
}


def _is_text(series: pd.Series) -> bool:
    return pdt.is_object_dtype(series) or pdt.is_string_dtype(series) or str(series.dtype) == "category"


def _fill_one(df: pd.DataFrame, col: str, strategy: str, report: list) -> None:
    if col not in df.columns:
        report.append({"step": "fillna", "detail": f"列 {col} 不存在，跳过"})
        return
    s = df[col]
    before_na = int(s.isna().sum())
    if before_na == 0:
        return
    if strategy == "mean":
        df[col] = s.fillna(s.mean())
    elif strategy == "median":
        df[col] = s.fillna(s.median())
    elif strategy == "mode":
        modes = s.mode()
        if len(modes):
            df[col] = s.fillna(modes.iloc[0])
    elif strategy == "ffill":
        df[col] = s.ffill().bfill()
    elif strategy == "bfill":
        df[col] = s.bfill().ffill()
    elif strategy == "interpolate":
        numeric = pd.to_numeric(s, errors="coerce")
        df[col] = numeric.interpolate(method="linear", limit_direction="both")
    else:
        df[col] = s.fillna(strategy)  # 常量
    after_na = int(df[col].isna().sum())
    report.append({"step": "fillna", "column": col, "strategy": strategy,
                   "filled": before_na - after_na, "remaining_na": after_na})


def _recommend(df: pd.DataFrame) -> list[str]:
    """基于原始数据给出清洗策略建议（即使默认已自动执行安全项，也保留建议供追溯）。"""
    rec: list[str] = []
    for c in df.columns:
        s = df[c]
        rate = float(s.isna().mean()) if len(s) else 0.0
        if rate == 1 and len(s):
            rec.append(f"列 {c} 全空，建议 drop_columns")
        elif rate > 0.8:
            rec.append(f"列 {c} 缺失 {rate:.0%}，建议评估是否删除")
        elif rate > 0:
            if pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s):
                rec.append(f"数值列 {c} 缺失 {rate:.0%}，建议 fillna=median 或 interpolate")
            elif infer_semantic_type(s) in ("category", "boolean"):
                rec.append(f"类别列 {c} 缺失 {rate:.0%}，建议 fillna=mode")
        if int(s.nunique(dropna=True)) <= 1 and int(s.notna().sum()) > 0:
            rec.append(f"列 {c} 为常量列，建议 drop_constant_columns")
        if _is_text(s):
            prof = numeric_text_profile(s)
            if prof["numeric_ratio"] >= 0.9:
                rec.append(f"列 {c} 的数字以文本存储，建议 clean_numeric_text")
    return rec


def handle(args: dict, context: dict):
    file_path = args.get("file_path")
    output_name = args.get("output_name")
    rules = dict(DEFAULT_RULES)
    rules.update(args.get("rules") or {})

    try:
        df, meta = load_table(file_path, context=context)
    except ToolDataError as e:
        return False, f"数据清洗失败：{e}", {}
    except Exception as e:  # noqa: BLE001
        return False, f"数据清洗失败：{type(e).__name__}: {e}", {}

    recommendations = _recommend(df)
    report: list[dict[str, Any]] = []
    before = {"rows": int(len(df)), "cols": int(df.shape[1]),
              "missing_cells": int(df.isna().sum().sum())}

    # 1) 重命名
    if rules.get("rename"):
        df = df.rename(columns=rules["rename"])
        report.append({"step": "rename", "detail": rules["rename"]})

    # 2) 文本规范化（NFKC → 空串转缺失 → strip/折叠 → 大小写 → 同义词归一）
    text_cols = [c for c in df.columns if _is_text(df[c])]
    case_map = rules.get("text_case") or {}
    category_map = rules.get("category_map") or {}
    for c in text_cols:
        if rules.get("nfkc_normalize", True):
            df[c] = df[c].map(lambda x: unicodedata.normalize("NFKC", x) if isinstance(x, str) else x)
        if rules.get("empty_string_to_na", True):
            df[c] = df[c].replace(r"^\s*$", np.nan, regex=True)
        if rules.get("strip_text", True):
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
        if rules.get("normalize_whitespace", True):
            df[c] = df[c].map(lambda x: " ".join(x.split()) if isinstance(x, str) else x)
        if case_map.get(c) == "lower":
            df[c] = df[c].map(lambda x: x.lower() if isinstance(x, str) else x)
        elif case_map.get(c) == "upper":
            df[c] = df[c].map(lambda x: x.upper() if isinstance(x, str) else x)
        if c in category_map and isinstance(category_map[c], dict):
            mapping = category_map[c]
            df[c] = df[c].map(lambda x: mapping.get(x, x) if isinstance(x, str) else x)
    if text_cols:
        report.append({"step": "text_normalize", "columns": text_cols})

    # 3) 货币/千分位/百分比文本 → 数值（须在 astype 之前）
    cnt_cfg = rules.get("clean_numeric_text")
    if cnt_cfg:
        pct = cnt_cfg.get("percent_to_fraction", True) if isinstance(cnt_cfg, dict) else True
        cols = cnt_cfg.get("columns", "auto") if isinstance(cnt_cfg, dict) else "auto"
        if cols in ("auto", None):
            cols = [c for c in df.columns if _is_text(df[c])
                    and numeric_text_profile(df[c])["numeric_ratio"] >= 0.9]
        for c in cols:
            if c not in df.columns:
                continue
            before_na = int(df[c].isna().sum())
            df[c] = parse_numeric_text(df[c], percent_to_fraction=pct)
            after_na = int(df[c].isna().sum())
            report.append({"step": "clean_numeric_text", "column": c,
                           "percent_to_fraction": pct,
                           "converted": int(df[c].notna().sum()),
                           "unparseable_to_na": max(after_na - before_na, 0)})

    # 4) 显式类型转换
    for col, target in (rules.get("astype") or {}).items():
        if col not in df.columns:
            continue
        try:
            if target == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif target == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().all():
                    df[col] = df[col].astype("int64")
            elif target == "category":
                df[col] = df[col].astype("category")
            elif target == "str":
                df[col] = df[col].map(lambda x: x if pd.isna(x) else str(x))
            report.append({"step": "astype", "column": col, "to": target})
        except Exception as e:  # noqa: BLE001
            report.append({"step": "astype", "column": col, "error": str(e)})

    # 名字像时间的文本列，高成功率时自动转 datetime
    if not rules.get("astype"):
        for c in list(df.columns):
            if _is_text(df[c]) and infer_semantic_type(df[c]) == "datetime":
                converted = pd.to_datetime(df[c], errors="coerce")
                if converted.notna().sum() >= 0.9 * max(df[c].notna().sum(), 1):
                    df[c] = converted
                    report.append({"step": "auto_datetime", "column": c})

    # 5) 去重
    dup_cfg = rules.get("drop_duplicates")
    if dup_cfg:
        before_n = len(df)
        if isinstance(dup_cfg, dict):
            df = df.drop_duplicates(
                subset=dup_cfg.get("subset"), keep=dup_cfg.get("keep", "first")
            ).reset_index(drop=True)
        else:
            df = df.drop_duplicates().reset_index(drop=True)
        report.append({"step": "drop_duplicates", "removed_rows": before_n - len(df)})

    # 6) 删全空 / 常量 / 显式列，保留列
    drop_cols: list[str] = []
    if rules.get("drop_all_null_columns", True):
        drop_cols += [c for c in df.columns if df[c].isna().all()]
    if rules.get("drop_constant_columns", False):
        drop_cols += [c for c in df.columns if c not in drop_cols and df[c].nunique(dropna=True) <= 1]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        report.append({"step": "drop_columns", "columns": drop_cols, "reason": "all_null/constant"})
    if rules.get("drop_columns"):
        extra = [c for c in rules["drop_columns"] if c in df.columns]
        if extra:
            df = df.drop(columns=extra)
            report.append({"step": "drop_columns", "columns": extra, "reason": "explicit"})
    if rules.get("keep_columns"):
        keep = [c for c in rules["keep_columns"] if c in df.columns]
        df = df[keep]
        report.append({"step": "keep_columns", "columns": keep})

    # 7) 缺失填充（显式优先；缺省对数值中位数、低基数类别众数）
    explicit_fill = rules.get("fillna") or {}
    for col, strategy in explicit_fill.items():
        _fill_one(df, col, strategy, report)
    if not rules.get("fillna"):
        for c in df.columns:
            if df[c].isna().any():
                sem = infer_semantic_type(df[c])
                if pdt.is_numeric_dtype(df[c]) and not pdt.is_bool_dtype(df[c]):
                    _fill_one(df, c, "median", report)
                elif sem in ("category", "boolean"):
                    _fill_one(df, c, "mode", report)

    # 8) 删缺失行
    na_cfg = rules.get("drop_na_rows")
    if na_cfg:
        before_n = len(df)
        subset = na_cfg.get("subset") if isinstance(na_cfg, dict) else None
        df = df.dropna(subset=subset).reset_index(drop=True)
        report.append({"step": "drop_na_rows", "removed_rows": before_n - len(df)})

    # 9) 异常值：IQR / Z-score × clip / winsorize / mark / drop
    for col, cfg in (rules.get("outliers") or {}).items():
        if col not in df.columns or not pdt.is_numeric_dtype(df[col]):
            continue
        # 容错：LLM 可能传 "iqr" 字符串而非 {"method": "iqr"} dict
        if isinstance(cfg, str):
            cfg = {"method": cfg}
        action = cfg.get("action", "clip")
        if action == "winsorize":
            q = float(cfg.get("quantile", 0.01))
            lo, hi = df[col].quantile(q), df[col].quantile(1 - q)
            method = f"winsorize@{q}"
        elif cfg.get("method", "iqr") == "zscore":
            k = float(cfg.get("k", 3))
            mu, sd = df[col].mean(), df[col].std(ddof=0)
            lo, hi, method = mu - k * sd, mu + k * sd, f"zscore(k={k})"
        else:
            k = float(cfg.get("k", 1.5))
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi, method = q1 - k * iqr, q3 + k * iqr, f"iqr(k={k})"
        mask = (df[col] < lo) | (df[col] > hi)
        n_out = int(mask.sum())
        if action in ("clip", "winsorize"):
            df[col] = df[col].clip(lo, hi)
        elif action == "drop":
            df = df[~mask].reset_index(drop=True)
        elif action == "mark":
            df[f"{col}_is_outlier"] = mask
        report.append({"step": "outliers", "column": col, "method": method,
                       "action": action, "outliers": n_out,
                       "bounds": [to_native(lo), to_native(hi)]})

    after = {"rows": int(len(df)), "cols": int(df.shape[1]),
             "missing_cells": int(df.isna().sum().sum())}

    # 10) 落盘 + 报告
    stem = meta.get("resolved_path", "data").replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    out_name = output_name or f"{stem}_cleaned"
    out_file = save_dataframe(df, out_name, context)
    md = _build_report_md(out_name, before, after, report, recommendations)
    report_file = save_text_report(f"{out_name}_清洗报告", md, context)

    cleaning = {
        "output": out_file, "report_file": report_file,
        "rules_used": to_native(rules), "before": before, "after": after,
        "steps": report, "recommendations": recommendations,
    }

    lines = [
        f"清洗完成：{before['rows']}→{after['rows']} 行，{before['cols']}→{after['cols']} 列，"
        f"缺失单元格 {before['missing_cells']}→{after['missing_cells']}。",
        f"干净数据集：{out_file['vfs_path']}；清洗报告：{report_file['vfs_path']}。",
        "主要操作：" + ("；".join(_step_brief(r) for r in report[:8]) or "（数据已较干净，无需显著处理）"),
    ]
    if recommendations:
        lines.append("后续建议：" + "；".join(recommendations[:4]))
    return True, truncate("\n".join(lines), 1400), {"cleaning": to_native(cleaning)}


def _step_brief(r: dict) -> str:
    step = r.get("step")
    if step == "fillna":
        return f"{r.get('column')} 用{r.get('strategy')}填 {r.get('filled')} 个"
    if step == "drop_duplicates":
        return f"去重 {r.get('removed_rows')} 行"
    if step == "drop_na_rows":
        return f"删缺失行 {r.get('removed_rows')} 行"
    if step == "drop_columns":
        return f"删列 {r.get('columns')}"
    if step == "clean_numeric_text":
        return f"{r.get('column')} 文本转数值（{r.get('converted')}）"
    if step == "outliers":
        return f"{r.get('column')} 异常 {r.get('outliers')} 个({r.get('action')})"
    if step in ("astype", "auto_datetime"):
        return f"{r.get('column')}→{r.get('to', 'datetime')}"
    return step


def _build_report_md(name, before, after, report, recommendations) -> str:
    lines = [
        f"# 数据清洗报告：{name}", "",
        "## 总览", "",
        "| 指标 | 清洗前 | 清洗后 |", "|---|---|---|",
        f"| 行数 | {before['rows']} | {after['rows']} |",
        f"| 列数 | {before['cols']} | {after['cols']} |",
        f"| 缺失单元格 | {before['missing_cells']} | {after['missing_cells']} |",
        "", "## 操作明细（按执行顺序）", "",
    ]
    for i, r in enumerate(report, 1):
        lines.append(f"{i}. `{r.get('step')}` " + ", ".join(
            f"{k}={v}" for k, v in r.items() if k != "step"))
    if not report:
        lines.append("（未执行任何变更）")
    lines += ["", "## 清洗策略建议", ""]
    lines += [f"- {r}" for r in recommendations] or ["- （无）"]
    return "\n".join(lines)


__all__ = ["TOOL_DEF", "handle"]
