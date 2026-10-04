"""tests._smoke_tools —— 内置数据分析工具冒烟测试（无需 API Key）。

当前覆盖 data_inspector：用一份故意做脏的 CSV 验证
行数/列数、重复行、全空列、常量列、高缺失、ID/时间/目标列识别与业务类型推断。

运行（项目根目录）：
    .venv\\Scripts\\python.exe -m tests._smoke_tools
"""

from __future__ import annotations

import os
import tempfile

import pytest

import pandas as pd

from harness.tool_broker import ToolBroker
from packages.data_analysis.tools import register_builtin_tools


def _make_dirty_csv(dir_path: str) -> str:
    rows = [
        # order_id 主键, order_date 时间, amount 数值(有缺失), category 类别,
        # label 目标, remark 全空列, const 常量列
        [1, "2024-01-01", 100.0, "A", 1, None, "x"],
        [2, "2024-01-02", None, "B", 0, None, "x"],
        [3, "2024-01-03", 300.0, "A", 1, None, "x"],
        [4, "2024-01-04", 400.0, "B", 0, None, "x"],
        [5, "2024-01-05", None, "A", 1, None, "x"],
        [5, "2024-01-05", None, "A", 1, None, "x"],   # 与上一行完全重复
    ]
    df = pd.DataFrame(rows, columns=[
        "order_id", "order_date", "amount", "category", "label", "remark", "const",
    ])
    path = os.path.join(dir_path, "sales_demo.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_data_inspector() -> None:
    tmp = tempfile.mkdtemp(prefix="etl_tools_")
    _make_dirty_csv(tmp)

    broker = ToolBroker()
    register_builtin_tools(broker)

    ok, text, artifacts = broker.invoke(
        "data_inspector",
        {"file_path": "sales_demo.csv", "sample_rows": 3},
        {"workspace_dir": tmp},
    )
    assert ok, text
    insp = artifacts["inspection"]

    # 结构层
    q = insp["quality"]
    assert q["rows"] == 6 and q["cols"] == 7, q
    assert q["duplicate_rows"] >= 1, "应识别出完全重复行"
    assert q["missing_cells"] > 0

    schema = {c["name"]: c for c in insp["schema"]}
    # 业务类型推断
    assert schema["order_id"]["semantic_type"] == "id", schema["order_id"]
    assert schema["order_date"]["semantic_type"] == "datetime"
    assert schema["amount"]["semantic_type"] == "numeric"
    assert schema["category"]["semantic_type"] == "category"
    assert schema["label"]["looks_like_target"] is True
    assert schema["category"]["looks_like_target"] is False, "category 不应被误判为目标列"
    # 全空列 / 常量列
    assert schema["remark"]["missing_rate"] == 1.0
    assert schema["const"]["n_unique"] == 1

    risk_types = {r["type"] for r in insp["risks"]}
    assert "all_null" in risk_types, "应标记全空列"
    assert "constant" in risk_types, "应标记常量列"
    assert "duplicate_rows" in risk_types, "应标记重复行"
    assert "duplicate_key" in risk_types, "重复的 order_id 应被主键唯一性检测到"
    assert "possible_target" in risk_types

    # 样例三层：head / tail / random
    assert len(insp["sample"]["head"]) == 3
    assert len(insp["sample"]["tail"]) >= 1

    # 第二份：唯一主键 + 货币/百分比被存成文本
    clean = pd.DataFrame({
        "uid": [1, 2, 3, 4],
        "price": ["¥1,200.50", "$300.00", "12.5%", "45"],
        "city": ["北京", "上海", "广州", "深圳"],
    })
    clean.to_csv(os.path.join(tmp, "clean.csv"), index=False, encoding="utf-8-sig")
    ok2, _, art2 = broker.invoke(
        "data_inspector", {"file_path": "clean.csv"}, {"workspace_dir": tmp}
    )
    assert ok2
    sch2 = {c["name"]: c for c in art2["inspection"]["schema"]}
    assert sch2["uid"]["is_unique"] is True, "唯一主键 is_unique 应为 True"
    rt2 = {r["type"] for r in art2["inspection"]["risks"]}
    assert "possible_key" in rt2, "唯一主键应标记 possible_key"
    assert "numeric_as_text" in rt2, "货币/百分比文本应识别为数值存文本"
    assert sch2["price"]["text_profile"]["numeric_ratio"] >= 0.9

    print("1. data_inspector 体检 ok")
    print("-" * 60)
    print(text)
    print("-" * 60)


def test_inspector_missing_file() -> None:
    broker = ToolBroker()
    register_builtin_tools(broker)
    ok, text, _ = broker.invoke(
        "data_inspector", {"file_path": "no_such_file.csv"}, {}
    )
    assert not ok and "不存在" in text, f"缺文件应返回清晰错误，got: {text}"
    print("2. 文件不存在错误兜底 ok")


def test_data_cleaner() -> None:
    tmp = tempfile.mkdtemp(prefix="etl_clean_")
    _make_dirty_csv(tmp)
    ctx = {"workspace_dir": tmp, "reports_dir": tmp}
    broker = ToolBroker()
    register_builtin_tools(broker)

    # 默认规则：去重 + 删全空列 + 数值中位数填充 + 日期自动转换
    ok, text, art = broker.invoke(
        "data_cleaner", {"file_path": "sales_demo.csv", "output_name": "clean"}, ctx
    )
    assert ok, text
    c = art["cleaning"]
    before, after = c["before"], c["after"]
    assert before["rows"] == 6 and after["rows"] == 5, (before, after)  # 去掉 1 行重复
    assert after["cols"] == 6, after                                     # 删除全空列 remark
    assert after["missing_cells"] == 0, after                            # amount 中位数填充
    assert os.path.exists(c["output"]["abs_path"]), "干净数据集应落盘"
    assert os.path.exists(c["report_file"]["abs_path"]), "清洗报告应落盘"
    steps = {s["step"] for s in c["steps"]}
    assert {"drop_duplicates", "fillna", "drop_columns"} <= steps
    print("3. data_cleaner 默认清洗 ok（去重/删全空列/填充/日期转换）")

    # 显式规则：额外删除常量列
    ok2, _, art2 = broker.invoke(
        "data_cleaner",
        {"file_path": "sales_demo.csv", "output_name": "clean2",
         "rules": {"drop_constant_columns": True}},
        ctx,
    )
    assert ok2
    assert art2["cleaning"]["after"]["cols"] == 5, "remark + const 两列应都被删除"
    print("4. data_cleaner 显式规则（删常量列）ok")
    print("-" * 60)
    print(text)
    print("-" * 60)


def test_data_cleaner_advanced() -> None:
    tmp = tempfile.mkdtemp(prefix="etl_adv_")
    df = pd.DataFrame({
        "price": ["¥1,200.50", "$300.00", "45.00", "45.00", "10.00"],
        "grade": ["A ", "a", "B", "b", "a"],
        "seq": [1.0, None, 3.0, None, 5.0],
        "v": [1.0, 2.0, 3.0, 100.0, 2.0],
    })
    df.to_csv(os.path.join(tmp, "adv.csv"), index=False, encoding="utf-8-sig")
    broker = ToolBroker()
    register_builtin_tools(broker)

    ok, text, art = broker.invoke("data_cleaner", {
        "file_path": "adv.csv", "output_name": "adv_clean",
        "rules": {
            "clean_numeric_text": {"columns": "auto", "percent_to_fraction": True},
            "text_case": {"grade": "lower"},
            "category_map": {"grade": {"a": "A", "b": "B"}},
            "fillna": {"seq": "interpolate"},
            "outliers": {"v": {"method": "iqr", "action": "clip"}},
        },
    }, {"workspace_dir": tmp, "reports_dir": tmp})
    assert ok, text

    out = pd.read_csv(art["cleaning"]["output"]["abs_path"])
    assert abs(out["price"][0] - 1200.50) < 1e-6, out["price"].tolist()  # 货币+千分位
    assert set(out["grade"].unique()) <= {"A", "B"}, out["grade"].tolist()  # 大小写+同义词归一
    assert list(out["seq"]) == [1.0, 2.0, 3.0, 4.0, 5.0], list(out["seq"])  # 线性插值
    assert out["v"].max() < 100, "v=100 应被 IQR 截断"
    steps = {s["step"] for s in art["cleaning"]["steps"]}
    assert {"clean_numeric_text", "fillna", "outliers"} <= steps
    assert art["cleaning"]["recommendations"], "应给出清洗建议"
    print("5. data_cleaner 高级规则 ok（货币文本转数值/同义词归一/插值/IQR截断/建议）")


def test_eda() -> None:
    import numpy as np
    tmp = tempfile.mkdtemp(prefix="etl_eda_")
    rng = np.random.default_rng(42)
    n = 40
    x = np.arange(1, n + 1, dtype=float)
    y = 2 * x + rng.normal(0, 1.0, n)
    category = rng.choice(["A", "B"], n, p=[0.6, 0.4])
    amount = np.where(category == "A", rng.normal(100, 8, n), rng.normal(50, 8, n))
    label = (amount > 75).astype(int)
    pd.DataFrame({
        "id": np.arange(1, n + 1),
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "x": x, "y": y, "category": category,
        "amount": np.round(amount, 2), "label": label,
    }).to_csv(os.path.join(tmp, "eda_demo.csv"), index=False, encoding="utf-8-sig")

    broker = ToolBroker()
    register_builtin_tools(broker)
    ok, text, art = broker.invoke(
        "eda", {"file_path": "eda_demo.csv", "target": "label",
                "group_by": "category", "value_col": "amount"},
        {"workspace_dir": tmp},
    )
    assert ok, text
    eda, o = art["eda"], art["eda"]["overview"]
    assert o["rows"] == n
    assert "x" in o["numeric_cols"] and "id" not in o["numeric_cols"], "id 不应进数值列"
    assert o["target"] == "label"

    # 单变量
    assert "mean" in eda["univariate"]["numeric"]["x"]
    assert len(eda["univariate"]["numeric"]["x"]["distribution"]) == 10
    assert eda["univariate"]["categorical"]["category"]["n_unique"] == 2
    assert "entropy" in eda["univariate"]["categorical"]["category"]
    # 时间
    tp = eda["univariate"]["datetime"]["date"]
    assert tp["granularity_inferred"] == "day" and tp["missing_time_points"] == 0
    # 双变量：相关 + 显著性
    assert o["statistical_tests_available"] is True, "venv 应已装 scipy"
    xy = [p for p in eda["bivariate"]["correlations"] if {p["a"], p["b"]} == {"x", "y"}]
    assert xy and xy[0]["pearson"] > 0.9 and xy[0]["p_value"] < 0.05, xy
    # 类别×数值 ANOVA
    assert "category ~ amount" in eda["bivariate"]["anova"]
    assert eda["bivariate"]["anova"]["category ~ amount"]["p_value"] < 0.05
    # 类别×类别卡方
    assert any(k.startswith("category ~") or k.endswith("~ category") for k in eda["bivariate"]["chi_square"])
    # 目标（0/1 label 应识别为分类目标）
    assert eda["target_relation"]["type"] == "categorical"
    # 业务汇总
    assert eda["business_summary"] and len(eda["business_summary"]["top"]) == 2
    # 自动发现
    assert len(eda["findings"]) >= 2
    print("6. eda 全角度统计 ok（描述/分布/相关+显著性/ANOVA/卡方/时间/目标/业务汇总/自动发现）")
    print("-" * 60)
    print(text)
    print("-" * 60)


def test_sql_query() -> None:
    import sqlite3
    tmp = tempfile.mkdtemp(prefix="etl_sql_")
    db = os.path.join(tmp, "shop.db")
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE sales(id INTEGER PRIMARY KEY, cat TEXT, amount REAL);\n"
        "INSERT INTO sales VALUES (1,'A',10),(2,'A',20),(3,'B',30),"
        "(4,'B',40),(5,'A',50),(6,'B',60);\n"
    )
    conn.commit()
    conn.close()

    broker = ToolBroker()
    register_builtin_tools(broker)
    ctx = {"workspace_dir": tmp}

    # 聚合查询
    ok, text, art = broker.invoke("sql_query", {
        "db_path": "shop.db",
        "sql": "SELECT cat, SUM(amount) AS total FROM sales GROUP BY cat ORDER BY total DESC",
    }, ctx)
    assert ok, text
    res = art["sql_result"]
    assert res["columns"] == ["cat", "total"] and res["row_count"] == 2
    totals = {row[0]: row[1] for row in res["rows"]}
    assert totals == {"A": 80.0, "B": 130.0}, totals

    # 自动 LIMIT
    ok, _, art = broker.invoke(
        "sql_query", {"db_path": "shop.db", "sql": "SELECT id FROM sales", "limit": 3}, ctx)
    assert ok and art["sql_result"]["row_count"] == 3
    assert art["sql_result"]["auto_limit"] == 3

    # 参数化查询
    ok, _, art = broker.invoke("sql_query", {
        "db_path": "shop.db", "sql": "SELECT COUNT(*) AS n FROM sales WHERE cat = ?",
        "params": ["B"],
    }, ctx)
    assert ok and art["sql_result"]["rows"][0][0] == 3

    # 写操作 / DDL / 多语句一律拒绝
    for bad in [
        "INSERT INTO sales VALUES(7,'C',1)",
        "UPDATE sales SET amount=0",
        "DELETE FROM sales",
        "DROP TABLE sales",
        "SELECT 1; DROP TABLE sales",
        "SELECT 1; SELECT 2",
    ]:
        ok, _, _ = broker.invoke("sql_query", {"db_path": "shop.db", "sql": bad}, ctx)
        assert not ok, f"应拒绝非法 SQL: {bad}"

    # 数据库不存在
    ok, _, _ = broker.invoke("sql_query", {"db_path": "nope.db", "sql": "SELECT 1"}, ctx)
    assert not ok
    print("7. sql_query 只读查询 ok（聚合/自动LIMIT/参数化/写操作DDL多语句拦截）")

    # 释放 db_path 懒建的进程级默认 manager 持有的 sqlite 连接
    import packages.data_analysis.tools.sql_query as _sq
    if _sq._default_manager is not None:
        _sq._default_manager.close()
        _sq._default_manager = None


def test_chart_generator() -> None:
    tmp = tempfile.mkdtemp(prefix="etl_chart_")
    pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=12, freq="ME"),
        "category": ["A", "B"] * 6,
        "amount": list(range(10, 130, 10)),
        "cost": list(range(5, 125, 10)),
    }).to_csv(os.path.join(tmp, "chart_demo.csv"), index=False, encoding="utf-8-sig")

    broker = ToolBroker()
    register_builtin_tools(broker)
    ctx = {"workspace_dir": tmp, "reports_dir": tmp}

    cases = [
        ("bar", {"x": "category", "y": "amount", "agg": "sum"}),
        ("line", {"x": "date", "y": "amount"}),
        ("hist", {"y": "amount"}),
        ("pie", {"x": "category"}),
        ("heatmap", {}),
        ("auto", {"x": "category", "y": "cost"}),  # 类别×数值 -> bar
    ]
    for ctype, extra in cases:
        ok, text, art = broker.invoke("chart_generator",
                                      {"file_path": "chart_demo.csv", "chart_type": ctype, **extra}, ctx)
        assert ok, f"{ctype}: {text}"
        ch = art["chart"]
        assert os.path.exists(ch["abs_path"]) and ch["size_bytes"] > 0, f"{ctype} PNG 未落盘"
        if ctype == "auto":
            assert ch["chart_type"] == "bar", ch["chart_type"]
        else:
            assert ch["chart_type"] == ctype
    print("8. chart_generator 出图 ok（bar/line/hist/pie/heatmap/auto 选型，PNG 落盘）")


@pytest.mark.needs_sandbox
def test_code_executor() -> None:
    """沙箱代码执行：经 OpenSandbox 隔离容器执行，含双层文件平面往返。

    依赖沙箱基础设施（OpenSandbox 服务端 + Docker）。pytest 下默认不可达时 skip，
    加 ``--require-sandbox`` 严格 fail（与 ``needs_sandbox`` marker 约定一致，见 pytest.ini）；
    直接 ``python -m tests._smoke_tools`` 运行时由函数内的 ``SandboxClient().available()``
    探测 assert fail。沙箱不可用时必须 fail closed、不降级为宿主进程的安全断言由
    同文件 ``test_sandbox_fail_closed`` 覆盖（无 marker，永远运行）。
    """
    from harness.sandbox.client import SandboxClient
    from packages.data_analysis.tools.code_executor import handle as code_handler

    ready, reason = SandboxClient().available()
    assert ready, f"沙箱基础设施不可用：{reason}"

    tmp = tempfile.mkdtemp(prefix="etl_code_")
    broker = ToolBroker()
    register_builtin_tools(broker)
    ctx = {"workspace_dir": tmp, "reports_dir": tmp, "role": "admin"}
    # 节点外直接走 Broker：code_executor 需人工审批，显式带上与工具匹配的审批
    # 凭证（模拟节点 interrupt 审批通过后注入），否则 Broker 第 3.5 步 fail closed。
    approved_ctx = dict(ctx, approval={
        "id": "apr_smoke_code_executor", "approved": True, "tool": "code_executor",
    })

    # 经 broker 端到端（pandas 计算 + 中文输出）
    ok, text, art = broker.invoke("code_executor", {"code": """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3, 4], 'g': ['x', 'x', 'y', 'y']})
print('分组结果：', df.groupby('g')['a'].sum().to_dict())
"""}, approved_ctx)
    assert ok, text
    assert "分组结果" in art["execution"]["stdout"]
    assert art["execution"]["returncode"] == 0
    assert art["execution"]["sandbox"] == "opensandbox"

    # 以下直接调 handler，避开 code_executor 5 次/分钟的限流（限流属 broker 层）
    ok, _, art = code_handler({"code": "print(123 * 456)"}, ctx)
    assert ok and "56088" in art["execution"]["stdout"]

    # 隔离：环境变量指向**沙箱内**路径，宿主路径不得外泄
    ok, text, art = code_handler({
        "code": "import os; w = os.environ['ETL_WORKSPACE_DIR']; print(w)"
    }, ctx)
    assert ok, text
    assert "/home/sandbox/in" in art["execution"]["stdout"], art
    assert tmp not in art["execution"]["stdout"], "宿主路径泄漏进了沙箱"

    # 文件平面 主机 -> 沙箱：输入同步
    with open(os.path.join(tmp, "input.csv"), "w", encoding="utf-8") as fh:
        fh.write("a,b\n1,2\n3,4\n")
    ok, text, art = code_handler({"code": """
import os, pandas as pd
df = pd.read_csv(os.path.join(os.environ['ETL_WORKSPACE_DIR'], 'input.csv'))
print('行数', len(df), '合计', int(df['a'].sum()))
"""}, ctx)
    assert ok, text
    assert "行数 2 合计 4" in art["execution"]["stdout"], art

    # 文件平面 沙箱 -> 主机：产物按类型回灌
    ok, text, art = code_handler({"code": """
import os, pandas as pd
out = os.environ['ETL_REPORTS_DIR']
pd.DataFrame({'x': [1, 2]}).to_csv(os.path.join(out, 'result.csv'), index=False)
with open(os.path.join(out, 'note.md'), 'w', encoding='utf-8') as f:
    f.write('# 来自沙箱的产物')
print('已写产物')
"""}, ctx)
    assert ok, text
    assert {w["name"] for w in art["execution"]["artifacts_written"]} == {"result.csv", "note.md"}, art
    assert os.path.isfile(os.path.join(tmp, "result.csv"))
    assert os.path.isfile(os.path.join(tmp, "note.md"))

    # 运行期错误应被捕获（ok=False，stderr 有 traceback）
    ok, text, art = code_handler({"code": "raise ValueError('boom')"}, ctx)
    assert not ok and "boom" in art["execution"]["stderr"], text

    # 安全守卫：危险导入 / 调用一律拒绝
    for bad in ["import subprocess", "import socket",
                "import os; os.system('echo hi')", "eval('1+1')", "__import__('os')"]:
        ok, text, _ = code_handler({"code": bad}, ctx)
        assert not ok and "守卫" in text, f"应拦截: {bad}"

    # 超时：由沙箱服务端强制终止
    ok, _, art = code_handler({"code": "import time; time.sleep(60)", "timeout_seconds": 5}, ctx)
    assert not ok and art["execution"]["timed_out"] is True, art

    print("9. code_executor 沙箱执行 ok（pandas/中文/文件平面往返/错误捕获/安全守卫/超时）")


def test_sandbox_fail_closed() -> None:
    """沙箱不可用时必须拒绝执行，绝不在宿主上退化为进程执行。

    这是安全边界本身，因此独立成项：指向一个必然连不上的端口模拟服务端故障，
    断言"拒绝执行"而不是"降级跑通"。
    """
    from harness.config import SandboxSettings
    from harness.sandbox import SandboxClient, SandboxExecutor
    from packages.data_analysis.tools.code_executor import TOOL_DEF

    # 指向一个必然拒绝连接的端口，模拟服务端故障
    dead = SandboxSettings(server_url="http://127.0.0.1:9", api_key="irrelevant")
    client = SandboxClient(dead)

    ready, _ = client.available()
    assert not ready, "指向死端口时不应判定为可用"

    executor = SandboxExecutor(client=client)
    ok, text, _ = executor.execute(TOOL_DEF, {"code": "print('不该被执行')"}, {}, None)
    assert not ok, f"沙箱不可用时必须拒绝执行，实际返回 ok=True：{text}"
    assert "沙箱不可用" in text, text
    assert "不该被执行" not in text, "拒绝信息里不得出现执行结果"

    print("10. 沙箱不可用时 fail closed ok（拒绝执行，未降级为宿主进程）")


def _main() -> None:
    test_data_inspector()
    test_inspector_missing_file()
    test_data_cleaner()
    test_data_cleaner_advanced()
    test_eda()
    test_sql_query()
    test_chart_generator()
    test_code_executor()
    test_sandbox_fail_closed()
    print("=== 数据分析工具冒烟测试通过 ===")


if __name__ == "__main__":
    _main()
