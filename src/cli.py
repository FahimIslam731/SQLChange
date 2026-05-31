#!/usr/bin/env python3
"""
    CLI script for the SQL Optimization Pipeline.

    Accepts a SQL query via stdin or argument, runs the full pipeline,
    and displays detailed model output for each step for debugging.

    Usage:
      python cli.py "SELECT * FROM orders JOIN customers ON ..."
      python cli.py --file query.sql
      python cli.py --interactive
      python cli.py --ddl "CREATE TABLE ..." "SELECT ..."

    Options:
      --provider    LLM provider (default: qwen)
      --model       Model name (default: qwen2.5-coder:7b)
      --iterations  Max optimization iterations (default: 3)
      --ddl         Optional DDL context string
      --ddl-file    Optional DDL context from file
      --file        Read SQL from file
      --interactive Interactive mode (prompt for SQL)
      --json        Output raw JSON instead of formatted
"""

import argparse
import json
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_parser():
    parser = argparse.ArgumentParser(
        description="SQL Optimization Pipeline — CLI debugger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.total > 100"
  python cli.py --file my_query.sql --ddl-file schema.sql --iterations 5
  python cli.py --interactive --provider qwen --model qwen2.5-coder:7b
        """,
    )
    parser.add_argument("sql", nargs="?", help="SQL query to optimize")
    parser.add_argument("--file", "-f", help="Read SQL query from file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--ddl", help="DDL context string")
    parser.add_argument("--ddl-file", help="Read DDL context from file")
    parser.add_argument("--provider", default="qwen", help="LLM provider (default: qwen)")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Model name (default: qwen2.5-coder:7b)")
    parser.add_argument("--api-key", help="API key (for Anthropic/OpenAI)")
    parser.add_argument("--iterations", "-n", type=int, default=3, help="Max iterations (default: 3)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    return parser


# ── Rich console formatting ──

def _has_rich():
    try:
        import rich
        return True
    except ImportError:
        return False

def _print_header(text, style="bold cyan"):
    if _has_rich():
        from rich.console import Console
        Console().print(f"\n{'─' * 60}", style="dim")
        Console().print(f"  {text}", style=style)
        Console().print(f"{'─' * 60}", style="dim")
    else:
        print(f"\n{'─' * 60}")
        print(f"  {text}")
        print(f"{'─' * 60}")

def _print_kv(key, value, indent=2):
    pad = " " * indent
    if _has_rich():
        from rich.console import Console
        Console().print(f"{pad}[bold]{key}:[/bold] {value}")
    else:
        print(f"{pad}{key}: {value}")

def _format_sql(sql):
    """Break a long SQL string into readable lines at major keywords."""
    if not sql or len(sql) < 80:
        return sql.strip()
    try:
        import sqlglot
        return sqlglot.transpile(sql, pretty=True)[0]
    except Exception:
        pass
    # Fallback: manual keyword-based line breaks
    import re
    keywords = r'\b(SELECT|FROM|WHERE|JOIN|LEFT JOIN|RIGHT JOIN|INNER JOIN|'
    keywords += r'CROSS JOIN|FULL JOIN|ON|AND|OR|ORDER BY|GROUP BY|HAVING|'
    keywords += r'LIMIT|UNION|EXCEPT|INTERSECT|WITH)\b'
    formatted = re.sub(keywords, r'\n\1', sql, flags=re.IGNORECASE)
    # Indent continuation lines
    lines = formatted.strip().split("\n")
    result = [lines[0].strip()]
    for line in lines[1:]:
        result.append("  " + line.strip())
    return "\n".join(result)

def _print_sql(sql, label="SQL"):
    # Pretty-format the SQL with line breaks at major keywords
    formatted = _format_sql(sql)
    if _has_rich():
        from rich.console import Console
        from rich.syntax import Syntax
        Console().print(f"  {label}:")
        syntax = Syntax(formatted, "sql", theme="monokai", line_numbers=False,
                        padding=1, word_wrap=True)
        Console().print(syntax)
    else:
        print(f"  {label}:")
        for line in formatted.split("\n"):
            print(f"    {line}")

def _print_json_block(data, label=""):
    if _has_rich():
        from rich.console import Console
        from rich.syntax import Syntax
        if label:
            Console().print(f"  {label}:")
        text = json.dumps(data, indent=2, default=str)
        syntax = Syntax(text, "json", theme="monokai", line_numbers=False, padding=1)
        Console().print(syntax)
    else:
        if label:
            print(f"  {label}:")
        print(f"    {json.dumps(data, indent=2, default=str)}")

def _print_status(text, style="green"):
    if _has_rich():
        from rich.console import Console
        Console().print(f"  ► {text}", style=style)
    else:
        print(f"  ► {text}")

def _print_label_badge(label, score=None, category=""):
    color_map = {
        "improves": "green", "low": "green", "equivalent": "green",
        "neutral": "yellow", "medium": "yellow", "narrower": "yellow",
        "degrades": "red", "high": "red", "broader": "yellow",
        "unknown": "dim", "different": "red",
    }
    color = color_map.get(label, "white")
    score_str = f" ({score}/10)" if score is not None else ""
    if _has_rich():
        from rich.console import Console
        Console().print(f"    {category}: [{color} bold]{label}{score_str}[/{color} bold]")
    else:
        print(f"    {category}: {label}{score_str}")


def display_step_logs(logs):
    """Display step logs in a readable format."""
    if not logs:
        return

    _print_header("STEP-BY-STEP DEBUG LOG", "bold magenta")

    for entry in logs:
        step = entry.get("step", "unknown")
        iteration = entry.get("iteration", 0)
        data = entry.get("data", {})

        step_label = f"[iter {iteration}] {step}" if iteration > 0 else step
        _print_header(f"Step: {step_label}", "bold yellow")

        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    _print_json_block(v, label=k)
                else:
                    _print_kv(k, v)
        else:
            _print_kv("data", data)


def display_results(result):
    """Display the pipeline results in a formatted way."""

    # ── Input ──
    _print_header("INPUT", "bold blue")
    _print_sql(result.get("original_sql", ""), "Original SQL")
    if result.get("ddl_context"):
        _print_kv("DDL", "Provided")
    _print_kv("Provider", result.get("provider", "?"))
    _print_kv("Model", result.get("model", "?"))

    # ── Parse & Extract ──
    _print_header("STAGE 1: Parse & Extract", "bold green")
    _print_kv("Inference method", result.get("inference_method", "?"))
    _print_kv("Tables found", result.get("table_names", []))
    _print_kv("Join keys", len(result.get("join_keys", [])))
    _print_kv("WHERE deps", len(result.get("where_details", [])))

    # ── ER Graph ──
    _print_header("STAGE 2: ER Graph", "bold green")
    er = result.get("er_graph", {})
    _print_kv("Total tables", er.get("total_tables", 0))
    _print_kv("Graph depth", er.get("graph_depth", 0))
    _print_kv("Cross-table risk", er.get("cross_table_risk", False))
    if er.get("table_importance"):
        for ti in er["table_importance"]:
            _print_kv(f"  {ti['table']}", ti['importance'], indent=4)

    # ── Execution Harness ──
    _print_header("STAGE 3: Execution Harness", "bold green")
    ev = result.get("execution_evidence", {})
    _print_kv("Output relation", ev.get("output_relation", "?"))
    _print_kv("Row counts", f"{ev.get('row_count_original', '?')} → {ev.get('row_count_modified', '?')}")
    _print_kv("Both succeeded", ev.get("both_succeeded", "?"))
    if ev.get("original_error"):
        _print_kv("Original error", ev["original_error"])
    if ev.get("modified_error"):
        _print_kv("Modified error", ev["modified_error"])
    perf = ev.get("performance", {})
    if isinstance(perf, dict) and "error" not in perf:
        for scale, data in perf.items():
            if isinstance(data, dict):
                orig = data.get("original_ms")
                mod = data.get("modified_ms")
                speedup = data.get("speedup")
                _print_kv(f"  {scale}", f"orig={orig:.3f}ms mod={mod:.3f}ms speedup={speedup:.2f}x" if all(v is not None for v in [orig, mod, speedup]) else str(data), indent=4)

    # ── Labels ──
    _print_header("STAGE 4: Labels", "bold green")
    pl = result.get("performance_label", {})
    rl = result.get("risk_label", {})
    sl = result.get("semantic_label", {})
    _print_label_badge(pl.get("label", "?"), pl.get("score"), "Performance")
    _print_kv("    Rationale", pl.get("rationale", ""), indent=0)
    _print_label_badge(rl.get("label", "?"), rl.get("score"), "Risk")
    _print_kv("    Rationale", rl.get("rationale", ""), indent=0)
    _print_label_badge(sl.get("label", "?"), category="Semantic")
    _print_kv("    Rationale", sl.get("rationale", ""), indent=0)

    # ── Recommendation ──
    _print_header("STAGE 5: Recommendation", "bold green")
    rec = result.get("recommendation", {})
    _print_kv("Action", rec.get("action", "?"))
    _print_kv("Score", f"{rec.get('score', '?')}/10")
    _print_kv("Confidence", rec.get("confidence", "?"))
    _print_kv("Valid SQL", rec.get("is_valid", "?"))
    _print_kv("SQLite valid", rec.get("is_sqlite_valid", "?"))
    if rec.get("optimizations_applied"):
        _print_kv("Optimizations", rec["optimizations_applied"])
    _print_kv("Rationale", rec.get("rationale", ""))
    rec_sql = result.get("recommended_sql", "")
    if rec_sql and rec_sql != result.get("original_sql"):
        preview = " ".join(rec_sql.split())[:120]
        _print_kv("Recommended SQL", preview + ("..." if len(rec_sql) > 120 else ""))
    else:
        _print_status("No optimization — keeping original query", "yellow")

    # ── Iteration History ──
    history = result.get("iteration_history", [])
    if len(history) > 1:
        _print_header("ITERATION HISTORY", "bold magenta")
        for h in history:
            _print_kv(
                f"Iteration {h['iteration']}",
                f"action={h.get('action')} score={h.get('score')} "
                f"perf={h.get('performance_score')} risk={h.get('risk_score')} "
                f"semantic={h.get('semantic_label')} valid={h.get('is_valid')}",
            )

    # ── Final Verdict ──
    best = result.get("best_result", {})
    _print_header("FINAL VERDICT", "bold cyan")
    improved = (
        rec_sql
        and rec_sql != result.get("original_sql")
        and rec.get("is_valid", False)
        and rec.get("action") != "keep_original"
    )
    if improved:
        _print_status(f"✅ Better query found! (score: {best.get('score', rec.get('score', '?'))}/10, iteration: {best.get('iteration', '?')})", "bold green")
        _print_sql(best.get("recommended_sql", rec_sql), "Best SQL")
    else:
        _print_status("⏸ Original query is already optimal or no safe improvement found.", "bold yellow")

    _print_kv("Total iterations", result.get("iteration", 0))


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Get SQL query
    sql = args.sql
    if args.file:
        with open(args.file) as f:
            sql = f.read().strip()
    elif args.interactive or (not sql and sys.stdin.isatty()):
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  SQL Optimization Pipeline — Interactive Mode            ║")
        print("║  Enter your SQL query, then press Enter twice to run.    ║")
        print("╚══════════════════════════════════════════════════════════╝")
        lines = []
        empty_count = 0
        while True:
            try:
                line = input("sql> " if not lines else "...> ")
                if line.strip() == "":
                    empty_count += 1
                    if empty_count >= 2 and lines:
                        break  # Two blank lines = submit
                    if lines:
                        lines.append(line)
                    continue
                else:
                    empty_count = 0
                lines.append(line)
                # Also accept ; as terminator
                if line.strip().endswith(";"):
                    break
            except EOFError:
                break
        sql = "\n".join(lines)
    elif not sql and not sys.stdin.isatty():
        sql = sys.stdin.read().strip()

    if not sql:
        parser.error("No SQL query provided. Use --interactive or pass SQL as argument.")

    # Get DDL context
    ddl = args.ddl
    if args.ddl_file:
        with open(args.ddl_file) as f:
            ddl = f.read().strip()

    # Run pipeline
    from pipeline import run_pipeline

    _print_header("RUNNING SQL OPTIMIZATION PIPELINE", "bold white on blue")
    _print_sql(sql, "Input SQL")
    _print_kv("Provider", args.provider)
    _print_kv("Model", args.model)
    _print_kv("Max iterations", args.iterations)
    if ddl:
        _print_kv("DDL", "Provided")

    start_time = time.time()

    try:
        result = run_pipeline(
            original_sql=sql,
            ddl_context=ddl,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            max_iterations=args.iterations,
        )
    except Exception as e:
        if _has_rich():
            from rich.console import Console
            Console().print(f"\n  [bold red]Pipeline failed:[/bold red] {e}")
        else:
            print(f"\n  Pipeline failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    if args.json:
        # Filter out non-serializable and large fields for clean JSON output
        output = {k: v for k, v in result.items() if k != "step_logs"}
        output["elapsed_seconds"] = round(elapsed, 2)
        print(json.dumps(output, indent=2, default=str))
    else:
        # Display formatted results
        display_results(result)

        _print_header(f"Done in {elapsed:.1f}s", "bold green")


if __name__ == "__main__":
    main()
