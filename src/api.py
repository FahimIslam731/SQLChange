#!/usr/bin/env python3
"""
    Minimal HTTP API for the SQL Optimization Pipeline.

    Endpoints:
      POST /optimize  — Submit a SQL query for optimization
      GET  /health     — Health check

    Request body (JSON):
      {
        "sql": "SELECT ...",
        "ddl": "CREATE TABLE ... (optional)",
        "max_iterations": 3,
        "provider": "qwen",
        "model": "qwen2.5-coder:7b"
      }

    Response body (JSON):
      {
        "improved": true/false,
        "original_sql": "...",
        "recommended_sql": "...",
        "action": "optimize|keep_original|...",
        "score": 8,
        "performance": {"score": 7, "label": "improves"},
        "risk": {"score": 3, "label": "low"},
        "semantic": {"label": "equivalent"},
        "iterations": 2,
        "iteration_history": [...],
        "elapsed_seconds": 12.3
      }

    Usage:
      python api.py                    # starts on port 5000
      python api.py --port 8080        # custom port

      curl -X POST http://localhost:5000/optimize \
        -H "Content-Type: application/json" \
        -d '{"sql": "SELECT * FROM orders"}'
"""

import json
import sys
import os
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class PipelineHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok", "service": "sql-optimizer"})
        else:
            self._json_response(404, {"error": "Not found. Use POST /optimize"})

    def do_POST(self):
        if self.path != "/optimize":
            self._json_response(404, {"error": "Not found. Use POST /optimize"})
            return

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._json_response(400, {"error": "Invalid JSON"})
            return

        sql = payload.get("sql", "").strip()
        if not sql:
            self._json_response(400, {"error": "Missing 'sql' field"})
            return

        ddl = payload.get("ddl", "")
        max_iter = payload.get("max_iterations", 3)
        provider = payload.get("provider", "qwen")
        model = payload.get("model", "qwen2.5-coder:7b")
        api_key = payload.get("api_key")

        # Run pipeline
        from pipeline import run_pipeline

        start = time.time()
        try:
            result = run_pipeline(
                original_sql=sql,
                ddl_context=ddl,
                provider=provider,
                model=model,
                api_key=api_key,
                max_iterations=max_iter,
            )
        except Exception as e:
            self._json_response(500, {"error": str(e)})
            return

        elapsed = time.time() - start

        # Build response
        rec = result.get("recommendation", {})
        best = result.get("best_result", {})
        rec_sql = best.get("recommended_sql") or result.get("recommended_sql", sql)

        improved = (
            rec_sql != sql
            and rec.get("is_valid", False)
            and rec.get("action") != "keep_original"
        )

        response = {
            "improved": improved,
            "original_sql": sql,
            "recommended_sql": rec_sql if improved else sql,
            "action": rec.get("action", "keep_original"),
            "score": rec.get("score", 0),
            "confidence": rec.get("confidence", "low"),
            "is_valid": rec.get("is_valid", False),
            "optimizations_applied": rec.get("optimizations_applied", []),
            "rationale": rec.get("rationale", ""),
            "performance": {
                "score": result.get("performance_label", {}).get("score"),
                "label": result.get("performance_label", {}).get("label"),
                "rationale": result.get("performance_label", {}).get("rationale"),
            },
            "risk": {
                "score": result.get("risk_label", {}).get("score"),
                "label": result.get("risk_label", {}).get("label"),
                "rationale": result.get("risk_label", {}).get("rationale"),
            },
            "semantic": {
                "label": result.get("semantic_label", {}).get("label"),
                "confidence": result.get("semantic_label", {}).get("confidence"),
                "rationale": result.get("semantic_label", {}).get("rationale"),
            },
            "er_graph": {
                "total_tables": result.get("er_graph", {}).get("total_tables"),
                "graph_depth": result.get("er_graph", {}).get("graph_depth"),
                "cross_table_risk": result.get("er_graph", {}).get("cross_table_risk"),
            },
            "iterations": result.get("iteration", 0),
            "iteration_history": result.get("iteration_history", []),
            "elapsed_seconds": round(elapsed, 2),
        }

        self._json_response(200, response)

    # CORS preflight
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode("utf-8"))

    def log_message(self, format, *args):
        sys.stderr.write(f"[API] {args[0]} {args[1]} {args[2]}\n")


def main():
    parser = argparse.ArgumentParser(description="SQL Optimizer API Server")
    parser.add_argument("--port", "-p", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), PipelineHandler)
    print(f"SQL Optimization API running on http://{args.host}:{args.port}")
    print(f"  POST /optimize  — submit SQL for optimization")
    print(f"  GET  /health    — health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
