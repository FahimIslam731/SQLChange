#!/bin/bash
# ──────────────────────────────────────────────────────
# SQL Optimizer — Test Query with 6 Anti-Patterns
# ──────────────────────────────────────────────────────

# Schema: 4 tables — HR analytics scenario
DDL="
CREATE TABLE employees (
  id INT, name TEXT, department_id INT,
  hire_date DATE, salary DECIMAL, is_active INT
);
CREATE TABLE departments (
  id INT, dept_name TEXT, manager_id INT, budget DECIMAL
);
CREATE TABLE projects (
  id INT, project_name TEXT, department_id INT,
  start_date DATE, end_date DATE, status TEXT
);
CREATE TABLE assignments (
  id INT, employee_id INT, project_id INT,
  role TEXT, hours_worked DECIMAL
);
"

# The intentionally bad query — 6 anti-patterns:
#
#  1. SELECT DISTINCT e.*     → should be explicit columns
#  2. Scalar subqueries       → project_count & total_hours re-scan
#                                assignments per row (N+1 pattern)
#  3. NOT IN (nested)         → should be NOT EXISTS or LEFT JOIN IS NULL
#                                (also breaks with NULLs)
#  4. UPPER(d.dept_name)      → function on column kills index usage
#  5. Correlated AVG subquery → should be CTE or window function
#  6. Redundant JOINs         → assignments/projects already queried
#                                in scalar subs, JOIN just adds dupes
#                                that DISTINCT then has to remove

SQL="
SELECT DISTINCT e.*, d.dept_name,
  (SELECT COUNT(*)
   FROM assignments a2
   WHERE a2.employee_id = e.id) AS project_count,
  (SELECT SUM(hours_worked)
   FROM assignments a3
   WHERE a3.employee_id = e.id) AS total_hours
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id
LEFT JOIN assignments a ON a.employee_id = e.id
LEFT JOIN projects p ON p.id = a.project_id
WHERE e.is_active = 1
AND e.id NOT IN (
  SELECT employee_id FROM assignments
  WHERE project_id IN (
    SELECT id FROM projects WHERE status = 'cancelled'
  )
)
AND UPPER(d.dept_name) != 'TEMP'
AND e.salary > (
  SELECT AVG(salary) FROM employees
  WHERE department_id = e.department_id
)
ORDER BY e.salary DESC
"

# Run the pipeline
#   --ddl         schema so parser gets 100% accurate extraction
#   --iterations  3 passes to refine the optimization
#   --model       your local Ollama model
python cli.py \
  --ddl "$DDL" \
  --iterations 3 \
  --model qwen2.5-coder:7b \
  "$SQL"
