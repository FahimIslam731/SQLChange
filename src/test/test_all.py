# test_all.py — run from your src/ folder
from mutation_engine import match_sql_to_mutation, mutation_function_mapping
from sql_parser import parse_sql, validate_sql_columns, get_join_keys

sql  = "SELECT s.name, t.salary FROM schools s JOIN teachers t ON s.school_id = t.school_id WHERE s.district = 'North' GROUP BY s.name"
ddl  = "CREATE TABLE schools (school_id INT, school_name VARCHAR, district VARCHAR, name VARCHAR); CREATE TABLE teachers (teacher_id INT, school_id INT, salary INT);"

print("Applicable:", match_sql_to_mutation(sql))
print("Schema:",     parse_sql(ddl))
print("Join keys:",  get_join_keys(sql))
print("Valid:",      validate_sql_columns(sql, parse_sql(ddl)))

for mutation in match_sql_to_mutation(sql):
    result = mutation_function_mapping[mutation](sql)
    print(f"\n{mutation}")
    print(f"  IN:  {sql}")
    print(f"  OUT: {result}")