"""
    This python file loads all the column names into the LLM to predict 
    the respective data types for each of the column names. If the llm
    confidence score in predicting the data types for each of the query is 
    really low then there is a fallback to find the backdoor for the pipeline

    Author: Dev Rathod
    Date: 05/11/2026
    File Name: mutation_engine.py
"""

import json
import sqlglot  
from sqlglot import exp

# Variable to store all the valid datatypes for sql lite query, to limit the llm model to guess anything random
_VALID_TYPES = {
    "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT",
    "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC",
    "TEXT", "VARCHAR", "CHAR", "STRING", "CLOB",
    "BOOLEAN", "BOOL",
    "DATE", "DATETIME", "TIMESTAMP", "TIME",
    "BLOB", "BINARY",
}

# Storing common time datatypes and keywords
_TIME_UNITS = {
    "MINUTE", "HOUR", "DAY", "WEEK", "MONTH", "QUARTER", "YEAR",
    "SECOND", "MICROSECOND", "minute", "hour", "day", "week",
    "month", "quarter", "year", "second",
}
 
def extract_aliases(sql_tree):
    """
        This function finds all the individual aliases and queries in a given sql
        query and return the set of all the individual parameters for validation.
    """
    # Variable to store all the individual alias in the sql query
    sql_query_aliases = set()

    # Searching all the aliases in the sql query 
    aliases = sql_tree.find_all(exp.Alias)
    for individual_aliases in aliases:
        if individual_aliases.alias:
            sql_query_aliases.add(individual_aliases.alias)

    # Searching through all the common result tables in the sql query
    ctes = sql_tree.find_all(exp.CTE)
    for individual_cte in ctes:
        if individual_cte.alias:
            sql_query_aliases.add(individual_cte.alias)
    
    return sql_query_aliases

def extract_columns(sql_tree):
    """
        This function extracts the columns for a given parsed sql tree and returns
        a set of all the unique columns present in the sql tree
    """
    # Variable to store all the columns present in the dictionary
    sql_columns = []
    sql_column_details = sql_tree.find_all(exp.Column)

    for column_entry in sql_column_details:
        if column_entry and column_entry.name != "*":
            sql_columns.append(column_entry.name)
    
    return sql_columns

def extract_tables(sql_query) -> dict[str, set[str]]:
    """
        This function extracts the table names from the parsed
        sql query and inputs the sql tree as input. Excludes all the
        aliasis and subquery aliases.
    """
    # Collect CTE aliases to exclude them
    cte_aliases = set()
    for cte_node in sql_query.find_all(exp.CTE):
        if cte_node.alias:
            cte_aliases.add(cte_node.alias)
 
    tables = set()
    for tbl in sql_query.find_all(exp.Table):
        name = tbl.name
        if name and name not in cte_aliases:
            tables.add(name)
 
    return tables

def validate_llm_response(llm_context: dict, columns: list, aliases: set, tables: set) -> bool:
    """
        This python function inputs the llm response for making a context json data array
        to categorize each column in a specific data type and validates if all the columns 
        names and datatypes are coherent.
    """

    # If context is null return false
    if not llm_context:
        return False
    
    # Finding all teh coluns that the llm returned in the context
    context_columns = set()
    for table_information in llm_context.values():
        if isinstance(table_information, dict):
            for col in table_information.get("columns", []):
                context_columns.add(col)

    # If the llm returns a type which is not recognised in by sql lite, have a fallback
    # Checkign if all the datatypes in the context are valid
    for table_information in llm_context.values():
        if not isinstance(table_information, dict):
            continue
        data_types = table_information.get("types", {})
        # Iterrating through all the types outputted by the llm model
        for individual_columns, column_type in data_types.items():
            if column_type.upper() not in _VALID_TYPES:
                return False
    
    # Checking if all the tables exisist in the context 
    for individual_table in tables:
        if individual_table not in llm_context:
            return False
        
    ## Checkign if all the columns exists in the LLM context if they don't then return false
    for col_name in columns:
        if col_name in aliases:
            continue
        if col_name in _TIME_UNITS:
            continue
        if col_name not in context_columns:
            return False
        
    return True

def infer_llm_context(sql_query, llm_call_function, provider, api_key, model) -> dict:
    """
        This function makes a llm call to infer context from the query to segregate each 
        individual column name to a specific datatype which will be used in later downstream stage
    """
    if not sql_query or not sql_query.strip():
        return {}
 
    # Step 1: parse the query and extract columns, aliases, tables for validation
    try:
        sql_tree = sqlglot.parse_one(sql_query)
        query_columns = extract_columns(sql_tree)
        query_aliases = extract_aliases(sql_tree)
        query_tables = extract_tables(sql_tree)
    except Exception:
        return {}
 
    # Step 2: ask LLM to infer the schema
    prompt = f"""You are a database schema analyst. Given the SQL query below,
    infer the complete table schema — every table referenced and every column
    used, with the most likely SQL data type for each column.
    
    SQL Query:
    {sql_query}
    
    Return ONLY a JSON object with this exact structure, no explanations, no markdown:
    {{
        "table_name": {{
            "columns": ["col1", "col2"],
            "types": {{"col1": "INT", "col2": "TEXT"}}
        }},
        "confidence": 0.85
    }}
    
    For each column, pick EXACTLY ONE type from the catalog below. Do NOT use any
    type not listed here:
    
    Numeric (whole numbers):
        - INT        : standard integer (id, count, quantity, age, year)
        - BIGINT     : large integer (large IDs, timestamps as epoch, very large counts)
        - SMALLINT   : small-range integer (status codes, flags as 0/1, enums)
        - TINYINT    : very small integer (boolean-like 0/1, single-digit codes)
    
    Numeric (decimals):
        - DECIMAL    : exact precision (price, salary, revenue, balance, amount, GPA)
        - FLOAT      : approximate single-precision (scientific measurements, ratios)
        - DOUBLE     : approximate double-precision (coordinates, high-precision calcs)
        - NUMERIC    : same as DECIMAL (exact fixed-point)
        - REAL       : same as FLOAT (approximate)
    
    Text:
        - TEXT       : unbounded text (descriptions, content, comments, notes, long strings)
        - VARCHAR    : bounded text (name, email, phone, address, short strings)
        - CHAR       : fixed-length text (country codes, currency codes, status codes)
        - STRING     : generic text (use when length is unclear)
    
    Boolean:
        - BOOLEAN    : true/false (is_active, has_permission, is_deleted, flags)
    
    Date / Time:
        - DATE       : date only (birth_date, order_date, created_date)
        - DATETIME   : date + time (created_at, updated_at, login_time)
        - TIMESTAMP  : date + time + timezone (event_timestamp, log entries)
        - TIME       : time only (start_time, end_time, duration)
    
    Binary:
        - BLOB       : binary data (images, files, serialized objects)
        - BINARY     : fixed-length binary (hashes, UUIDs as binary)
    
    Include ALL columns that appear anywhere in the query (SELECT, WHERE, JOIN ON,
    GROUP BY, ORDER BY, HAVING, UPDATE SET, DELETE, PARTITION BY, CASE WHEN).
    Do not omit any column.
    
    At the end of the JSON, include a "confidence" field (0.0 to 1.0) indicating
    how confident you are that ALL the inferred types are correct. Use:
    0.9-1.0 : column names strongly suggest types (e.g. "price" → DECIMAL, "id" → INT)
    0.7-0.9 : most types are clear but some are ambiguous
    0.5-0.7 : many columns have ambiguous types
    below 0.5 : largely guessing
    """
 
    try:
        response = llm_call_function(
            prompt=prompt, provider=provider,
            api_key=api_key, model=model
        )
 
        # Parse the LLM response
        text = response.strip().replace("```json", "").replace("```", "").strip()
        context = json.loads(text)
 
    except Exception:
        # LLM call failed or JSON parse failed → kill
        return {}
 
    # Step 3: validate — if anything is missing or types are bad, kill it
    if not validate_llm_response(context, query_columns, query_aliases, query_tables):
        return {}
 
    # Normalize types to uppercase
    for table_info in context.values():
        if not isinstance(table_info, dict):
            continue
        types = table_info.get("types", {})
        for col in list(types.keys()):
            types[col] = types[col].upper()
 
    return context
