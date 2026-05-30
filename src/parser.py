"""
    This python file parses the csv file and parses the individual string entries 
    into specific columns and dataframes

    Author: Dev Rathod
    Date: 05/11/2026
    File Name: parser.py
"""

import re
import sqlglot
from sqlglot import exp

# Time-unit keywords that sqlglot misparses as Column nodes
time_units = {"MINUTE", "HOUR", "DAY", "WEEK", "MONTH", "QUARTER", "YEAR",
                "SECOND", "MICROSECOND", "minute", "hour", "day", "week",
                "month", "quarter", "year", "second"}

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


def validate_sql_columns(dataframe_string, sql_column_details, original_sql_query):
    """
        This function inputs all the sql columns names from the dictonary and checks
        if all the columns are present in the database or not
    """
    # Variables to store all the individual alias in the sql query
    sql_columns_original = set()
    # Variables to store all the columns present in the dictionary
    sql_query_aliases_mutated = set()
    sql_query_aliases_original = set()

    # Parsing the sql string into a sql tree
    try:
        sql_tree = sqlglot.parse_one(dataframe_string)
        sql_query_aliases_mutated = extract_aliases(sql_tree=sql_tree)
    except Exception:
        print(f"ERROR: parser.py : Dataframe could not be parsed by sqlglot library: {dataframe_string}")
        return None

    # If the original sql query was inputed by the user then find the aliases in the original query
    if original_sql_query:
        try:
            # Converting the original sql query into a sql tree
            original_sql_tree = sqlglot.parse_one(original_sql_query)
            # Finding the list of all the aliases present in the tree
            sql_query_aliases_original = extract_aliases(sql_tree=original_sql_tree)
            sql_columns_original = extract_columns(sql_tree=original_sql_tree)
        except Exception as e:
            print(f"Error: parser.py: {str(e)} while finding aliasses in the original query")

    # Taking a union of aliases found in both original and mutated sql query
    sql_query_aliases = sql_query_aliases_mutated | sql_query_aliases_original

    # Variable to store all the columns present in the dictionary
    sql_columns = []

    for column_entry in sql_column_details.values():
        for column in column_entry["columns"]:
            sql_columns.append(column)

    # Variable to find all the column names present in the sql database 
    sql_dataframe_columns = sql_tree.find_all(exp.Column)

    for column in sql_dataframe_columns:
        # Skip if the user wants to select through all the columns in the database
        if column.name == "*" or not column.name:
            continue
        elif column.name in sql_columns:
            continue
        elif column.name in sql_query_aliases:
            continue
        elif column.name in time_units:
            continue
        elif column.name in sql_columns_original:
            continue

        print(f"ERROR: parser.py : Could not find the column entry {column.name} in the database")
        return False

    return True

def parse_sql(dataframe_string: str):
    """
        This function extracts the data from the dataframe present in the csv file 
        and seperates out the columns while checking if they exists in the csv file
    """

    # Variable to store the dictionary containing the column names and the column types
    sql_column_details = {}

    # Parsing the sql string into a sql tree
    sql_tree = sqlglot.parse(dataframe_string)

    # Building a dataset only if the user wanted to create a table
    for sql_statemnt in sql_tree:
        # Variable to store the individual columns and types of columns build by the user
        individual_columns = []
        column_types = {}

        create_table_function = exp.Create

        # Only build a dataset if the user wanted to build a table 
        if not isinstance(sql_statemnt, create_table_function):
            continue

        # Extracting the table name from the sql statement
        table_node = sql_statemnt.find(exp.Table)
        table_name = table_node.name
        # Boolean variable find if a column entry is been found in the dataset
        column_found = False

        # Iterrating through all the columns present in the table dataset
        for column in sql_statemnt.find_all(exp.ColumnDef):
            # Extracting the column type from the query
            type = column.args.get("kind")  
            if type:
                type = str(type)
            else:
                type = "UNKNOWN"

            individual_columns.append(column.name)
            column_found = True
            # Storing the types of the columns in a dictionary 
            column_types[column.name] = type

        # Storing all the individual columns in the squery in a database 
        if column_found:
            sql_column_details[table_name] = {
                "columns": individual_columns,
                "types": column_types
            }

    return sql_column_details

def get_where_details(dataframe_string: str):
    """
        This functions finds all the details present in the where sql query and 
        extracts them into a dictionary wwhich would be used to build langraph later in pipeline 
    """
    try:
        # Parsing the sql string into a sql tree 
        sql_tree = sqlglot.parse_one(dataframe_string)
        # Extracting the where condition in the dataframe for building the langgraph
        where_details = sql_tree.find_all(exp.Where)

        if not where_details:
            return []
        
        # Variable to store all the dependencies in the where queries
        dependencies_where = []

        # Iterating through all the individual where conditions in the sql query strings
        for individual_where_query in where_details:
            for columns in individual_where_query.find_all(exp.Column):
                dependencies_where.append({
                    "condition": str(individual_where_query.this),
                    "table":     columns.table or "UNKNOWN",
                    "column":    columns.name
                })

        return dependencies_where
    
    except Exception:
        print(f"Error: parser.py : could dnot parse the where sql query : {dataframe_string}")
        return []


def get_join_keys(dataframe_string: str):
    """
        This functions finds all the relationships between the join keys for 
        building the langraph

        example query = JOIN Customers ON Orders.CustomerID = Customers.CustomerID;
    """
    try:
        # Parsing the sql string into a sql tree
        sql_tree = sqlglot.parse_one(dataframe_string)
        join_statement = sql_tree.find_all(exp.Join)
        # Building a relationship between the keys from the join condition
        join_relationships = []

        # Iterating though all the join conditions in the dataframe
        for individual_join_conditions in join_statement:
            # Extracting the on conditions from the join condition
            join_on_parameter = individual_join_conditions.args.get("on")

            # If there are not hyperparameters for join conditions, skip the entry as no reationships could be built
            if not join_on_parameter:
                continue

            # Variable to find all the join keys present in the join condition
            join_entries = join_on_parameter.find_all(exp.EQ)

            # Iterrating through all the join keys present in the on clause statement 
            for individual_relations in join_entries:
                left_key = individual_relations.left
                right_key = individual_relations.right

                # If both sides contain valid column entries then build a relationship between them
                if isinstance(left_key, exp.Column) and isinstance(right_key, exp.Column):
                    join_relationships.append({
                        "right_table": right_key.table or "UNKNOWN",
                        "left_table": left_key.table or "UNKNOWN",
                        "right_column": right_key.name,
                        "left_column": left_key.name
                    })

        return join_relationships
    
    except Exception:
        print(f"Error: parser.py : Error while building the relationship map for the dataframe {dataframe_string}")   
        return []