"""
    This function inputs a SQL query and reproduced a slightly modified query while 
    validating if the modified query is a legal query present in the SQL databsase.

    Author: Dev Rathod
    Date: 05/11/2026
    File Name: mutation_engine.py
"""

import sqlglot
from sqlglot import exp

# Variable to store the specific sql dialect for building the dataset
dialect = "sqlite"

def _where_drop(sql_string : str):
    """
        Function to drop the where condition from the sql query 
    """
    try:
        # Parsing the SQL query form a string to a tree and finding the where condition
        sql_tree = sqlglot.parse_one(sql_string)
        sql_where = sql_tree.find(exp.Where)
        
        # If there is no where condition return None
        if not sql_where:
            return None

        # Store the condition in the where sql condition
        where_condition = sql_where.this
        # Finding if there are multiple conditions present in the where condition 
        # If multiple conditions are present -> drop one else drop all
        if isinstance(where_condition, exp.And):
            sql_where.set("this", where_condition.left)
        else:
            sql_where.pop()

        return sql_tree.sql(dialect=dialect)       
    except Exception:
        print(f"Error: mutation_engine.py : Error while dropping where condition: {sql_string}")
        return None

def _join_swap(sql_string : str):
    """
        Function to swap the items in the join statement to create mutations
    """
    try:
        # Parsing the SQL query form a string to a tree and finding the join condition
        sql_tree = sqlglot.parse_one(sql_string)
        sql_join = sql_tree.find(exp.Join)
        
        # If there is no where condition return None
        if not sql_join:
            return None
        
        # Store the condition in the where sql condition
        join_condition = sql_join.args.get("kind", "")
        if join_condition:
            join_condition = str(join_condition).upper()
        else:
            join_condition = ""

        # If there is a left condition present then swap it with the right condition
        if "LEFT" in join_condition:
            sql_join.set("kind", "INNER")
        else:
            sql_join.set("kind", "LEFT OUTER")

        return sql_tree.sql(dialect=dialect)       
    except Exception:
        print(f"Error: mutation_engine.py : Error while swapping the join condition: {sql_string}")
        return None

def _join_drop(sql_string: str):
    """
        Function to drop the items in the join statement to create mutations
    """
    try:
        # Parsing the SQL query form a string to a tree and finding the join condition
        sql_tree = sqlglot.parse_one(sql_string)
        joins = list(sql_tree.find_all(exp.Join))
        
        # If there is no join condition return None
        if not joins:
            return None
        
        # Finding the last join statement in the sql sequence to remove 
        last_statement_join = joins[-1]
        if last_statement_join.find(exp.Table):
            dropped_element = last_statement_join.find(exp.Table).name.lower()
        else:
            dropped_element = None
        
        # Removing the last join statement from the sequence of SQL conditions
        last_statement_join.pop()

        # Dropping all the columns created in the the new join statement in later statements
        if dropped_element:
            # Finding the items from all the SELECT statements
            select_statement = sql_tree.find(exp.Select)

            # Checking if the select statement exists in the sequence of sql statements 
            if select_statement:
                # Storing all the new columns for the select statements
                new_columns = []
                for column_names in select_statement.expressions:
                    individual_columns = column_names.find(exp.Column)

                    # If the column names are not in the join statement then store it else drop them
                    if not (individual_columns and 
                            individual_columns.table and 
                            individual_columns.table.lower() == dropped_element):
                        new_columns.append(column_names)

                # Setting all the expressions to new column names in the sql tree 
                select_statement.set("expressions", new_columns if new_columns else [exp.Star()])

        return sql_tree.sql(dialect=dialect) 
    
    except Exception:
        print(f"Error: mutation_engine.py : Error while dropping the join condition: {sql_string}")
        return None

def _group_by_drop(sql_string: str):
    """
        Function to drop the items in the group by statement to create mutations
    """
    try:
        # Parsing the SQL query form a string to a tree and finding the join condition
        sql_tree = sqlglot.parse_one(sql_string)
        sql_group_by = sql_tree.find(exp.Group)
        
        # If there is no group by condition return None
        if not sql_group_by:
            return None
        
        sql_group_by.pop()

        # Removing the having statement if present in the sql statement 
        if sql_tree.find(exp.Having):
            sql_tree.find(exp.Having).pop()

        # Finding the items from all the SELECT statements
        select_statement = sql_tree.find(exp.Select)

        # Checking if the select statement exists in the sequence of sql statements 
        if select_statement:
            # Storing all the new columns for the select statements
            new_columns = []
            for column_names in select_statement.expressions:
                individual_columns = column_names.find(exp.AggFunc)

                # If aggregation is present unwrap it, else keep the column as is
                if individual_columns:
                    inner = individual_columns.find(exp.Column)
                    if inner:
                        new_columns.append(inner)
                    # COUNT(*) has no inner column — skip it
                else:
                    new_columns.append(column_names)

            # Setting all the expressions to new column names in the sql tree 
            select_statement.set("expressions", new_columns if new_columns else [exp.Star()])

        return sql_tree.sql(dialect=dialect) 
    
    except Exception:
        print(f"Error: mutation_engine.py : Error while dropping the group by condition: {sql_string}")
        return None

def _limit_add(sql_string: str):
    """
        Function to drop the items in the group by statement to create mutations
    """
    try:
        # Parsing the SQL query form a string to a tree and finding the where condition
        sql_tree = sqlglot.parse_one(sql_string)
        sql_limit = sql_tree.find(exp.Limit)
        
        # If there is no where condition return None
        if sql_limit:
            return None

        sql_tree.set("limit", exp.Limit(expression=exp.Literal.number(100)))

        return sql_tree.sql(dialect=dialect)       
    except Exception:
        print(f"Error: mutation_engine.py : Error while adding the limit condition: {sql_string}")
        return None

def _column_drop(sql_string: str):
    """
        Function to drop the items in the column statement to create mutations
    """
    try:
        # Parsing the SQL query form a string to a tree and finding the where condition
        sql_tree = sqlglot.parse_one(sql_string)
        sql_select = sql_tree.find(exp.Select)
        
        # If there is no where condition return None
        if not sql_select:
            return None

        # If there are two columns then drop one 
        if len(sql_select.expressions) <= 1:
            return None
        
        sql_select.set("expressions", sql_select.expressions[:-1])

        return sql_tree.sql(dialect=dialect)       
    except Exception:
        print(f"Error: mutation_engine.py : Error while dropping a column: {sql_string}")
        return None


"""
    Variable to map the functions to the specific types of mutation/modifications operations:
    - where_drop: Drops the where condition from the sql query 
    - join_swaps: Swaps the internal join elements within a query
    - join_drops: Deletes the join condition from the sql query 
    - limit_add: adds the limit to a certain number of rows
    - column_drop: Drops a specific column from the sql query
    - group_by_drop: Drops a group by operations from the specific query
"""
mutation_function_mapping = {
    "where_drop": _where_drop,
    "join_swap": _join_swap,
    "join_drop": _join_drop,
    "group_by_drop": _group_by_drop,
    "limit_add": _limit_add,
    "column_drop": _column_drop,
}

# Total number of mutuations to make on a specific dataset according to the input operations
mutation_function_occurances = {
    "where_drop": 72,
    "join_swap": 23,
    "join_drop": 23,
    "group_by_drop": 53,
    "limit_add": 100,
    "column_drop": 100,
}

def create_mutations():
    """ 
        Internal function which creates mutations on individual sql commands present in the
        dataset
    """

