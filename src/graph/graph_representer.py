"""
    This python file creates a lang graph using the details from the original join keys,
    ddl and where dependencies. It handles the full pipeline from raw SQL input through
    schema extraction (via DDL or inference) to building the entity relationship graph.

    The pipeline first checks if DDL context is provided by the user. If DDL is available
    it uses parse_sql() for 100% accurate schema extraction. If no DDL is provided it 
    falls back to infer_context() which uses regex-based extraction with an LLM fallback
    for ~85-90% accuracy.

    The LLM utility has been extracted to utils/llm.py for shared access across modules.
    Provider configuration (model, provider, api_key) is passed through the GraphState
    instead of global variables for thread safety.

    Author: Dev Rathod
    Date: 05/13/2026
    File Name: graph_representer.py
"""

import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

from utils.llm import llm_universal_call_utility

# Global variable to store the cached compiled graph to prevent rebuilding
_cached_graph = None

# Defining the shared state graph for the full pipeline
class GraphState(TypedDict):
    original_sql: str
    ddl_context: Optional[str]
    join_keys: list
    where_details: list
    table_names: list
    table_relationships: list
    join_keys_relationships: list 
    sql_column_details: dict
    data_graph: dict
    provider: str
    model: Optional[str]
    api_key: Optional[str]
    inference_method: Optional[str]
    error: Optional[str]

def python_node_parse_ddl(state_graph: GraphState) -> GraphState:
    """
        Node 0a (DDL provided path)
        This python function extracts the schema context from the DDL statements provided
        by the user. Uses parse_sql() from the parsing module for 100% accurate schema
        extraction. Additionally it extracts the join keys and where dependencies from the
        original sql query for building the entity relationship graph.
    """
    from parsing.parser import parse_sql, get_join_keys, get_where_details

    # Extracting the schema context from the DDL statements using the parser
    context = parse_sql(state_graph["ddl_context"])

    # Getting all the join keys relationships for the sql query
    join_keys = get_join_keys(state_graph["original_sql"])

    # Getting all the where dependencies inside the sql query
    where_details = get_where_details(state_graph["original_sql"])

    # Storing the extracted information in the state graph
    state_graph["sql_column_details"] = context
    state_graph["join_keys"] = join_keys
    state_graph["where_details"] = where_details
    state_graph["inference_method"] = "ddl_parse"

    return state_graph

def python_node_infer_context(state_graph: GraphState) -> GraphState:
    """
        Node 0b (No DDL path)
        This python function infers the schema context directly from the SQL query when
        no DDL is provided by the user. It first attempts regex-based extraction for speed
        and falls back to LLM inference if the regex extraction returns low confidence or
        empty context. The fallback uses the provider configuration from the state graph.
    """
    from parsing.infer_context import infer_context

    # Inferring the schema context from the sql query using regex with LLM fallback
    inferred = infer_context(
        sql=state_graph["original_sql"],
        provider=state_graph["provider"],
        model=state_graph.get("model"),
        api_key=state_graph.get("api_key"),
    )

    # Storing the inferred context and extracted relationships in the state graph
    state_graph["sql_column_details"] = inferred.get("context", {})
    state_graph["join_keys"] = inferred.get("join_keys", [])
    state_graph["where_details"] = inferred.get("where_details", [])
    state_graph["inference_method"] = inferred.get("inference", {}).get("method", "inferred")

    return state_graph

def python_node_table_parsor(state_graph: GraphState) -> GraphState:
    """
        Node 1
        This python function parses information in the input dictionary to extract the 
        table names present in the database. Always innvoked, regardless of the path
        the input has been taken in.
    """
    # Extracting the table names from the sql column details dictionary keys
    tables = list(state_graph["sql_column_details"].keys())
    state_graph["table_names"] = tables
    return state_graph

def python_node_build_graph_join_keys(state_graph: GraphState) -> GraphState:
    """
        Node 2 (part a: Python function)
        This python function builds the relationships from the extracted join keys. The
        function only runs if the join keys are available for the particular entry and no
        LLM call is needed. This node is the path for queries that have join conditions
        present in them.
    """
    # Variable to store the join key information
    join_keys = state_graph["join_keys"]
    # Variable to store the relationships between the join keys
    join_keys_relationships = []

    # Iterating through each join keys present in the entry
    for join_key_entry in join_keys:
        join_keys_relationships.append({
            "source":      join_key_entry["left_table"],
            "target":      join_key_entry["right_table"],
            "join_column": join_key_entry["left_column"],
            "confidence":  "high",
            "origin":      "join_key"
        })

    # Adding the information from the join keys to the state graph
    state_graph["join_keys_relationships"] = join_keys_relationships

    return state_graph

def llm_node_build_relationships(state_graph: GraphState) -> GraphState:
    """
        Node 2 (part b: LLM inference)
        This function utilizes a large language model for inference for finding relationships
        between tables where no join keys exist. This function only runs when no join keys are
        associated with a sql entry.
    """
    try:
        # Fetching all the available sql context from the individual query and transforming it into a json entry
        sql_context = state_graph["sql_column_details"]
        sql_context = json.dumps(sql_context, indent=2)

        # Prompt to infer the llm inorder to build a relational map with certain instructions
        prompt = f"""
            You are a database schema analyst. Analyze the table schemas below and infer entity relationships based on shared column names and naming conventions.
            Schema: {sql_context}

            Prompt Output Details: response should only be made in json with no explanations or markdowns. example:
            {{
            "table_relationships": [
                {{
                "source": "parent_table_name",
                "target": "child_table_name",
                "join_column": "shared_column_name",
                "confidence": "high|medium|low",
                "origin": "inferred"
                }}
            ]
            }}
            If no relationships to the table are found respond {{"table_relationships":[]}}
        """

        # Getting the response for the prompt from the LLM using the provider config from state
        response = llm_universal_call_utility(
            prompt=prompt,
            provider=state_graph["provider"],
            api_key=state_graph.get("api_key"),
            model=state_graph.get("model"),
        )

        # If the response from the llm is none log the error
        if response is None:
            state_graph["table_relationships"] = []
            state_graph["error"] = "Error while getting a response from LLM"
        else:
            ## Reference: https://python.plainenglish.io/getting-structured-json-responses-from-llms-a-simple-solution-f819fc389ebc 
            # Parse the response into a dictionary
            text = response.strip()
            clean = text.replace("```json", "").replace("```", "").strip()
            data  = json.loads(clean)

            state_graph["join_keys_relationships"] = data.get("table_relationships", [])
            
        return state_graph
    
    except Exception as e:
        print(f"Error: graph_representer.py: LLM inference failed: {e}")
        state_graph["error"] = str(e)
        state_graph["join_keys_relationships"] = [] 
        return state_graph

def node_build_graph(state_graph: GraphState) -> GraphState:
    """
        Node 3 
        This python function builds the output graph with levels of heirarchy identifing
        the imporatance of each table and column using identifiers: {root, intermidiate,
        leaf}. This node runs regardless of which path was chosen by the input for a given 
        output.

        Input:  state["tables"] + state["join_keys_relationships"] + state["where_details"]
        Output: state["data_graph"] complete graph dict
    """
    # Variables to store all the details to build the level of importance graph
    join_key_relationships = state_graph["join_keys_relationships"]
    where_details = state_graph["where_details"]
    tables = state_graph["table_names"]

    # Fetching all the source and target tables in seperate vectors 
    target_tables = {}
    all_tables = set()
    for relationship in join_key_relationships:
        if relationship.get("target") and relationship["target"] != "UNKNOWN":
            target_tables[relationship["target"]] = relationship["target"]
            all_tables.add(relationship["target"])
    source_tables = {}
    for relationship in join_key_relationships:
        if relationship.get("source") and relationship["source"] != "UNKNOWN":
            source_tables[relationship["source"]] = relationship["source"]
            all_tables.add(relationship["source"])

    # Determining the level of imporatance as:
    # root: if the chain is started from this table
    # intermidiate: if the table has both source and target relationships
    # leaf: if the table has only target relationships and not source relationships
    table_imporatance = []
    for individual_table in tables:
        if individual_table not in target_tables:
            importance = "root"
        elif individual_table not in source_tables:
            importance = "leaf"
        else:
            importance = "intermediate"

        table_imporatance.append({"table": individual_table, "importance": importance})

    # Finding all the table entries in the join statements
    all_joined_tables = set()
    for relationship in join_key_relationships:
        if relationship.get("target"):
            all_joined_tables.add(relationship["target"])

    # Finding all the cross table entries present in join relationships and where tables
    table_join_where_entries = []
    for where in where_details:
        if where.get("table") in all_joined_tables:
            table_join_where_entries.append(where)

    # Building a graph entry in a dictonary data variable
    output_graph = {
        "join_relationships": join_key_relationships,
        "table_importance": table_imporatance,
        "where_dependencies": where_details,
        "join_where_tables": table_join_where_entries,
        "cross_table_risk": len(table_join_where_entries) > 0,
        "graph_depth": len(all_tables),
        "total_tables": len(tables)
    }

    state_graph["data_graph"] = output_graph

    return state_graph

def route_ddl_or_infer(state_graph: GraphState) -> str:
    """
        This function is a conditional node which routes the input for the initial
        schema extraction step. If DDL context is provided by the user, the input takes
        the path with parse_sql() for accurate extraction. If no DDL is provided it falls
        back to infer_context() which uses regex extraction with LLM fallback.
    """
    # If DDL context is provided and not empty then use the parse_sql path
    # Otherwise fall back to inferring the context from the query itself
    if state_graph.get("ddl_context") and state_graph["ddl_context"].strip():
        return "python_node_parse_ddl"
    else:
        return "python_node_infer_context"

def route_post_parsing_details(state_graph: GraphState) -> str:
    """
        This function is a conditional node which routes the input for node 2.
        If join relationships are present, the input takes the path with node 2 (python)
        else it calls LLM to infer the relationship.
    """
    # If join keys is not present then make a LLM API call for building relationships else
    # Call an internal function to build relationships without api calls
    if state_graph["join_keys"] or len(state_graph["sql_column_details"]) <= 1:
        return "python_node_build_graph_join_keys"
    else:
        return "llm_node_build_relationships"

def build_graph_pipeline():
    """
        This python function creates a layout for all the nodes and edges to form a
        graph, inorder to transverse through all the queries in the pipeline. The graph
        now includes the initial DDL routing step before the ER graph construction.
    """
    # Initialising a lang graph state graph to build the full pipeline
    graph = StateGraph(GraphState)

    # Adding the DDL routing nodes to the graph
    graph.add_node("python_node_parse_ddl", python_node_parse_ddl)
    graph.add_node("python_node_infer_context", python_node_infer_context)

    # Adding all the ER graph building nodes to the graph
    graph.add_node("python_node_table_parsor", python_node_table_parsor)
    graph.add_node("python_node_build_graph_join_keys", python_node_build_graph_join_keys)
    graph.add_node("llm_node_build_relationships", llm_node_build_relationships)
    graph.add_node("node_build_graph", node_build_graph)

    # Setting the entry point to the DDL routing conditional
    graph.set_entry_point("route_ddl_or_infer")
    graph.add_node("route_ddl_or_infer", lambda state: state)

    # Adding conditional edges for the DDL routing after the entry point
    graph.add_conditional_edges("route_ddl_or_infer", route_ddl_or_infer,
                                {
                                    "python_node_parse_ddl": "python_node_parse_ddl",
                                    "python_node_infer_context": "python_node_infer_context"
                                })

    # Both DDL paths converge into the table parser node
    graph.add_edge("python_node_parse_ddl", "python_node_table_parsor")
    graph.add_edge("python_node_infer_context", "python_node_table_parsor")

    # Adding conditional edges to the lang graph after the table parser
    graph.add_conditional_edges("python_node_table_parsor", route_post_parsing_details,
                                {
                                    "python_node_build_graph_join_keys": "python_node_build_graph_join_keys",
                                    "llm_node_build_relationships": "llm_node_build_relationships"
                                })

    # Adding the linear edges to building the final graph
    graph.add_edge("python_node_build_graph_join_keys", "node_build_graph")
    graph.add_edge("llm_node_build_relationships", "node_build_graph")

    # Point the final node to the end of the graph
    graph.add_edge("node_build_graph", END)

    return graph.compile()

def build_graph(original_sql: str, ddl_context: str = None, model_name: str = None, provider: str = "qwen", api_key: str = None) -> dict:
    """
        This python function is the main entry point to the program which is called by
        other modules in the pipeline. It takes the raw SQL query and optional DDL context
        and runs the full pipeline from schema extraction through ER graph construction.
        Additionally we would also cache the graph inorder to prevent it from rebuilding
        each time.
    """
    global _cached_graph

    if _cached_graph is None:
        _cached_graph = build_graph_pipeline()

    # Running the langgraph for the query element through the full pipeline
    langgraph_output = _cached_graph.invoke({
        "original_sql": original_sql,
        "ddl_context": ddl_context,
        "join_keys": [],
        "where_details": [],
        "table_names": [],
        "table_relationships": [],
        "join_keys_relationships": [],
        "sql_column_details": {},
        "data_graph": {},
        "provider": provider,
        "model": model_name,
        "api_key": api_key,
        "inference_method": None,
        "error": None
    })

    return langgraph_output