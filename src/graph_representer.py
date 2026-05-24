"""
    This python file creates a lang graph using the details from the original join keys,
    ddl and where dependencies.

    Author: Dev Rathod
    Date: 05/13/2026
    File Name: mutation_engine.py
"""

import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

"""
sql_mutated_pairs.append({
                        "unique_id": len(sql_mutated_pairs),
                        "source_id": int(rows["id"]),
                        "domain": rows["domain"],
                        "complexity": rows["sql_complexity"],
                        "context": context,
                        "original_sql": individual_sql_commands,
                        "mutation_type": mutation,
                        "modified_sql": modified_sql_query,
                        "join_keys": join_keys,
                        "where_details": where_details,
                        "risk_label": None,
                        "performance_label": None,
                        "semantic_label": None
                    })
"""

# Global variable to store the cached comipled graph to prevent rebuilding
_cached_graph = None

# Global variable to store the model name, provider the api key
_model_name = None
_provider = None
_api_key = None

# Defining the shared state graph for join keys
class GraphState(TypedDict):
    join_keys: list
    where_details: list
    table_names: list
    table_relationships: list
    join_keys_relationships: list 
    sql_column_details: dict
    data_graph: dict
    error: Optional[str]

def llm_universal_call_utility(prompt: str, provider: str, api_key: str = None, model: str = None, **kwargs):
    """
        This python funciton is a universal utility to pick and choose different types of llms
        for inferencing and getting response for prompts
    """
    response = None
    if provider in ("local", "caliper"):
        import re
        import requests
        import time
        port = 11435 if provider == "caliper" else 11434
        num_predict = kwargs.get("num_predict", 128)
        timeout = kwargs.get("timeout", 600)
        think = kwargs.get("think", True)
        opts = {"num_predict": num_predict}

        resolved_model = model
        if provider == "caliper" and not model:
            try:
                tags = requests.get(f"http://localhost:{port}/api/tags", timeout=5).json()
                models = tags.get("models", [])
                if models:
                    resolved_model = models[0]["name"]
            except Exception:
                pass
        resolved_model = resolved_model or "llama3"

        if provider == "caliper":
            body = {
                "model": resolved_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": opts
            }
            if not think:
                body["think"] = False
            for attempt in range(3):
                r = requests.post(f"http://localhost:{port}/api/chat",
                                  json=body, timeout=timeout)
                data = r.json()
                if data.get("error") == "inference already in progress":
                    print(f"  Caliper busy, retrying in 10s... ({attempt+1}/3)")
                    time.sleep(10)
                    continue
                if "error" in data:
                    raise RuntimeError(data["error"])
                break
            else:
                raise RuntimeError("Caliper busy after 3 retries")
            response = data["message"]["content"]
        else:
            body = {
                "model": resolved_model,
                "prompt": prompt,
                "stream": False,
                "options": opts
            }
            if not think:
                body["think"] = False
            r = requests.post(f"http://localhost:{port}/api/generate",
                              json=body, timeout=timeout)
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            response = data["response"]
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    elif provider == "anthropic":
        import anthropic
        # Initialize the client
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model = model,
            max_tokens = 1024,
            messages=[{"role":"user", "content":prompt}]
        )
        response = response.content[0].text
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model = model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        response = response.choices[0].message.content
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")

    return response

def python_node_table_parsor(state_graph: GraphState) -> GraphState:
    """
        Node 1
        This python function parses information in the input dictionary to extract the 
        table names present in the database. Always innvoked, regardless of the path
        the input has been taken in.
    """
    tables = list(state_graph["sql_column_details"].keys())
    state_graph["table_names"] = tables
    return state_graph

def python_node_build_graph_join_keys(state_graph: GraphState) -> GraphState:
    """
        Node 2 (part a: Python function)
        This python function builds the relationships from the extracted join keys. The
        function only runs if the join keys are available for the particular entry and no
        LLM call is needed. This node is the path for queires that have join conditions
        present in them
    """
    # Variable to store the join key information
    join_keys = state_graph["join_keys"]
    # Variable store the relationships between the join keys
    join_keys_relationships = []

    # Iterating throuh each join keys present in the entry
    for join_key_entry in join_keys:
        join_keys_relationships.append({
            "source":  join_key_entry["left_table"],
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
        This function utilizes an large language model for inferance for finding relationships
        between tables where no join keys exists. This function only runs when no join keys are
        associated with a sql entry.
    """
    # Global variables to store the model name and the API key for inference
    global _model_name
    global _api_key

    try:
        # Fetching all the available sql context from the individual query and tranforming it into a json entry
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
            If no relationships to the table are found repond {{"table_relationships":[]}}
        """

        # Getting the response for the prompt from the LLM
        response = llm_universal_call_utility(prompt=prompt, provider=_provider, api_key=_api_key, model=_model_name)
        # If the response form the llm is none log the error
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
        Output: state["er_graph"] complete graph dict
    """
    # Variables to store all the details to build the level of importance graph
    join_key_relationships = state_graph["join_keys_relationships"]
    where_details = state_graph["where_details"]
    tables = state_graph["table_names"]

    # Fetching all the source and target tables in seperate vectiors 
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
    # root: if the chain is stated from this table
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
        all_joined_tables.add(relationship["target"])

    # Finding all the cross table entries present in join relationships and where tables
    table_join_where_entries = []
    for where in where_details:
        if where["table"] in all_joined_tables:
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

def route_post_parsing_details(state_graph: GraphState) -> GraphState:
    """
        This function is a conditional node which route's the input for node 2.
        If join relationships are present, the inputs takes the path with node 2 (python)
        else it calls LLM to infer the relationship.
    """
    # If join keys is not present then make a LLM API call for building relationships else
    # Call an internal function to build relationships without api calls
    if state_graph["join_keys"] or len(state_graph["sql_column_details"]) <= 1:
        return "python_node_build_graph_join_keys"
    else:
        return "llm_node_build_relationships"

def build_graph_pipleine():
    """
        This python funciton creates a layout for all the nodes and edges to form a
        graph, inorder to transverse through all the original and mutated queries in the 
        dataset
    """
    # Initialising a lang graph state graph to build the inference pipeline
    graph = StateGraph(GraphState)

    # Adding all the nodes to the graph
    graph.add_node("python_node_table_parsor", python_node_table_parsor)
    graph.add_node("python_node_build_graph_join_keys", python_node_build_graph_join_keys)
    graph.add_node("llm_node_build_relationships", llm_node_build_relationships)
    graph.add_node("node_build_graph", node_build_graph)

    # Setting the entry point
    graph.set_entry_point("python_node_table_parsor")

    # Adding conditional edges to the lang graph after the entry point
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

def build_graph(context: dict, join_keys: list, where_details: list, model_name: str, provider: str, api_key: str) -> dict:
    """
        This python function is the main entry point to the program which is called by dataset.py.
        Additionally we would also cache the graph inorder to prevent it from rebuilding each time.
    """
    global _cached_graph
    global _model_name
    global _api_key
    global _provider

    _model_name = model_name
    _api_key = api_key
    _provider = provider

    if _cached_graph is None:
        _cached_graph = build_graph_pipleine()

    # Running the langgraph for each query element
    langgraph_output = _cached_graph.invoke({
        "sql_column_details": context,
        "join_keys": join_keys,
        "where_details": where_details,
        "table_names": [],
        "table_relationships": [],
        "join_keys_relationships": [],
        "data_graph": {},
        "error": None
    })

    return langgraph_output