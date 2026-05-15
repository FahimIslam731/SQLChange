"""
    This python file is the maestro for extracting the individual queries from the csv file 
    building multations for each of the query and building the sql pairs

    Author: Dev Rathod
    Date: 05/11/2026
    File Name: dataset_loader.py
"""

import pandas as pd
from mutation_engine import(match_sql_to_mutation, mutation_function_mapping, mutation_function_occurances)
from parser import validate_sql_columns, parse_sql, get_join_keys, get_where_details
from graph_representer import build_graph

def build_mutation_maps(csv_filename: str, model_name: str, provider: str, api_key: str):
    """
        Python funciton to create different mutations for specific sql squeries on individual sql 
        dataframes from the csv file. This functions analyses queries and returning a list of mutation
        and original pairs
    """
    try:
        csv_dataset = pd.read_csv(csv_filename)
        # Storing the original and mutated pairs of the string 
        sql_mutated_pairs = []

        # Storing the mutated pairs in a dictionary for getting the total counts
        counts_mutation = {}
        # Initializing all the possible entries to 9 
        for mutation in mutation_function_mapping:
            counts_mutation[mutation] = 0

        # Iterrating though all the rows in the sql database
        for _, rows in csv_dataset.iterrows():
            # Finding all the inidivudal entries from the "sql" column
            individual_sql_commands = rows["sql"]

            # Load all the mutations that can be made for the input string
            applicable_multations = match_sql_to_mutation(individual_sql_commands)

            # Getting all the join keys relationships for the sql queries
            join_keys = get_join_keys(individual_sql_commands)

            # Getting all the where dependencies inside the sql query
            where_details = get_where_details(individual_sql_commands)

            # Builidng the context for the individual sql string
            context = parse_sql(rows["sql_context"])

            # Building the entity relationship graph for the query 
            er_graph = build_graph(context, join_keys, where_details, model_name, provider, api_key)

            # Iterrating through all the mutation types applicable for the sql strings
            for mutation in applicable_multations:
                # To limit the number of mutations in the dataset check if we don't exceed the individual limts
                if counts_mutation[mutation] >= mutation_function_occurances[mutation]:
                    continue
                else:
                    # Applind the mutations to the sql strings 
                    modified_sql_query = mutation_function_mapping[mutation](individual_sql_commands)

                    # If mutation function failed or mutations not been made perfectly skip the entry
                    if modified_sql_query is None or modified_sql_query.strip() == individual_sql_commands.strip() or not validate_sql_columns(modified_sql_query, context, individual_sql_commands):
                        continue

                    # Appending the mutated string to the specific pairs
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
                        "er_graph": er_graph["data_graph"],
                        "risk_label": None,
                        "performance_label": None,
                        "semantic_label": None
                    })

                    # Incrementing the mutation type after appending the original and mutated pairs
                    counts_mutation[mutation]+= 1
        
        print(f"Total pairs generated: {len(sql_mutated_pairs)}")
        for mutation_type, count in counts_mutation.items():
            target = mutation_function_occurances[mutation_type]
            status = "✓" if count >= target else f"⚠ target was {target}"
            print(f"  {mutation_type:<20} {count:>4}  {status}")

        return sql_mutated_pairs

    except Exception:
        print(f"Error: dataset_loader.py: Error while building mutations for {csv_filename} file")
        return {}