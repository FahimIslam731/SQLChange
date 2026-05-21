from synthetic_db import run_query_pair


def check_equivalence(record, seed=0, rows_per_table=50):
    result = run_query_pair(record, seed=seed, rows_per_table=rows_per_table)
    relation = result["comparison"]["output_relation"]
    return {
        "equivalent": relation == "identical",
        "output_relation": relation,
        "row_count_original": result["comparison"]["row_count_original"],
        "row_count_modified": result["comparison"]["row_count_modified"],
    }
