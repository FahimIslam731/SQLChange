"""
    Command line interface for generating the SQLChange dataset.
    Takes user inputs for CSV path, LLM provider, model name, and API key.
    Outputs the final dataset as a JSON file.

    Author: Dev Rathod
    Date: 05/13/2026
    File Name: cli.py
"""

import os
import sys
import json
import argparse
from dataset_loader import build_mutation_maps


def main():
    """
        Main entry point for the SQLChange dataset generation pipeline.
        Parses command line arguments and runs the full pipeline.
    """
    # Setting up argument parser for command line inputs
    parser = argparse.ArgumentParser(
        description="SQLChange Dataset Generator — generates mutation pairs from SQL queries"
    )

    # Required arguments
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the source queries CSV file"
    )

    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["anthropic", "openai", "local"],
        help="LLM provider to use for schema inference (anthropic, openai, or local)"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for inference (e.g. claude-sonnet-4-20250514, gpt-4o, llama3)"
    )

    # Optional arguments
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the LLM provider (can also be set via environment variable)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/sqlchange_dataset.json",
        help="Output path for the generated dataset JSON (default: data/sqlchange_dataset.json)"
    )

    # Parsing the command line arguments
    args = parser.parse_args()

    # Resolving the API key from argument or environment variable
    api_key = args.api_key
    if api_key is None:
        if args.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif args.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif args.provider == "local":
            api_key = ""  # local Ollama doesn't need an API key

    # Checking if the API key was resolved
    if api_key is None and args.provider != "local":
        print(f"Error: No API key provided for {args.provider}")
        print(f"Either pass --api-key or set the environment variable:")
        print(f"  export ANTHROPIC_API_KEY=sk-ant-...")
        print(f"  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Checking if the CSV file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    # Printing the configuration before running
    print("=" * 60)
    print("SQLChange Dataset Generator")
    print("=" * 60)
    print(f"  CSV file:    {args.csv}")
    print(f"  Provider:    {args.provider}")
    print(f"  Model:       {args.model}")
    print(f"  Output:      {args.output}")
    print(f"  API key:     {'set' if api_key else 'not set'}")
    print("=" * 60)

    # Running the mutation pipeline
    print("\nGenerating mutation pairs...")
    dataset = build_mutation_maps(args.csv, args.model, args.provider, api_key)

    # Checking if the pipeline returned valid results
    if not dataset:
        print("Error: No pairs generated. Check your CSV file and error messages above.")
        sys.exit(1)

    # Creating the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Saving the dataset to a JSON file
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2, default=str)

    print(f"\nDataset saved to: {args.output}")
    print(f"Total pairs: {len(dataset)}")
    print("\nDone!")


if __name__ == "__main__":
    main()