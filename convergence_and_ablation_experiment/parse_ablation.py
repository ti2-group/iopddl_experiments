import os
import re
from collections import defaultdict


def parse_log_file(file_path):
    # Regular expressions to match the lines
    # Cost: 1.36000000e+20 (greedy) 0.05 s
    valid_solution_re = re.compile(
        r"# Solver time:(\d+\.\d+), Greedy time: (\d+\.\d+), before greedy: (\d+), after greedy: (\d+) in iteration: (\d+) after: (\d+\.\d+) s"
    )

    # Read the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse the lines
    results = []
    for i in range(len(lines) - 1):
        if lines[i].startswith("["):
            continue
        valid_solution_match = valid_solution_re.match(lines[i])
        if valid_solution_match:
            result = {
                "solver_time": float(valid_solution_match.group(1)),
                "greedy_time": float(valid_solution_match.group(2)),
                "before_greedy": int(valid_solution_match.group(3)),
                "after_greedy": int(valid_solution_match.group(4)),
                "iteration": int(valid_solution_match.group(5)),
                "time": float(valid_solution_match.group(6)),
            }
            results.append(result)

    return results


# Example usage


def read_all_results(base_path):
    results = []
    version_path = base_path
    if os.path.isdir(version_path):
        for seed in os.listdir(version_path):
            if not seed.startswith("s"):
                continue
            seed_path = os.path.join(version_path, seed)
            if os.path.isdir(seed_path):
                for output_file in os.listdir(seed_path):
                    output_file_path = os.path.join(seed_path, output_file)
                    if os.path.isfile(output_file_path):
                        if "global" in output_file or "bin" in output_file:
                            continue
                        print(f"Processing {output_file_path}...")
                        parsed_results = parse_log_file(output_file_path)
                        print(f"Found {len(parsed_results)} entries")
                        for r in parsed_results:
                            if r["before_greedy"] > 1e17:
                                continue
                            results.append(
                                {
                                    "instance": output_file[0],
                                    "seed": seed,
                                    "solver_time": r["solver_time"],
                                    "greedy_time": r["greedy_time"],
                                    "before_greedy": r["before_greedy"],
                                    "after_greedy": r["after_greedy"],
                                    "iteration": r["iteration"],
                                    "time": r["time"],
                                }
                            )

    return results


def print_results(results):
    for output_file, seeds in results.items():
        print(f"Results for {output_file}:")
        for seed_data in seeds:
            seed = seed_data["seed"]
            parsed_results = seed_data["results"]

            print(f"    Seed {seed}:")
            for result in parsed_results:
                print(f"      {result}")


# Example usage
base_path = "./"
all_results = read_all_results(base_path)
# print(all_results)


import json

with open("ablation.json", "w") as fp:
    json.dump(all_results, fp)
