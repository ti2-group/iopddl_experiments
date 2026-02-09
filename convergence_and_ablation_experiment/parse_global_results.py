import os
import re
from collections import defaultdict


def parse_log_file(file_path):
    # Regular expressions to match the lines
    # Cost: 1.36000000e+20 (greedy) 0.05 s
    valid_solution_re = re.compile(
        rf"# Cost: ([\d\.e\+]+) \((.*?)\) ([\d\.]+) s \(?iteration: (\d+)\)?"
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
            cost = float(valid_solution_match.group(1))
            method = str(valid_solution_match.group(2))
            time = float(valid_solution_match.group(3))
            iteration = int(valid_solution_match.group(4))

            results.append(
                {
                    "cost": f"{cost:.8e}",
                    "time": time,
                    "method": method,
                    "iteration": iteration,
                }
            )

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
                        if "global" not in output_file or "bin" in output_file:
                            continue
                        print(f"Processing {output_file_path}...")
                        parsed_results = parse_log_file(output_file_path)
                        print(f"Found {len(parsed_results)} entries")
                        for r in parsed_results:
                            results.append(
                                {
                                    "instance": output_file[0],
                                    "seed": seed,
                                    "cost": r["cost"],
                                    "time": r["time"],
                                    "method": r["method"],
                                    "iteration": r["iteration"],
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

with open("global_convergence.json", "w") as fp:
    json.dump(all_results, fp)
