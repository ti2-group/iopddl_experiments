from collections import defaultdict
import numpy as np
from fnnls import fnnls
from helpers import (
    read_problem,
    get_concurrent_intervals,
    get_free_usage_per_set,
    solve,
)
import os

if __name__ == "__main__":

    instance_id = 22
    instance_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", instance_id))
    chars = [chr(c) for c in range(ord("A"), ord("Y") + 1)]

    instance_name = f"benchmarks/asplos-2025-iopddl-{chars[instance_id]}.json"

    penalties_file = f"{chars[instance_id]}_{instance_id}_{7}_lstsq_penalties_2.npy"

    print("Reading instance", instance_name, flush=True)
    problem, node_max_usages, node_min_usages = read_problem(instance_name)
    print(
        "Problem",
        problem["name"],
        "max_interval",
        problem["max_interval"],
        "usage_limit",
        problem["usage_limit"],
        "num_nodes",
        len(problem["nodes"]),
        "num_edges",
        len(problem["edges"]),
        flush=True,
    )

    concurrentSets = get_concurrent_intervals(problem)
    print("Number of concurrent sets", len(concurrentSets), flush=True)

    sets_by_var = defaultdict(list)
    for cs_idx, concurrentSet in enumerate(concurrentSets):
        for var in concurrentSet:
            sets_by_var[var].append(cs_idx)

    A = np.zeros((len(problem["nodes"]), len(concurrentSets)), dtype=int)
    for cs_idx, concurrentSet in enumerate(concurrentSets):
        for var in concurrentSet:
            A[var, cs_idx] = 1
    print("Ready matrix A", A.shape, flush=True)
    b = np.fromfile(
        f"best_node_usage_weights_{chars[instance_id]}.bin", dtype=np.float64
    )
    print("Ready vector b", b.shape, flush=True)

    penalties, res = fnnls(A, b)
    print("Lstsq penalties calculated", res, flush=True)
    np.save(penalties_file, penalties)
