from collections import defaultdict
import json
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB
import os
import numpy as np
from helpers import (
    evaluate,
    read_problem,
    get_concurrent_intervals,
    get_free_usage_per_set,
    solve,
)

instance_id = 10
instance_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", instance_id))
chars = [chr(c) for c in range(ord("A"), ord("Y") + 1)]

instance_name = f"benchmarks/asplos-2025-iopddl-{chars[instance_id]}.json"

penalties_file = f"{chars[instance_id]}_{instance_id}_{14}_penalties.npy"

print(penalties_file, flush=True)

with open(penalties_file, "rb") as file:
    penalties = np.load(file)

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

free_usage_per_set = get_free_usage_per_set(concurrentSets, problem, node_min_usages)


n = len(problem["nodes"])

node_tuples, node_costs = gp.multidict(
    {
        (nodeIdx, stratIdx): strat["cost"]
        + penalties[sets_by_var[nodeIdx]].sum() * (strat["usage"])
        for nodeIdx, node in enumerate(problem["nodes"])
        for stratIdx, strat in enumerate(node["strategies"])
    }
)

node_usages = {
    (nodeIdx, stratIdx): strat["usage"]
    for nodeIdx, node in enumerate(problem["nodes"])
    for stratIdx, strat in enumerate(node["strategies"])
}

edge_tuples, edge_costs = gp.multidict(
    {
        (
            edge["nodes"][0],
            edge["nodes"][1],
            (stratIdx) // len(problem["nodes"][edge["nodes"][1]]["strategies"]),
            (stratIdx) % len(problem["nodes"][edge["nodes"][1]]["strategies"]),
        ): strat
        for edge in problem["edges"]
        for stratIdx, strat in enumerate(edge["strategies"])
    }
)


print(
    max(
        min([strat["cost"] for strat in node["strategies"]])
        for node in problem["nodes"]
    ),
    flush=True,
)

print("=> Creating model", flush=True)
m = gp.Model("iopddl")
print("=> Adding variables", flush=True)
nodes_strats = m.addVars(
    node_tuples, vtype=GRB.BINARY, obj=node_costs, name="nodes_strats"
)

print(
    "Offset",
)
m.addVar(
    lb=1,
    vtype=GRB.BINARY,
    obj=-(penalties.sum() * problem["usage_limit"]),
    name="lagrange offset",
)
edges = m.addVars(edge_tuples, vtype=GRB.BINARY, obj=edge_costs, name="edges")

print("=> Add unique strategy constraints", flush=True)
m.addConstrs(
    (nodes_strats.sum(nodeIdx, "*") == 1 for nodeIdx in range(len(problem["nodes"])))
)


# print("=> Add edge constraints", flush=True)
for nodeIdx1, nodeIdx2, stratIdx1, stratIdx2 in tqdm(edge_tuples):
    m.addConstr(
        edges[nodeIdx1, nodeIdx2, stratIdx1, stratIdx2]
        >= nodes_strats[nodeIdx1, stratIdx1] + nodes_strats[nodeIdx2, stratIdx2] - 1
    )


print("=> Add usage limit constraints", flush=True)

toulbar_solution_file = f"{chars[instance_id]}_{instance_id}_{14}_restart_solution.npy"
toulbar_sol = np.load(toulbar_solution_file)
m.NumStart = 1
m.update()
m.params.StartNumber = 0
for v in m.getVars():
    if v.VarName.startswith("nodes_strats"):
        parts = v.VarName.split("[")[1].rstrip("]").split(",")
        nodeIdx = int(parts[0])
        stratIdx = int(parts[1])
        if stratIdx == toulbar_sol[nodeIdx]:
            # print("Set", v.VarName, "to", 1, flush=True)
            v.Start = 1
        else:
            v.Start = 0
            # print("Set", v.VarName, "to", 0, flush=True)
        # print("Set", v.VarName, "to", v.Start, toulbar_sol[nodeIdx], flush=True)
# sets = [tuple(concurrentSet) for concurrentSet, interval in concurrentSets]
# for setIdx, set in tqdm(enumerate(sets)):
#     m.addConstr(
#         nodes_strats.prod(node_usages, set, "*") <= problem["usage_limit"],
#     )
m.update()

m.setParam("TimeLimit", 3600 * 24 * 7)
m.setParam("Threads", 48)

print("=> Optimizing model")
m.optimize()


print("=> Optimization complete", flush=True)
print(f"Final solution cost: {m.ObjVal:.6e}", flush=True)
# List variable assignments
solution = [-1] * len(problem["nodes"])
for v in m.getVars():
    if v.X > 0.5 and v.VarName.startswith("nodes_strats"):
        parts = v.VarName.split("[")[1].rstrip("]").split(",")
        nodeIdx = int(parts[0])
        stratIdx = int(parts[1])
        solution[nodeIdx] = stratIdx

# print(solution, flush=True)

# write solution to file
solution_file = f"{chars[instance_id]}_{instance_id}_{14}_gurobi_solution.npy"
np.save(solution_file, solution)

costs, broken, diffs = evaluate(problem, solution, concurrentSets)

# print(penalties, diffs)
# print(len(broken))
print(
    f"Evaluated solution cost: {float(costs):.6e}, {np.array(diffs).dot(penalties):.6e}, {(np.array(diffs).dot(penalties) + float(costs)):.6e}, {len(broken)}",
    flush=True,
)
