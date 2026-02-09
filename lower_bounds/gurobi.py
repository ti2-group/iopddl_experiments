from collections import defaultdict
import json
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB
import os


def read_problem(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    problem = {
        "name": data["problem"]["name"],
        "nodes": [],
        "edges": [],
        "usage_limit": data["problem"].get("usage_limit", None),
        "max_interval": 0,
    }
    max_interval = 0

    nodes = data["problem"]["nodes"]
    for node_interval in nodes["intervals"]:
        problem["nodes"].append({"interval": (node_interval[0], node_interval[1])})
        max_interval = max(max_interval, node_interval[1])

    for node_idx in range(len(problem["nodes"])):
        node = problem["nodes"][node_idx]
        costs = nodes["costs"][node_idx]
        usages = nodes["usages"][node_idx]
        node["strategies"] = []
        for strategy_idx in range(len(costs)):
            node["strategies"].append(
                {
                    "cost": costs[strategy_idx],
                    "usage": (
                        usages[strategy_idx]
                        if problem["nodes"][node_idx]["interval"][0]
                        < problem["nodes"][node_idx]["interval"][1]
                        else 0
                    ),
                }
            )

    edges = data["problem"]["edges"]
    for node_list in edges["nodes"]:
        edge = {"nodes": node_list}
        # for node_idx in node_list:
        #     edge["nodes"].append(node_idx)
        problem["edges"].append(edge)

    for edge_idx in range(len(problem["edges"])):
        edge = problem["edges"][edge_idx]
        edge["strategies"] = []
        for cost in edges["costs"][edge_idx]:
            edge["strategies"].append(cost)

    problem["max_interval"] = max_interval
    return problem


problems = sorted(os.listdir("./benchmarks"))

instance_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

filename = f"benchmarks/{problems[instance_id]}"

print(filename, flush=True)
problem = read_problem(filename)

n = len(problem["nodes"])

print("=> Calculating concurrent sets", flush=True)

intervals = [
    (id, node["interval"])
    for id, node in enumerate(problem["nodes"])
    if node["interval"][0] != node["interval"][1]
]
intervals.sort(key=lambda x: x[1][0])
starts = set([interval[0] for id, interval in intervals])
ends = set([interval[1] for id, interval in intervals])

all_times = starts.union(ends)

concurrentSets = []
intervals = [
    (id, node["interval"])
    for id, node in enumerate(problem["nodes"])
    if node["interval"][0] != node["interval"][1]
]
intervals.sort(key=lambda x: x[1][0])
starts = [(id, interval[0]) for id, interval in intervals]
ends = [(id, interval[1]) for id, interval in intervals]
ends.sort(key=lambda x: x[1])
activeNodes = []
concurrentSets = []
startPointer = 0
endPointer = 0
status = "adding"
# print(intervals)
# Add all starting at zero
# while startPointer < len(intervals) and starts[startPointer][1] == 0:
#     activeNodes.append(starts[startPointer][0])
#     startPointer += 1

while startPointer < len(intervals) and endPointer < len(intervals):
    if starts[startPointer][1] < ends[endPointer][1]:
        activeNodes.append(starts[startPointer][0])
        startPointer += 1
        while (
            startPointer < len(intervals)
            and starts[startPointer][1] == starts[startPointer - 1][1]
        ):
            activeNodes.append(starts[startPointer][0])
            startPointer += 1
        if status == "removing":
            status = "adding"
    else:
        if status == "adding":
            concurrentSets.append(
                (activeNodes.copy(), (starts[startPointer - 1][1], ends[endPointer][1]))
            )
            status = "removing"
        activeNodes.remove(ends[endPointer][0])
        endPointer += 1
concurrentSets.append((activeNodes.copy(), (starts[-1][1], ends[-1][1])))

sets = list(set(frozenset(cs) for cs, time in concurrentSets))

print("Number of intervals", len(intervals))
print("Number of concurrent sets", len(sets), flush=True)
node_tuples, node_costs = gp.multidict(
    {
        (nodeIdx, stratIdx): strat["cost"]
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

print("=> Creating model", flush=True)
m = gp.Model("iopddl")
print("=> Adding variables", flush=True)
nodes_strats = m.addVars(
    node_tuples, vtype=GRB.BINARY, obj=node_costs, name="nodes_strats"
)
edges = m.addVars(edge_tuples, vtype=GRB.BINARY, obj=edge_costs, name="edges")

print("=> Add unique strategy constraints", flush=True)
m.addConstrs(
    (nodes_strats.sum(nodeIdx, "*") == 1 for nodeIdx in range(len(problem["nodes"])))
)

print("=> Add edge constraints", flush=True)
for nodeIdx1, nodeIdx2, stratIdx1, stratIdx2 in tqdm(edge_tuples):
    m.addConstr(
        edges[nodeIdx1, nodeIdx2, stratIdx1, stratIdx2]
        >= nodes_strats[nodeIdx1, stratIdx1] + nodes_strats[nodeIdx2, stratIdx2] - 1
    )

print("=> Add usage limit constraints", flush=True)
# sets = [tuple(concurrentSet) for concurrentSet, interval in concurrentSets]
for setIdx, set in tqdm(enumerate(sets)):
    m.addConstr(
        nodes_strats.prod(node_usages, set, "*") <= problem["usage_limit"],
    )
m.update()

m.setParam("TimeLimit", 3600 * 24 * 14)
m.setParam("Threads", 8)

print("=> Optimizing model")
m.optimize()
