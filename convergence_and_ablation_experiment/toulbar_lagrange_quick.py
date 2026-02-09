import datetime
import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict

import numpy as np
import pytoulbar2
from joblib import Parallel, delayed


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

    node_max_usages = [0] * len(problem["nodes"])
    node_min_usages = [0] * len(problem["nodes"])

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
        node_max_usage = max([strategy["usage"] for strategy in node["strategies"]])
        node_max_usages[node_idx] = node_max_usage
        node_min_usage = min([strategy["usage"] for strategy in node["strategies"]])
        node_min_usages[node_idx] = node_min_usage

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
    return problem, node_max_usages, node_min_usages


def evaluate(problem, solution, concurrentSets=None):
    if len(solution) != len(problem["nodes"]):
        raise ValueError("Incorrect solution size")

    max_time = 0
    for node in problem["nodes"]:
        max_time = max(max_time, node["interval"][1])

    cost = 0
    total_usages = [0] * max_time

    for node_idx, node in enumerate(problem["nodes"]):
        strategy_idx = solution[node_idx]
        if strategy_idx < 0 or strategy_idx >= len(node["strategies"]):
            raise IndexError("Invalid strategy index")
        cost += node["strategies"][strategy_idx]["cost"]
        for t in range(node["interval"][0], node["interval"][1]):
            total_usages[t] += node["strategies"][strategy_idx]["usage"]

    for edge in problem["edges"]:
        strategy_idx = 0
        for node_idx in edge["nodes"]:
            strategy_idx *= len(problem["nodes"][node_idx]["strategies"])
            strategy_idx += solution[node_idx]
        cost += edge["strategies"][strategy_idx]  # ["cost"]

    broke_limit = 0
    if "usage_limit" in problem and problem["usage_limit"] is not None:
        for time, total_usage in enumerate(total_usages):
            if total_usage > problem["usage_limit"]:
                broke_limit += 1
                # print("Usage limit exceeded at time", time, flush=True)

    brokenSets = []
    diffs = []
    if concurrentSets is not None:
        for cs_idx, concurrentSet in enumerate(concurrentSets):
            usage_in_set = 0
            for node_idx in concurrentSet:
                strategy_idx = solution[node_idx]
                usage_in_set += problem["nodes"][node_idx]["strategies"][strategy_idx][
                    "usage"
                ]
            if usage_in_set > problem["usage_limit"]:
                brokenSets.append(cs_idx)

            diffs.append(
                usage_in_set - problem["usage_limit"]
            )  # positive if over the limit

    # print("Broken limits", broke_limit, flush=True)
    print("Broken sets", len(brokenSets), f"cost {cost:.6e}", flush=True)
    if broke_limit > 0:
        assert len(brokenSets) > 0
    return cost, brokenSets, diffs


# Replace the argparse section with the following:
if len(sys.argv) > 1:
    instance_id = int(sys.argv[1])  # Read the instance ID from the command line
else:
    instance_id = ord("G") - ord("A")  # Default value if no argument is provided


instance_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", instance_id))

global_timeouts = {
    "A": 60,
    "B": 60,
    "C": 60,
    "D": 60,
    "E": 60,
    "F": 120,
    "G": 120,
    "H": 120,
    "I": 120,
    "J": 120,
    "K": 180,
    "L": 180,
    "M": 180,
    "N": 180,
    "O": 180,
    "P": 240,
    "Q": 240,
    "R": 240,
    "S": 240,
    "T": 240,
    "U": 300,
    "V": 300,
    "W": 300,
    "X": 300,
    "Y": 300,
}


# instance_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", instance_id))
chars = [chr(c) for c in range(ord("A"), ord("Y") + 1)]


seeds = [1, 43, 1457, 789, 2468, 3579, 9876, 5432, 6543, 8765]
instance_name = f"benchmarks/asplos-2025-iopddl-{chars[instance_id]}.json"
instance_short_name = chars[instance_id]
global_timeout = global_timeouts[instance_short_name]

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

print("Number of intervals", len(intervals), flush=True)
print("Number of concurrent sets", len(concurrentSets), flush=True)

cs_lens = [len(cs) for cs, interval in concurrentSets]
print(
    f"Concurrent set lengths min: {min(cs_lens)}, max: {max(cs_lens)}, mean: {np.mean(cs_lens)}, median: {np.median(cs_lens)}",
    flush=True,
)
concurrentSets = [cs for cs, interval in concurrentSets]

print("Calculating free usage per concurrent set", flush=True)
free_usage_per_set = np.zeros(len(concurrentSets), dtype=np.float64)
for cs_idx, concurrentSet in enumerate(concurrentSets):
    free_usage_per_set[cs_idx] = float(problem["usage_limit"]) - sum(
        node_min_usages[node_idx] for node_idx in concurrentSet
    )
print("Done calculating free usage per concurrent set", flush=True)

sets_by_var = defaultdict(list)
for cs_idx, concurrentSet in enumerate(concurrentSets):
    for var in concurrentSet:
        sets_by_var[var].append(cs_idx)


n = len(problem["nodes"])


def solve(penalties, ub=None, constraints=None, time_limit=20, seed=1):
    # cfn = pytoulbar2.CFN(vac=0, verbose=0, ubinit=1e17, vns=-2)
    ub_init = None
    if ub is not None:
        ub_init = ub + float(problem["usage_limit"]) * float(np.sum(penalties))
    cfn = pytoulbar2.CFN(vac=0, verbose=-1, ubinit=ub_init, seed=seed)

    try:
        penalty_term = -free_usage_per_set.dot(penalties)
        cfn.AddFunction([], [penalty_term])
        for n_idx, node in enumerate(problem["nodes"]):
            min_usage_costs = [
                strategy["cost"]
                + penalties[sets_by_var[n_idx]].sum()
                * (strategy["usage"] - node_min_usages[n_idx])
                for strategy in node["strategies"]
            ]
            min_usage_costs = [min(cost, 8e18) for cost in min_usage_costs]
            cfn.AddVariable(str(n_idx), list(range(len(min_usage_costs))))
            cfn.AddFunction([str(n_idx)], min_usage_costs)

        for edge in problem["edges"]:
            node1, node2 = edge["nodes"]
            costs = [min(cost, 8e18) for cost in edge["strategies"]]
            cfn.AddFunction([node1, node2], costs)

        if constraints is not None:
            print("Add additional constraints", flush=True)
            for constraint in constraints:
                params = str(-problem["usage_limit"])
                cs = concurrentSets[constraint]
                for var in cs:
                    strats = problem["nodes"][var]["strategies"]
                    vtuples = [
                        str(j) + " " + str(-strat["usage"])
                        for j, strat in enumerate(strats)
                        if strat["usage"] > 0
                    ]
                    params += " " + str(len(vtuples)) + " "
                    params += " ".join([e for e in vtuples])
                cfn.CFN.wcsp.postKnapsackConstraint(list(cs), params, kp=True)

        print("Solving CFN...", flush=True)
        sol, cfn_obj, *r = cfn.Solve(timeLimit=time_limit)
        lower_bound_cfn = cfn.GetDDualBound()
        print("Done solving", flush=True)

        # The cfn objective should equal f(x) + sum_s lambda_s * usage_in_set_s(x)
        dual_cost_ub = float(cfn_obj)  # - penalty_term

        dual_cost_lb = float(lower_bound_cfn)  # - penalty_term
        costs, brokenSets, diffs = evaluate(problem, sol, concurrentSets=concurrentSets)
        diffs = np.array(diffs, dtype=np.float64)
        print(
            f"Costs {float(costs):.6e} Dual cost upper bound {float(dual_cost_ub):.6e}, dual cost lower bound {float(dual_cost_lb):.6e}, gap {float(dual_cost_ub - dual_cost_lb):.6e} ({(float(dual_cost_ub - dual_cost_lb) / float(dual_cost_ub)) * 100:.6f}%)",
            flush=True,
        )

        return costs, dual_cost_lb, diffs, brokenSets
    except Exception as e:
        print("Solver error:", e, flush=True)
        return math.inf, -math.inf, None, list(range(len(concurrentSets)))


def _write_results_db(
    db_path, results_list, instance_name=None, max_attempts=5, backoff=0.5
):
    """Write results_list (a list of dicts) into a sqlite DB at db_path.

    Retries on sqlite3.OperationalError when the database is locked.
    """
    if instance_name is None:
        instance_name = globals().get("instance_name", "")

    for attempt in range(1, max_attempts + 1):
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            cur = conn.cursor()
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS toulbar_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_name TEXT,
                ts TEXT,
                iteration TEXT,
                primal_UB REAL,
                dual_LB REAL,
                gap REAL,
                gap_percent REAL,
                best_broken REAL,
                seed INTEGER
            )
            """
            )

            insert_sql = (
                "INSERT INTO toulbar_results (instance_name, ts, iteration, primal_UB, dual_LB, gap, gap_percent, best_broken,seed)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )

            for r in results_list:
                cur.execute(
                    insert_sql,
                    (
                        instance_name,
                        datetime.datetime.utcnow().isoformat(),
                        str(r.get("iteration")),
                        (
                            None
                            if r.get("primal_UB") is None
                            else float(r.get("primal_UB"))
                        ),
                        None if r.get("dual_LB") is None else float(r.get("dual_LB")),
                        None if r.get("gap") is None else float(r.get("gap")),
                        (
                            None
                            if r.get("gap_percent") is None
                            else float(r.get("gap_percent"))
                        ),
                        r.get("best_broken"),
                        r.get("seed"),
                    ),
                )

            conn.commit()
            conn.close()
            print(f"Wrote {len(results_list)} rows to {db_path}", flush=True)
            return True

        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "locked" in msg or "database is locked" in msg:
                if attempt < max_attempts:
                    wait = backoff * (2 ** (attempt - 1))
                    print(
                        f"Database locked, retrying in {wait:.2f}s (attempt {attempt}/{max_attempts})",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                else:
                    print(
                        f"Failed to write DB after {max_attempts} attempts due to lock: {e}",
                        flush=True,
                    )
                    return False
            else:
                print("SQLite OperationalError:", e, flush=True)
                return False
        except Exception as e:
            print("Unexpected error when writing DB:", e, flush=True)
            return False


def batch_solve(
    penalties_batch, primal_UB, dual_LB, batch_scalings=None, time_limit=20, seed=1
):
    with Parallel(n_jobs=8) as parallel:
        results = parallel(
            delayed(solve)(
                penalties_batch[i], primal_UB, time_limit=time_limit, seed=seed
            )
            for i in range(len(penalties_batch))
        )
    improved = False
    penalties = None
    diffs = None
    best_broken = None
    for b_idx, (primal, dual, b_diffs, brokenSets) in enumerate(results):
        scale = ""
        if batch_scalings is not None:
            scale = batch_scalings[b_idx % len(batch_scalings)]

        print(f"--- Batch scale {scale} ({b_idx})---", flush=True)
        if primal < primal_UB and len(brokenSets) == 0:
            primal_UB = primal
            print("Updated primal UB", flush=True)
        if dual > dual_LB:
            dual_LB = dual
            improved = True
            best_broken = brokenSets
            penalties = batch_penalties[b_idx].copy()
            diffs = b_diffs
            if b_idx == 7:
                for i in range(len(batch_scalings)):
                    batch_scalings[i] *= 10
            print("Updated dual LB", flush=True)
    return primal_UB, dual_LB, improved, penalties, diffs, best_broken


beta = 0.05

penalties = np.full(len(concurrentSets), 1 / len(concurrentSets), dtype=np.float64)
batch_penalties = [penalties * s for s in [0.001, 0.1, 0.5, 2.5, 10.0, 50, 100, 1000]]


for seed in seeds:
    print("=====> Processing seed ", seed)
    primal_UB = math.inf
    dual_LB = -math.inf

    print("Initial solve", flush=True)

    primal_UB, dual_LB, improved, penalties, diffs, best_broken = batch_solve(
        batch_penalties, primal_UB, dual_LB, time_limit=300, seed=seed
    )

    print(min(diffs), max(diffs), np.mean(diffs), flush=True)
    print(min(penalties), max(penalties), np.mean(penalties), flush=True)

    momentum = 0.6
    second_momentum = 0.9

    diffs_list = []

    batch_scalings = [0.001, 0.1, 1, 2.5, 10.0, 50, 100, 1000]
    m_t = np.zeros(len(concurrentSets))
    v_t = np.zeros(len(concurrentSets))

    time_limit_per_iter = 12
    if instance_short_name == "W":
        time_limit_per_iter = 30

    max_timeout = time_limit_per_iter * 2 + 1
    used_time = 0

    overall_best_dual = dual_LB

    print(
        f"Intial solve, primal_UB {primal_UB:.6e}, overall best dual_LB {overall_best_dual:.6e}, gap {(primal_UB - overall_best_dual):.6e} ({(primal_UB - overall_best_dual) / primal_UB * 100:.6f}%)",
        flush=True,
    )

    results = [
        {
            "iteration": "inital",
            "primal_UB": primal_UB,
            "dual_LB": overall_best_dual,
            "gap": primal_UB - overall_best_dual,
            "gap_percent": (primal_UB - overall_best_dual) / primal_UB * 100,
            "best_broken": len(best_broken),
            "seed": seed,
        }
    ]

    db_file = os.environ.get(
        "RESULTS_DB", os.path.join(os.getcwd(), "quick_results.db")
    )

    for i in range(100):
        print("===> Iteration", i, flush=True)
        m_t = momentum * m_t + (1 - momentum) * diffs
        norm_g2 = np.dot(m_t, m_t)

        step = beta * (primal_UB - dual_LB) / norm_g2
        step = max(step, 0.0)  # avoid negative

        print(f"Norm g^2 {norm_g2:.6e}, step size {step:.6e}", flush=True)

        # Adjust penalties

        batch_penalties = []

        for scale in batch_scalings:
            new_penalty = penalties + step * scale * 1 * m_t
            new_penalty = np.maximum(new_penalty, 0)

            batch_penalties.append(new_penalty)

        start = time.time()
        primal_UB, dual_LB, improved, new_penalties, new_diffs, new_best_broken = (
            batch_solve(
                batch_penalties,
                primal_UB,
                dual_LB,
                batch_scalings=batch_scalings,
                time_limit=time_limit_per_iter,
            )
        )
        end = time.time()
        real_time = end - start
        if real_time < time_limit_per_iter:
            used_time += real_time
        else:
            used_time += time_limit_per_iter

        if dual_LB > overall_best_dual:
            overall_best_dual = dual_LB

        if improved:
            penalties = new_penalties
            diffs = new_diffs
            best_broken = new_best_broken

        print(min(diffs), max(diffs), np.mean(diffs), flush=True)
        print(min(penalties), max(penalties), np.mean(penalties), flush=True)

        print(
            f"===> End of iteration {i}, primal_UB {primal_UB:.6e}, overall best dual_LB {overall_best_dual:.6e}, gap {(primal_UB - overall_best_dual):.6e} ({(primal_UB - overall_best_dual) / primal_UB * 100:.6f}%)",
            f"===> End of iteration {i}, primal_UB {primal_UB:.6e}, overall best dual_LB {overall_best_dual:.6e}, gap {(primal_UB - overall_best_dual):.6e} ({(primal_UB - overall_best_dual) / primal_UB * 100:.6f}%)",
            flush=True,
        )
        if not improved:
            print("No improvement in this iteration, stopping", flush=True)
            time_limit_per_iter = math.floor(1.5 * time_limit_per_iter)
            time_limit_per_iter = min(time_limit_per_iter, max_timeout)
            batch_scalings = [s / 10 for s in batch_scalings]
        else:
            time_limit_per_iter += 1
            if np.any(np.abs(penalties) > 1e12):
                raise RuntimeError(
                    "Penalties exploded — check step size or sign conventions"
                )

        if used_time < global_timeout:
            results.append(
                {
                    "iteration": i,
                    "primal_UB": primal_UB,
                    "dual_LB": overall_best_dual,
                    "gap": primal_UB - overall_best_dual,
                    "gap_percent": (primal_UB - overall_best_dual) / primal_UB * 100,
                    "best_broken": len(best_broken),
                    "seed": seed,
                }
            )
        else:
            break
        try:
            _write_results_db(db_file, [results[-1]], instance_name=instance_name)
        except Exception as e:
            print("Failed to write results to DB:", e, flush=True)
