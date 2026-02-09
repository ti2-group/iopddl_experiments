# Supplement for Soft Constraints, Strong Solutions: Optimizing Intra-Operator Parallelism for Distributed Deep Learning


## Requirements

You will need to install uv: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)


Afterwards to install the requirements, run:

```bash
uv sync
```
To get the dependencies for the plots you will need LaTeX and run sync with `--all-groups`.

To run python files with the correct venv you can use `uv run <script.py>`

## Experiments

The competition results provided by the organizers are in the `iopddl_final_results.csv` file, which includes the scores for XLA.

The [benchmark instances](https://github.com/google/iopddl/tree/main/benchmarks) need to be downloaded into a benchmarks folder on the same level as this folder. The benchmarks are not included in the supplement zip due to their size. You can do that with the following commands:

```
wget https://github.com/google/iopddl/archive/refs/heads/main.zip
unzip main.zip
mv iopddl-main/benchmarks/ ./
gzip -d benchmarks/*
rm -r main.zip iopddl-main/
```


### Comparison with XLA

Uses the data provided by the contest organizers and the median score  for V and W obtained in the convergence experiments using the fixed solver.

### Convergence and ablation

All relevent files are in the `convergence_and_ablation` folder. All experiments use the same base data, 10 runs with different seeds for each instance. The solver was modified to output more data, the modified main.cpp as well as a zipped linux binary are in the solver folder. 
The experiments can be run with the provided bash scripts. 

For the convergence and ablation of the node specific penalties, run:
```bash
bash convergence_and_ablation.sh
```

For the convergence of the global penalty, run:
```bash
bash convergence_global_penalty.sh
```

The solver outputs are included in the seed folder, e.g. for seed 1 in s1. Additionally, include the scripts to parse the solver outputs as well as the parsed data. 

###  Lower bounds

The lower bounds were generated with the python scripts in the 
`lower_bounds` folder. They were primarily obtained using Toulbar2.

#### Toulbar2

To run lagrange gradient ascent with moment use:
```bash
uv run lower_bounds/toulbar_lagrange.py
```

The benchmark file is selected based on a SLURM environment variable, but can also be passed as cli argument.

#### Gurobi
To run gurobi you will need a license, which can be obtained for free for academic use.  

```
uv run lower_bounds/gurobi.py
```

The benchmark file is selected based on a SLURM environment variable, which is set in the `lower_bounds/gurobi.py` script.

To improve the lower bounds found by toulbar use:
```bash
uv run lower_bounds/gurobi_lagrange.py
```





## Solver

The source of the solver is in the `solver` folder, it includes another README.md with instructions on how to compile and run it. The solver is a C++ implementation of the algorithm described in the paper. 

The solver is accessible via a Dockerfile that builds and runs it efficiently. Build the Docker image in the supplement folder using:

```bash
docker build -t iopddl-solver .
```

### Solver Command-Line Usage

The solver binary (`iopddl`) accepts an input problem in JSON format, a timeout, and several optional parameters for advanced control.

**Usage:**
```sh
./iopddl <input_file_path> <timeout_in_seconds> [options]
```

**Positional Arguments:**
- `<input_file_path>`: Path to the input JSON file describing the problem instance.
- `<timeout_in_seconds>`: Timeout duration in seconds for the solver.

**Options:**
- `-h`        Show this help message and exit.
- `-s <seed>`    Set the random seed value (default: 1).
- `-t <timeoutWCSP>` Set the timeout in seconds for the internal WCSP solver (default: 12).
- `-j <numForks>`  Set the number of forks for parallelism (default: 8).
- `-q`        Quiet mode (do not print node strategies).


The internal solver timeout is the only tunable hyperparameter and may need to be increased for large instances.

**Example:**
```sh
./iopddl example.json 60 -s 42 -t 10 -j 4 -q
```

**Docker Example with Mounted Input:**
```sh
docker run --rm -v $(pwd):/data iopddl-solver /data/example.json 60 -s 42 -t 10 -j 4 -q
```
This command mounts the current directory into the container, allowing the solver to access input files.


## Plots
The code to generate the plots is in `plots.ipynb`