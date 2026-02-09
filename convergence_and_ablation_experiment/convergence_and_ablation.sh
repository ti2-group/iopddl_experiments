#!/bin/bash
#SBATCH --partition=short,long
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9

export executable=./iopddl

# Define input files and their corresponding timeouts
declare -A problems=(
    ["A"]="asplos-2025-iopddl-A.json 60 12"
    ["B"]="asplos-2025-iopddl-B.json 60 12"
    ["C"]="asplos-2025-iopddl-C.json 60 12"
    ["D"]="asplos-2025-iopddl-D.json 60 12"
    ["E"]="asplos-2025-iopddl-E.json 60 12"
    ["F"]="asplos-2025-iopddl-F.json 120 12"
    ["G"]="asplos-2025-iopddl-G.json 120 12"
    ["H"]="asplos-2025-iopddl-H.json 120 12"
    ["I"]="asplos-2025-iopddl-I.json 120 12"
    ["J"]="asplos-2025-iopddl-J.json 120 12"
    ["K"]="asplos-2025-iopddl-K.json 180 12"
    ["L"]="asplos-2025-iopddl-L.json 180 12"
    ["M"]="asplos-2025-iopddl-M.json 180 12"
    ["N"]="asplos-2025-iopddl-N.json 180 12"
    ["O"]="asplos-2025-iopddl-O.json 180 12"
    ["P"]="asplos-2025-iopddl-P.json 240 12"
    ["Q"]="asplos-2025-iopddl-Q.json 240 12"
    ["R"]="asplos-2025-iopddl-R.json 240 12"
    ["S"]="asplos-2025-iopddl-S.json 240 12"
    ["T"]="asplos-2025-iopddl-T.json 240 12"
    ["U"]="asplos-2025-iopddl-U.json 300 12"
    ["V"]="asplos-2025-iopddl-V.json 300 12"
    ["W"]="asplos-2025-iopddl-W.json 300 30"
    ["X"]="asplos-2025-iopddl-X.json 300 12"
    ["Y"]="asplos-2025-iopddl-Y.json 300 12"
)

# Seeds to use
seeds=(1 43 1457 789 2468 3579 9876 5432 6543 8765)

seed=${seeds[$SLURM_ARRAY_TASK_ID]:-${seeds[0]}}
# Function to run solver and validator for a problem
run_problem() {
    local problem=$1
    local filename=$2
    local timeout=$3
    local seed=$4
    local solver_timeout=$5

    export IP="../benchmarks/$filename"
    $executable $IP $timeout -s $seed -t $5 > "${problem}_solver.out"
}

# Main loop

echo "Processing seed $seed"
mkdir -p "s$seed"
cd "s$seed"
for key in "${!problems[@]}"; do
    read -r filename timeout solver_timeout<<< "${problems[$key]}"
    run_problem "$key" "$filename" "$timeout" "$seed" "$solver_timeout"
done
cd ..
