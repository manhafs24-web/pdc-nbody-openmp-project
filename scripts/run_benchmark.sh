#!/bin/bash
# Benchmark sweep: sequential and OpenMP at several thread counts

N=30000
STEPS=20
SEED=42
REPS=3
OUT=results/benchmark_N${N}.csv

echo "program,threads,run,time_s,sum_x" > $OUT

run_case() {
    # $1 = label, $2 = threads, $3 = run number, rest = command
    label=$1; th=$2; r=$3
    shift 3
    out=$("$@")
    t=$(echo "$out" | grep "Time" | awk '{print $3}')
    s=$(echo "$out" | grep "Sum of x" | awk '{print $4}')
    echo "$label,$th,$r,$t,$s" | tee -a $OUT
    sleep 5
}

for r in $(seq 1 $REPS); do
    run_case seq 1 $r ./bin/nbody_seq $N $STEPS $SEED
    for th in 1 2 4 6 8; do
        run_case omp $th $r ./bin/nbody_omp $N $STEPS $SEED $th
    done
done

echo "Done"