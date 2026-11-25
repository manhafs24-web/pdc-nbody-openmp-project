import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/benchmark_N30000.csv")

seq = df[df["program"] == "seq"]["time_s"]
omp = df[df["program"] == "omp"]

# T1 = median sequential time (robust against the one outlier run)
t1 = seq.median()

summary = (omp.groupby("threads")["time_s"]
              .agg(["median", "mean", "min", "max"])
              .reset_index())
summary["speedup"] = t1 / summary["median"]
summary["efficiency"] = summary["speedup"] / summary["threads"]

print("Sequential runs:", list(seq))
print("T1 (median sequential): %.3f s" % t1)
print()
print(summary.round(3).to_string(index=False))

summary.to_csv("results/summary_N30000.csv", index=False)

threads = summary["threads"]

# 1. Execution time
plt.figure(figsize=(6, 4))
yerr = [summary["median"] - summary["min"], summary["max"] - summary["median"]]
plt.errorbar(threads, summary["median"], yerr=yerr, marker="o", capsize=4,
             label="OpenMP (median, min to max)")
plt.axhline(t1, linestyle="--", color="gray", label="Sequential (median)")
plt.xlabel("Number of threads")
plt.ylabel("Execution time (s)")
plt.title("Execution time vs threads (N = 30,000, 20 steps)")
plt.xticks(threads)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/time_vs_threads.png", dpi=200)
plt.close()

# 2. Speedup
plt.figure(figsize=(6, 4))
plt.plot(threads, summary["speedup"], marker="o", label="Measured")
plt.plot(threads, threads, linestyle="--", color="gray", label="Ideal (S = p)")
plt.axvline(4, linestyle=":", color="red", label="4 physical cores")
plt.xlabel("Number of threads")
plt.ylabel("Speedup (T1 / Tp)")
plt.title("Speedup vs threads (N = 30,000, 20 steps)")
plt.xticks(threads)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/speedup_vs_threads.png", dpi=200)
plt.close()

# 3. Efficiency
plt.figure(figsize=(6, 4))
plt.plot(threads, summary["efficiency"], marker="o", label="Measured")
plt.axhline(1.0, linestyle="--", color="gray", label="Ideal (E = 1)")
plt.axvline(4, linestyle=":", color="red", label="4 physical cores")
plt.xlabel("Number of threads")
plt.ylabel("Efficiency (S / p)")
plt.title("Efficiency vs threads (N = 30,000, 20 steps)")
plt.xticks(threads)
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/efficiency_vs_threads.png", dpi=200)
plt.close()

print("\nSaved 3 graphs and results/summary_N30000.csv")