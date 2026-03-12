"""
Speed benchmark: fss(method='fft') vs fss(method='integral').

Runs each method REPEATS times for every window size in WINDOWS, prints a
summary table, then checks that both methods produce numerically identical
results for each window.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import time
import numpy as np
from meteothek.calc import fss

# ── Benchmark settings ────────────────────────────────────────────────────────
REPEATS   = 300
SHAPE     = (500, 500)          # field size
THRESHOLD = 0.2                 # accumulation threshold
WINDOWS   = [3, 5, 10, 15, 25, 50, 100]  # neighbourhood window sizes to test
NAN_FRAC  = 0.05                # fraction of randomly placed NaNs
SEED      = 0
ABS_TOL   = 1e-9                # tolerance for result-match check
# The SAT prefix-sum can reach O(N²×max_field) ≈ 200,000 for a 500×500 binary
# field.  Float64 machine epsilon is ~2.2e-16, so absolute cancellation error in
# a single SAT lookup is ~4e-11.  The FSS score is computed from means of squared
# differences of such fractions, so combined errors from both SAT and FFT (which
# also accumulates O(N²) internally) can exceed 1e-10 for large windows.  1e-9
# is a conservative safe tolerance for this comparison.
# ─────────────────────────────────────────────────────────────────────────────

rng  = np.random.RandomState(SEED)
fcst = rng.rand(*SHAPE)
obs  = rng.rand(*SHAPE)
# sprinkle NaNs
fcst[rng.rand(*SHAPE) < NAN_FRAC] = np.nan
obs [rng.rand(*SHAPE) < NAN_FRAC] = np.nan

print(f"Field shape : {SHAPE}")
print(f"Repeats     : {REPEATS}")
print(f"Windows     : {WINDOWS}")
print()


def run_benchmark(method, window):
    times = []
    result = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = fss(fcst, obs, threshold=THRESHOLD, window=window,
                     thrsd_type="accumulation", method=method)
        times.append(time.perf_counter() - t0)
    return result, np.array(times)


# ── Run benchmarks for all window sizes ───────────────────────────────────────
results = {}   # (window, method) -> (fss_result, times_array)
for window in WINDOWS:
    for method in ("fft", "integral"):
        print(f"  window={window:>4},  method={method} …", flush=True)
        results[(window, method)] = run_benchmark(method, window)

# ── Timing table ──────────────────────────────────────────────────────────────
col = 10   # column width for ms values
header = (f"{'window':>8}  {'fft mean':>{col}}  {'int mean':>{col}}  "
          f"{'fft min':>{col}}  {'int min':>{col}}  {'speedup':>8}  match")
sep = "─" * len(header)
print(f"\n{sep}")
print(header)
print(sep)

match_failures = []
for window in WINDOWS:
    res_fft, t_fft = results[(window, "fft")]
    res_int, t_int = results[(window, "integral")]

    mean_fft = t_fft.mean() * 1e3
    mean_int = t_int.mean() * 1e3
    min_fft  = t_fft.min()  * 1e3
    min_int  = t_int.min()  * 1e3
    speedup  = mean_fft / mean_int   # >1 means integral is faster

    # result-match check across num, denom, score
    labels = ["num", "denom", "score"]
    match_ok = True
    fail_details = []
    for i, (a, b) in enumerate(zip(res_fft, res_int)):
        if np.isnan(a) and np.isnan(b):
            continue
        diff = abs(a - b)
        if diff >= ABS_TOL:
            match_ok = False
            fail_details.append(f"{labels[i]}: |diff|={diff:.2e}")
    match_str = "OK" if match_ok else "FAIL(" + ", ".join(fail_details) + ")"
    if not match_ok:
        match_failures.append((window, fail_details))

    faster = "int" if speedup > 1 else "fft"
    print(f"{window:>8}  {mean_fft:>{col}.2f} ms  {mean_int:>{col}.2f} ms  "
          f"{min_fft:>{col}.2f} ms  {min_int:>{col}.2f} ms  "
          f"{speedup:>6.2f}x {faster}  {match_str}")

print(sep)
print("speedup >1 means integral is faster; <1 means fft is faster")

# ── Final result-match summary ────────────────────────────────────────────────
print()
if match_failures:
    print("RESULT MISMATCH DETECTED for windows:", [w for w, _ in match_failures])
    for window, details in match_failures:
        print(f"  window={window}: {details}")
else:
    print(f"Result match: all windows PASSED (|diff| < {ABS_TOL:.0e})")
