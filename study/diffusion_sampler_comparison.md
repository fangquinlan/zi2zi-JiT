# Diffusion Sampler Comparison

This note summarizes the practical difference between diffusion samplers such as `euler`, `heun`, and `ab2` for the current inference path, using the MPS benchmark run from the study worktree.

## Context

The codebase is doing diffusion ODE-style sampling during image generation. The relevant samplers currently studied are:

- `euler`
- `heun`
- `ab2` (Adams-Bashforth 2)

`heun-50` is treated as the quality reference point in the current benchmark.

This document is not limited to `DDPM` or `DDIM`. Those are broader diffusion sampling formulations. The focus here is the practical solver choice used in this repo's inference loop.

## Heun vs AB2

### Heun

For one update from `t` to `t_next`, Heun does two model evaluations:

1. Evaluate the slope at `t`
2. Take a temporary Euler step
3. Evaluate the slope again at `t_next`
4. Average the two slopes for the final update

In simplified form:

```python
v_t = f(z_t, t)
z_euler = z_t + h * v_t
v_t_next = f(z_euler, t_next)
z_next = z_t + h * 0.5 * (v_t + v_t_next)
```

Practical effect:

- better local correction
- more robust
- roughly 2 model evaluations per step

### AB2

AB2 is a 2-step explicit multistep method. It uses one previous slope and one current slope:

```python
z_next = z_t + h * (1.5 * v_t - 0.5 * v_prev)
```

Practical effect:

- only 1 new model evaluation per step after bootstrap
- faster than Heun
- depends on previous-step history, so stability and quality must be validated empirically

## Why AB2 Can Be Faster

The dominant inference cost is model evaluation, not the simple arithmetic in the solver update.

- Heun: 2 evaluations per step
- AB2: 1 new evaluation per step after the first step

That is why AB2 can reduce wall-clock substantially even when both are nominally second-order methods.

## Current Benchmark Result

Benchmark setup:

- Device: `mps`
- Dataset: `combined_test_400.npz`
- Random subset size: `50`
- Seed: `42`
- Reference checkpoint: the active evaluation checkpoint used for the benchmark run

Measured against `heun-50`:

| Method | Steps | Sampling (s) | Img/s | Speedup vs Heun-50 | SSIM | LPIPS | L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| heun | 50 | 131.666 | 0.38 | 1.00 | 0.6845 | 0.2426 | 0.1168 |
| heun | 30 | 70.440 | 0.71 | 1.87 | 0.6976 | 0.2443 | 0.1203 |
| heun | 20 | 53.157 | 0.94 | 2.48 | 0.7306 | 0.2350 | 0.1111 |
| euler | 20 | 34.074 | 1.47 | 3.86 | 0.7319 | 0.2301 | 0.1103 |
| ab2 | 20 | 28.246 | 1.77 | 4.66 | 0.7332 | 0.2290 | 0.1096 |
| ab2 | 30 | 39.616 | 1.26 | 3.32 | 0.6899 | 0.2375 | 0.1139 |

## Interpretation

- `heun-50` is the slow quality reference, not the best routine default.
- `ab2-20` gave the best speed/metric tradeoff in this benchmark.
- `euler-20` was also strong and much cheaper than `heun-50`.
- `heun-20` outperformed `heun-30` and `heun-50` on this particular subset.
- `ab2-30` did not improve over `ab2-20`, so more steps were not automatically better here.

## Conclusion

For the current model and this MPS benchmark subset:

- keep `heun-50` as the upper-reference baseline
- prefer `ab2-20` as the main fast candidate
- keep `euler-20` as a simpler fallback option

## Artifacts

Study benchmark root:

- `benchmark/combined_test_400_sample50_seed42`

AB2 sample folders:

- `benchmark/combined_test_400_sample50_seed42/ab2-steps20/generated`
- `benchmark/combined_test_400_sample50_seed42/ab2-steps20/compare_pairs`
- `benchmark/combined_test_400_sample50_seed42/ab2-steps20/compare_grids`
