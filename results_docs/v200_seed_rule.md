# v200: seed rule applied (after the freeze 4849a02, before any scored run)

The prereg's seed rule (`v200_analyze.seed_rule`) reads one number from the spread pilot: s, the larger over the two
models of the sample s.d. of φ at k = 1 over the pilot seeds 9001–9003, on test. The φ means were not printed or read.

**Pilot runs used:**
- seeds 9002–9003: full, floor and k = 1 from the spread pilot (astrocyte-v200-{pln5,r5}-spread);
- seed 9001:
  - k = 1 from the spread pilot;
  - full at 1e-4 from pilot-lr3 (embedding frozen);
  - floor at 3e-4 from pilot 1. The floor never trains the embedding, so the freeze does not affect it.

All runs had valid controls (≤ 10 non-finite steps, finite losses), and every selftest passed.

| model | s.d. of φ at k = 1 |
|---|---|
| pln5 | 0.0050 |
| r5 | 0.0106 |

**s = 0.0106, so N = 5** evaluation seeds plus 2 backups per model (seeds 2001101–2001107).

**Scored study.** Kernels built, not yet pushed:
- aurmerin/astrocyte-v200-{pln5,r5}-study0;
- aurshalevmerin/astrocyte-v200-{pln5,r5}-study1.

That is 70 runs, about 14 T4-hours. Analysis:

```
V200_N_SEEDS=5 python v200_analyze.py <the four study output dirs>
```
