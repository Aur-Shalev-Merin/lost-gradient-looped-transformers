"""v166 pre-freeze step (prereg draft docs/prereg_v166_slam_dunk.md; pilots docs/v166_pilots_2026-10-03.md).

Runs the frozen v164 driver (v164_parcae_predict.py, unchanged) with the block-A candidate variants swapped in:
  --measure   frozen Parcae-140m err_k / rho for (R 64, a 0.3) and (R 48, a 0.4), k in {4, 8, 12, 16}
  --train     control LR checks for the untuned (R 48, a 0.4) variant, cells 'R:a:k:lr:seed'
No treatment (truncated) cell is trained here.
"""
import v164_parcae_predict as p

p.MEASURE = [(64, 0.3), (48, 0.4)]
p.KS = (4, 8, 12, 16)

if __name__ == '__main__':
    p.main()
