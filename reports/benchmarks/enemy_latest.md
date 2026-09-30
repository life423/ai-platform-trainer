| Enemy | Objective | Net/min | Catch rate | Missile-hit rate | Timeout | Catches/min | Hits/min | Time to catch (s) | Closing (px/s) | Missiles evaded |
|---|---|---|---|---|---|---|---|---|---|---|
| Random movement | -1.029 | -8.43 | 9.3% | 79.1% | 11.6% | 1.12 | 9.55 | 2.21 | 23 | 61.0% |
| Direct chase | -0.519 | -32.68 | 26.9% | 73.1% | 0.0% | 19.03 | 51.70 | 0.79 | 450 | 0.0% |
| Adaptive AI (scripted) | -0.989 | -4.47 | 15.4% | 67.6% | 17.0% | 1.32 | 5.79 | 5.43 | 28 | 85.6% |
| Adaptive AI (as shipped, speed 10+) | -0.532 | -4.55 | 34.4% | 62.7% | 2.9% | 5.53 | 10.07 | 2.65 | 83 | 75.3% |
| Legacy supervised NN | -0.992 | -9.87 | 10.3% | 80.9% | 8.8% | 1.44 | 11.31 | 1.92 | 27 | 43.9% |
| PPO agent | -0.615 | 0.61 | 26.6% | 15.6% | 57.8% | 1.47 | 0.86 | 4.22 | 17 | 97.9% |
