# Steady Error Diagnosis

## A) Error signal sanity
unique_count(e)=1999
top10=[(-1.11022, 2), (1.129601, 1), (-0.87005, 1), (-0.882657, 1), (-0.881973, 1), (-0.881539, 1), (-0.880415, 1), (-0.879285, 1), (-0.878151, 1), (-0.87701, 1)]
min=-1.1262, max=1.1296, mean=-0.0056, mean_abs=0.8775

## B) Actuator saturation
sat_upper=48.40%, sat_lower=49.00%, mean(u_applied)=-0.0171

## C) Integral test
k_i=0.1: median|e|=1.0191, sat_fraction=97.55%
k_i=0.2: median|e|=1.0371, sat_fraction=97.55%
k_i=0.5: median|e|=1.0478, sat_fraction=97.55%
