# Step A with friction term

c_hat=1.2061, f_v_hat=0.3455
c_relerr=0.0051
corr(residual,v)=-0.0242

# Residual correlation table
corr(v)=-0.0242, corr(signv)=-0.0168, corr(clip)=0.0290

# Step B comparison (baseline vs friction-aware)
baseline: {'rmse': 0.256499833967289, 'recovery': 0.25, 'steady': 6.475211367096279e-10, 'effort': 330.0750766430855, 'smooth': 5.09848348081584, 'corr_b': 0.9334374407751532, 'alpha': 0.9692469931915708, 'beta': 0.05790324596197468}
friction-aware: {'rmse': 0.256499833967289, 'recovery': 0.25, 'steady': 6.475211367096279e-10, 'effort': 330.0750766430855, 'smooth': 5.09848348081584, 'corr_b': 0.9340919147404333, 'alpha': 0.978906479393179, 'beta': 0.05264232056733202}
