# Phase 1.6 A2-4 Report

## A2-4.1 Ground-truth regression
c_hat=1.1936, c_relerr=0.0053, R2=0.9902

## A2-4.2 Residual analysis (baseline)
corr(residual, sign(v))=0.8708
corr(residual, v)=0.9959
corr(residual, |v|)=0.3615
corr(residual, clip)=nan

## A2-4.3 Toggle test
baseline: signv=0.8708, v=0.9959, |v|=0.3615, clip=nan
no_friction: signv=0.0739, v=0.0463, |v|=0.0399, clip=nan
no_clip: signv=0.8708, v=0.9959, |v|=0.3615, clip=nan
no_delay: signv=0.8711, v=0.9960, |v|=0.3581, clip=nan
