%% AR(p) parameter estimation
clear; clc; close all;

%% Load data
x = readmatrix('ar5_dataset.csv');
x = x(:);
N = length(x);
p = 5;

%% 1) Yule–Walker (autocorrelation based) estimator
[aYW, sigma2_YW] = aryule(x, p);
eYW = filter(aYW, 1, x);
eYW = eYW(p+1:end);

fprintf('Yule–Walker residual mean  = %.3g\n', mean(eYW));
fprintf('Yule–Walker residual var   = %.3g (aryule σ² = %.3g)\n\n', var(eYW), sigma2_YW);

%% 2) Least-Squares (LS) estimator
M   = N - p;
Phi = zeros(M, p);
for n = 1:M
    Phi(n,:) = -x(p+n-1:-1:n).';
end
y = x(p+1:end);

aLS  = (Phi' * Phi) \ (Phi' * y);
ALS  = [1; aLS];
eLS  = y - Phi * aLS;

fprintf('LS residual mean           = %.3g\n', mean(eLS));
fprintf('LS residual var            = %.3g\n\n', var(eLS));

%% Regularized LS (ridge)
lambda = 1e-2;
aLSreg  = (Phi' * Phi + lambda * eye(p)) \ (Phi' * y);
ALSreg  = [1; aLSreg];
eLSreg  = y - Phi * aLSreg;

fprintf('Reg. LS residual mean      = %.3g\n', mean(eLSreg));
fprintf('Reg. LS residual var       = %.3g\n\n', var(eLSreg));

%% Compare AR parameter estimates
T = table((1:p).', aYW(2:end).', aLS, aLSreg, ...
          'VariableNames', {'Order','YW','LS','LS_reg'});
disp(T);

%% Stability check
rYW    = roots(aYW);
rLS    = roots(ALS);
rLSreg = roots(ALSreg);

fprintf('Model stable?  YW:%d  LS:%d  RegLS:%d\n\n', ...
        all(abs(rYW)    < 1), ...
        all(abs(rLS)    < 1), ...
        all(abs(rLSreg) < 1));

%% 3) Cramér–Rao Bound (approx.)
maxLag = p-1;
[r,lags] = xcorr(x, maxLag, 'biased');
r_pos = r(lags >= 0);
R = toeplitz(r_pos(1:p));

CRB_cov = (sigma2_YW / N) * inv(R);
CRB_var = diag(CRB_cov);

CRB_table = table((1:p).', CRB_var,'VariableNames', {'Order','CRB_variance'});
disp(CRB_table); 
