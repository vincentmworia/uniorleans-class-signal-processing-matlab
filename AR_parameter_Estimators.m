%% Lab: Autoregressive Model Parameter Estimation
% AR(p) model: x[n] + a1 x[n-1] + ... + ap x[n-p] = e[n]

clear; clc; close all;

%% 0) Load data from CSV (same folder as this script)
x = readmatrix('ar5_dataset.csv');   % Load the CSV file
x = x(:);                            % Ensure column vector
N = length(x);

p  = 5;                              % AR model order (given in the lab)
fs = 1;                              % Sampling frequency (set if known)

%% 1) AR parameter estimation: Yule-Walker (autocorrelation-based)
% Uses MATLAB's aryule (Yule-Walker solver)
[a_YW, noiseVar_YW] = aryule(x, p);

% aryule returns [1 a1 a2 ... ap] (note the leading 1 for x[n])
% So AR polynomial A(z) = 1 + a1 z^-1 + ... + ap z^-p
disp('Yule-Walker AR coefficients (including leading 1):');
disp(a_YW(:).');

% Residuals using Yule–Walker parameters
% e[n] = filter(A(z), 1, x[n]) = filter([1 a1 ... ap], 1, x)
e_YW = filter(a_YW, 1, x);

% Check residual mean and variance (ignore first p samples for transient)
mean_e_YW = mean(e_YW(p+1:end));
var_e_YW  = var(e_YW(p+1:end));
fprintf('\nYule-Walker residual mean: %g\n', mean_e_YW);
fprintf('Yule-Walker residual variance: %g (aryule noiseVar = %g)\n', ...
    var_e_YW, noiseVar_YW);

%% 2) Least Squares (LS) Estimator
% AR model:
%   x[n] = -a1 x[n-1] - ... - ap x[n-p] + e[n]
% Put it as: y = Phi * a + e, with
%   y(n)   = x[p+1 : N]
%   Phi(n) = [ -x[n-1]  -x[n-2] ... -x[n-p] ]

M = N - p;                       % number of usable samples
Phi = zeros(M, p);               % design matrix

for n = 1:M
    % past samples for time index n+p: x[p+n-1],...,x[n]
    x_past = x(p + n - 1 : -1 : n);      % x[n+p-1],...,x[n]
    Phi(n, :) = -x_past.';               % negative for AR form
end

y = x(p+1 : N);                  % current samples

% LS solution: a_LS = (Phi'Phi)^(-1) Phi' y
a_LS = (Phi' * Phi) \ (Phi' * y);

% Full AR polynomial (including leading 1)
A_LS = [1; a_LS];

disp('LS AR coefficients (including leading 1):');
disp(A_LS(:).');

% Residuals for LS estimator
e_LS = y - Phi * a_LS;           % directly via regression equation
mean_e_LS = mean(e_LS);
var_e_LS  = var(e_LS);
fprintf('\nLS residual mean: %g\n', mean_e_LS);
fprintf('LS residual variance: %g\n', var_e_LS);

%% 3) Regularized LS (Ridge regression)
% Modified LS: (Phi'Phi + lambda I) * a = Phi'y
lambda = 1e-2;                   % choose small regularization (tune if needed)

a_LS_reg = (Phi' * Phi + lambda * eye(p)) \ (Phi' * y);
A_LS_reg = [1; a_LS_reg];

disp('Regularized LS AR coefficients (including leading 1):');
disp(A_LS_reg(:).');

% Residuals for regularized LS
e_LS_reg = y - Phi * a_LS_reg;
mean_e_LS_reg = mean(e_LS_reg);
var_e_LS_reg  = var(e_LS_reg);
fprintf('\nRegularized LS residual mean: %g\n', mean_e_LS_reg);
fprintf('Regularized LS residual variance: %g\n', var_e_LS_reg);

%% 4) Compare estimated parameters
disp('=== Comparison of AR parameter estimates (without leading 1) ===');
T = table( (1:p).', ...
    a_YW(2:end).', ...     % Yule-Walker (skip leading 1)
    a_LS(:).', ...
    a_LS_reg(:).', ...
    'VariableNames', {'Order', 'YW', 'LS', 'LS_reg'});
disp(T);

%% 5) Compare residuals visually
figure;
subplot(3,1,1);
plot(e_YW);
title('Residuals - Yule-Walker'); xlabel('n'); ylabel('e_{YW}[n]');

subplot(3,1,2);
plot(e_LS);
title('Residuals - LS'); xlabel('n'); ylabel('e_{LS}[n]');

subplot(3,1,3);
plot(e_LS_reg);
title('Residuals - Regularized LS'); xlabel('n'); ylabel('e_{LS\_reg}[n]');

%% 6) Check model stability
% AR model is stable if all roots of A(z) lie inside unit circle (|z|<1)

% Yule-Walker stability
r_YW = roots(a_YW);
stable_YW = all(abs(r_YW) < 1);
fprintf('\nYule-Walker model stable? %d (1 = yes, 0 = no)\n', stable_YW);

% LS stability
r_LS = roots(A_LS);
stable_LS = all(abs(r_LS) < 1);
fprintf('LS model stable? %d (1 = yes, 0 = no)\n', stable_LS);

% Regularized LS stability
r_LS_reg = roots(A_LS_reg);
stable_LS_reg = all(abs(r_LS_reg) < 1);
fprintf('Regularized LS model stable? %d (1 = yes, 0 = no)\n', stable_LS_reg);

% Plot roots and unit circle
theta = linspace(0, 2*pi, 400);
uc_x = cos(theta); uc_y = sin(theta);

figure;
subplot(1,3,1);
plot(uc_x, uc_y, '--'); hold on;
plot(real(r_YW), imag(r_YW), 'x', 'MarkerSize', 8);
axis equal; grid on;
title('Roots - YW'); xlabel('Re'); ylabel('Im');

subplot(1,3,2);
plot(uc_x, uc_y, '--'); hold on;
plot(real(r_LS), imag(r_LS), 'x', 'MarkerSize', 8);
axis equal; grid on;
title('Roots - LS'); xlabel('Re'); ylabel('Im');

subplot(1,3,3);
plot(uc_x, uc_y, '--'); hold on;
plot(real(r_LS_reg), imag(r_LS_reg), 'x', 'MarkerSize', 8);
axis equal; grid on;
title('Roots - LS (reg)'); xlabel('Re'); ylabel('Im');

%% 7) Cramér–Rao Bound (CRB) approximation
% For an AR(p) with white noise variance sigma^2, the asymptotic covariance
% of an unbiased parameter estimator is approximately:
%   Cov(a_hat) ≈ (sigma^2 / N) * R^{-1}
% where R is the autocorrelation matrix of x:
%   R(i,j) = r_x(|i-j|),  i,j=1..p
%
% We estimate r_x(k) from data using xcorr, then build R.

maxLag = p-1;
[r_all, lags] = xcorr(x, maxLag, 'biased');  % lags = -maxLag: maxLag
% Extract r(0), r(1), ..., r(p-1)
r_pos = r_all(lags >= 0);          % r_pos(1)=r(0), r_pos(2)=r(1), ...
r0_to_pminus1 = r_pos(1:p);

% Build Toeplitz autocorrelation matrix R (p x p)
R = toeplitz(r0_to_pminus1);       % R(i,j) = r(|i-j|)

% Use noise variance from Yule-Walker estimate (aryule output)
sigma2 = noiseVar_YW;

% CRB covariance matrix (approximate)
CRB_cov = (sigma2 / N) * inv(R);

% CRB on each parameter (variance lower bound)
CRB_diag = diag(CRB_cov);

fprintf('\nApproximate Cramér–Rao Bound (variance) for AR parameters:\n');
for k = 1:p
    fprintf('Var(a_%d) >= %g\n', k, CRB_diag(k));
end

%% DONE
disp('Lab AR estimation script finished.');
