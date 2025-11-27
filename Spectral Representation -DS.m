clear; close all; clc;

%% ===================================================================== %%
%  EXERCISE 1 – PART 1 : Analog Signals & Spectral Representation
%% ===================================================================== %%

% Continuous-time approximation (high Fs to mimic analog)
Fs_ct = 5000;                 % "continuous-time" sampling frequency [Hz]
Ts_ct = 1/Fs_ct;
t_start = -2;                 % signal observed from t = -2 ...
t_end   =  7;                 % ... to t = 7 seconds
t_ct    = t_start:Ts_ct:t_end;

% Window parameters
a = 0;                        % window start [s]
b = 5;                        % window end   [s]

% Continuous-time signals
w_ct = double((t_ct >= a) & (t_ct <= b));   % rectangular window w(t)
x_ct = cos(t_ct);                            % x(t) = cos(t)
y_ct = exp(-t_ct) .* (t_ct >= 0);           % y(t) = e^{-t} u(t)
h_ct = x_ct .* y_ct .* w_ct;                % h(t) = x(t) y(t) w(t)

% FFT of x(t) and h(t)
N_ct = numel(t_ct);
X_ct = fftshift( fft(x_ct) );
H_ct = fftshift( fft(h_ct) );

% Frequency axis (Hz)
f_ct = (-N_ct/2 : N_ct/2-1) * (Fs_ct/N_ct);

% Plot spectra of x(t) and h(t)
figure;
plot(f_ct, abs(X_ct)/max(abs(X_ct)), 'LineWidth', 1.2); hold on;
plot(f_ct, abs(H_ct)/max(abs(H_ct)), 'LineWidth', 1.2);
xlabel('Frequency (Hz)');
ylabel('Normalized magnitude');
legend('X(f)', 'H(f)');
title('Exercise 1 – Part 1: Spectra of x(t) and h(t)');
grid on;
xlim([-50 50]);    % zoom around low frequencies


%% ===================================================================== %%
%  EXERCISE 1 – PART 2 : Sampling & Spectral Representation
%% ===================================================================== %%

Fs_s = 10;                     % discrete-time sampling frequency [Hz]
Ts_s = 1/Fs_s;
N_dft = 8;                     % DFT length

n     = 0:N_dft-1;             % discrete index n
t_s   = n * Ts_s;              % corresponding continuous-time instants

% Sampled signals x[n], y[n], w[n], h[n]
w_n = double((t_s >= a) & (t_s <= b));    % window sampled
x_n = cos(t_s);                           % x[n]
y_n = exp(-t_s) .* (t_s >= 0);           % y[n]
h_n = x_n .* y_n .* w_n;                 % h[n]

% DFTs of length N_dft
Xk = fft(x_n, N_dft);
Hk = fft(h_n, N_dft);

% Shift zero frequency to the centre
Xk_shift = fftshift(Xk);
Hk_shift = fftshift(Hk);

% Frequency axis for shifted DFT (Hz)
k  = 0:N_dft-1;
f_s = (k - N_dft/2) * (Fs_s/N_dft);

% Plot DFT magnitudes
figure;
stem(f_s, abs(Xk_shift)/max(abs(Xk_shift)), 'filled'); hold on;
stem(f_s, abs(Hk_shift)/max(abs(Hk_shift)), 'filled');
xlabel('Frequency (Hz)');
ylabel('Normalized magnitude');
legend('X[k]', 'H[k]');
title(sprintf('Exercise 1 – Part 2: DFT of x[n] and h[n] (F_s = %g Hz, N = %d)', ...
              Fs_s, N_dft));
grid on;


%% ===================================================================== %%
%  EXERCISE 1 – PART 3 : Random Noise, ACF & PSD (Non-stationary Output)
%% ===================================================================== %%

% White noise b(t) : zero-mean, unit-variance Gaussian
b_ct = randn(size(t_ct));           % approx. E{b}=0, Var{b}=1

% Output process: s(t) = h(t) * b(t) (pointwise product)
s_ct = h_ct .* b_ct;

% --- Visualise h(t), b(t), and s(t) ---
figure;
subplot(3,1,1);
plot(t_ct, h_ct);
xlabel('t (s)'); ylabel('h(t)');
title('h(t) – deterministic input');
grid on;

subplot(3,1,2);
plot(t_ct, b_ct);
xlabel('t (s)'); ylabel('b(t)');
title('b(t) – white Gaussian noise');
grid on;

subplot(3,1,3);
plot(t_ct, s_ct);
xlabel('t (s)'); ylabel('s(t)');
title('s(t) = h(t)\cdot b(t)');
grid on;

% --- Time-varying variance (non-stationarity check) ---
winSamples = round(0.1 * Fs_ct);    % 0.1 s sliding window
var_s = movvar(s_ct, winSamples);

figure;
plot(t_ct, var_s, 'DisplayName','Estimated Var\{s(t)\}'); hold on;
plot(t_ct, h_ct.^2 / max(h_ct.^2) * max(var_s), '--', ...
     'DisplayName','Scaled h(t)^2');
xlabel('t (s)');
ylabel('Variance / scaled h(t)^2');
title('Exercise 1 – Part 3: Time-varying variance of s(t)');
legend('Location','best');
grid on;

% --- Empirical ACF and PSD (time-averaged, process is NOT WSS) ---
[Rs, lags_s] = xcorr(s_ct, 'biased');
tau_s = lags_s / Fs_ct;

figure;
plot(tau_s, Rs);
xlabel('\tau (s)');
ylabel('R_s(\tau)');
title('Estimated autocorrelation of s(t) (time-averaged)');
grid on;

figure;
pwelch(s_ct, [], [], [], Fs_ct);
title('Welch PSD estimate of s(t) (non-stationary process)');
grid on;


%% ===================================================================== %%
%  EXERCISE 2 – LTI System with White-Noise Input
%% ===================================================================== %%

% System: y[n] = x[n] + x[n-1] + x[n-2]  ->  h2 = [1 1 1]
h2 = [1 1 1];

% Input white Gaussian noise x[n], zero mean, variance sigma_x^2
N2       = 5e4;                  % number of samples
sigma_x2 = 1;                    % input variance
x2       = sqrt(sigma_x2) * randn(1, N2);

% Output y[n] of the LTI system
y2 = filter(h2, 1, x2);

% --- Theoretical autocorrelation R_y[m], m = -2..2 ---
Ry_theo   = sigma_x2 * conv(h2, fliplr(h2));  % [1 2 3 2 1]*sigma_x2
lags_theo = -2:2;

% --- Estimated autocorrelation from data (lags -2..2) ---
[Ry_est, lags_est] = xcorr(y2, 2, 'biased');

figure;
stem(lags_est, Ry_est, 'filled'); hold on;
stem(lags_theo, Ry_theo, 'LineWidth', 1.5);
xlabel('Lag m');
ylabel('R_y[m]');
legend('Estimated R_y[m]', 'Theoretical R_y[m]');
title('Exercise 2: Autocorrelation of y[n]');
grid on;

% --- PSD: theoretical vs Welch estimate ---
% Welch estimate (Fs = 1 for normalized frequency)
[Sy_est, f_est] = pwelch(y2, 1024, 512, 1024, 1);

% Theoretical PSD on same frequency grid
omega = 2*pi*f_est;                     % rad/sample
H_omega = 1 + exp(-1j*omega) + exp(-1j*2*omega);
Sy_theo = sigma_x2 * abs(H_omega).^2;

figure;
plot(f_est, Sy_theo, 'LineWidth', 1.5); hold on;
plot(f_est, Sy_est, '--');
xlabel('Normalized frequency (cycles/sample)');
ylabel('S_y');
legend('Theoretical PSD', 'Estimated PSD (Welch)');
title('Exercise 2: PSD of y[n]');
grid on;
