A = delsq(numgrid('S', 12));
n = length(A);
epsilon = 1E-8;
T = 5000;
trials = 40;
tinf_costs = zeros(T, trials);
cheb_costs = zeros(T, trials);
tinfcb_costs = zeros(T, trials);
opt_costs = zeros(T, trials);
omega_costs = zeros(T, trials);

parfor trial = 1:trials
    tinf = TsallisINF(linspace(1, 1.9, floor((2*T)^(1/3))), T);
    cheb = ChebCB(linspace(1, 1.9, 19), T, 6, -.15, .65, 1000);
    tinfcb = TsallisINFCB(linspace(1, 1.9, 19), 5, -.15, .65);
    for t = 1:T
        c = -.15 + .6 * betarnd(.5, 1.5);
        At = A + c * eye(n);
        bt = truncated_normal(n);
        tinf_costs(t, trial) = sor(At, bt, zeros(n, 1), tinf.predict(), epsilon);
        tinf.update((tinf_costs(t, trial)-1.) / 100.);
        cheb_costs(t, trial) = sor(At, bt, zeros(n, 1), cheb.predict(c), epsilon);
        cheb.update(cheb_costs(t, trial)-1);
        tinfcb_costs(t, trial) = sor(At, bt, zeros(n, 1), tinfcb.predict(c), epsilon);
        tinfcb.update((tinfcb_costs(t, trial)-1) / 100.);
        opt_costs(t, trial) = sor(At, bt, zeros(n, 1), omega_opt(At), epsilon);
        omega_costs(t, trial) = sor(At, bt, zeros(n, 1), 1.8, epsilon);
    end
    fprintf('trial %2d finished\n', trial);
end

ax = gca(figure(1));
plot(mean(cumsum(tinf_costs), 2), T-(1:T), 'LineWidth', 2);
hold on;
plot(mean(cumsum(omega_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(tinfcb_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(cheb_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(opt_costs), 2), T-(1:T), 'LineWidth', 2);
legend('Tsallis-INF', '\omega=1.8 (\approx\omega^\ast)', 'Tsallis-INF-CB', 'ChebCB', 'Instance-Optimal', 'FontSize', 20);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
xlabel('total iterations', 'FontSize', 20);
ylabel('instances remaining', 'FontSize', 20);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print('contextual_high_variance.png', '-dpng', '-r256');
hold off;

parfor trial = 1:trials
    tinf = TsallisINF(linspace(1, 1.9, floor((2*T)^(1/3))), T);
    cheb = ChebCB(linspace(1, 1.9, 19), T, 6, -.15, .65, 1000);
    tinfcb = TsallisINFCB(linspace(1, 1.9, 19), 5, -.15, .65);
    for t = 1:T
        c = -.15 + .6 * betarnd(2., 6.);
        At = A + c * eye(n);
        bt = truncated_normal(n);
        tinf_costs(t, trial) = sor(At, bt, zeros(n, 1), tinf.predict(), epsilon);
        tinf.update((tinf_costs(t, trial)-1.) / 100.);
        cheb_costs(t, trial) = sor(At, bt, zeros(n, 1), cheb.predict(c), epsilon);
        cheb.update(cheb_costs(t, trial)-1);
        tinfcb_costs(t, trial) = sor(At, bt, zeros(n, 1), tinfcb.predict(c), epsilon);
        tinfcb.update((tinfcb_costs(t, trial)-1) / 100.);
        opt_costs(t, trial) = sor(At, bt, zeros(n, 1), omega_opt(At), epsilon);
        omega_costs(t, trial) = sor(At, bt, zeros(n, 1), 1.6, epsilon);
    end
    fprintf('trial %2d finished\n', trial);
end

ax = gca(figure(2));
plot(mean(cumsum(tinf_costs), 2), T-(1:T), 'LineWidth', 2);
hold on;
plot(mean(cumsum(omega_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(tinfcb_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(cheb_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(opt_costs), 2), T-(1:T), 'LineWidth', 2);
legend('Tsallis-INF', '\omega=1.6 (\approx\omega^\ast)', 'Tsallis-INF-CB', 'ChebCB', 'Instance-Optimal', 'FontSize', 20);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
xlabel('total iterations', 'FontSize', 20);
ylabel('instances remaining', 'FontSize', 20);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print('contextual_low_variance.png', '-dpng', '-r256');
hold off;