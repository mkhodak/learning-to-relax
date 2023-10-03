A = delsq(numgrid('S', 12));
n = length(A);
epsilon = 1E-8;
T = 5000;
trials = 40;
omegas = linspace(1., 1.8, 5);
omega_costs = zeros(T, trials, length(omegas));
grid = linspace(1, 1.9, floor((2*T)^(1/3)));
tinf_costs = zeros(T, trials);

parfor trial = 1:trials
    tinf = TsallisINF(linspace(1, 1.9, floor((2*T)^(1/3))), T);
    for t = 1:T
        c = -.15 + .6 * betarnd(.5, 1.5);
        At = A + c * eye(n);
        bt = truncated_normal(n);
        tinf_costs(t, trial) = sor(At, bt, zeros(n, 1), tinf.predict(), epsilon);
        tinf.update((tinf_costs(t, trial)-1.) / 100.);
        for i = 1:5
            omega_costs(t, trial, i) = sor(At, bt, zeros(n, 1), omegas(i), epsilon);
        end
    end
end

ax = gca(figure(1));
for i = 1:5
    plot(mean(cumsum(omega_costs(:, :, i)), 2), T-(1:T), 'LineWidth', 2, 'LineStyle', '--');
    hold on;
end
plot(mean(cumsum(tinf_costs), 2), T-(1:T), 'LineWidth', 2, 'Color', 'black');
legend('\omega=1.0', '\omega=1.2', '\omega=1.4', '\omega=1.6', '\omega=1.8 (\approx\omega^\ast)', 'Tsallis-INF', 'FontSize', 20);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
xlabel('total iterations', 'FontSize', 20);
ylabel('instances remaining', 'FontSize', 20);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print('learning_high_variance.png', '-dpng', '-r256');
hold off;

parfor trial = 1:trials
    tinf = TsallisINF(linspace(1, 1.9, floor((2*T)^(1/3))), T);
    for t = 1:T
        c = -.15 + .6 * betarnd(2., 6.);
        At = A + c * eye(n);
        bt = truncated_normal(n);
        tinf_costs(t, trial) = sor(At, bt, zeros(n, 1), tinf.predict(), epsilon);
        tinf.update((tinf_costs(t, trial)-1.) / 100.);
        for i = 1:5
            omega_costs(t, trial, i) = sor(At, bt, zeros(n, 1), omegas(i), epsilon);
        end
    end
end

ax = gca(figure(2));
for i = 1:5
    plot(mean(cumsum(omega_costs(:, :, i)), 2), T-(1:T), 'LineWidth', 2, 'LineStyle', '--');
    hold on;
end
plot(mean(cumsum(tinf_costs), 2), T-(1:T), 'LineWidth', 2, 'Color', 'black');
legend('\omega=1.0', '\omega=1.2', '\omega=1.4', '\omega=1.6 (\approx\omega^\ast)', '\omega=1.8', 'Tsallis-INF', 'FontSize', 20);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
xlabel('total iterations', 'FontSize', 20);
ylabel('instances remaining', 'FontSize', 20);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print('learning_low_variance.png', '-dpng', '-r256');
hold off;