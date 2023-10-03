for region = ['L' 'S']
    for s = [12 32]
        for offset = [0. .5]

A = delsq(numgrid(region, s));
n = length(A);
A = A + offset*eye(n);
b = rand(n,1)-0.5;
D = diag(diag(A));
L = tril(A,-1);

K = 100;
epsilon = 1E-8;
omegas = linspace(2.*sqrt(2.)-2., 1.9, K);
costs = zeros(size(omegas));
bounds = cgbound(A, omegas, epsilon) - 1.;

for i = 1:K
    x = zeros(n,1);
    omega = omegas(i);
    M1 = (D+omega*L)'*inv(D);
    M2 = (D+omega*L) / omega / (2.-omega);
    [~, ~, ~, costs(i), ~] = pcg(A, b, epsilon, 1000, M1, M2, x);
end

tau = max(rdivide(costs-1., bounds));


ax = gca(figure(1));
plot(omegas, costs, 'LineWidth', 2);
hold on;
plot(omegas, 1. + tau * bounds, 'LineWidth', 2, 'LineStyle', '--');
legend('actual cost', 'upper bound', 'Location', 'north', 'FontSize', 20);
title([region '-shaped domain: size=' int2str(n) ', offset=' num2str(offset)], 'Fontsize', 24)
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
ylabel('iterations', 'FontSize', 20)
xlabel('\omega', 'FontSize', 24);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print(['cgbound-' region '-' int2str(n) '-' num2str(offset) '.png'], '-dpng', '-r256');
hold off;

        end
    end
end