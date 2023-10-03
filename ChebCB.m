classdef ChebCB < handle

    properties
        grid
        d
        m
        t
        eta
        actions
        theta
        features
        losses
        a
        bma
        lb
        ub

        predicted
        probabilities
        contexts

    end

    methods
        function obj = ChebCB(grid, T, m, a, b, scale)
            obj.grid = grid;
            obj.d = length(grid);
            obj.m = m;
            obj.t = 1;
            obj.actions = zeros(T, 1);
            obj.theta = zeros(obj.d, m);
            obj.features = zeros(T, m);
            obj.losses = zeros(T, 1);
            obj.eta = sqrt(obj.d*m^2*T/(obj.d*m^3+T));
            obj.a = a;
            obj.bma = b - a;
            obj.lb = [-scale, rdivide(-scale/(b-a), 1:m-1)];
            obj.ub = [scale, rdivide(scale/(b-a), 1:m-1)];

            obj.predicted = zeros(T, 1);
            obj.probabilities = zeros(T, 1);
            obj.contexts = zeros(T, 1);

        end

        function out = predict(obj, context)

            feature = chebyshevT(0:obj.m-1, 2./obj.bma*(context-obj.a)-1.);
            obj.features(obj.t, :) = feature;
            yhat = obj.theta * feature';
            [ystar, istar] = min(yhat);
            other = (1:obj.d) ~= istar;
            probs = zeros(obj.d, 1);
            probs(other) = rdivide(1., obj.d + obj.eta * (yhat(other)-ystar));
            probs(istar) = 1. - sum(probs(other));

            i = randsample(1:obj.d, 1, true, probs);
            obj.actions(obj.t) = i;
            out = obj.grid(i);

            obj.predicted(obj.t) = obj.theta(i,:) * feature';
            obj.probabilities(obj.t) = probs(istar);
            obj.contexts(obj.t) = context;

        end

        function update(obj, loss)

            obj.losses(obj.t) = loss;
            i = obj.actions(obj.t);
            idx = obj.actions == i;
            obj.theta(i, :) = lsqlin(obj.features(idx, :), obj.losses(idx), [], [], [], [], obj.lb, obj.ub, obj.theta(i, :), optimoptions('lsqlin', 'Display', 'off'));
            obj.t(1) = obj.t(1) + 1;

        end
    end
end