classdef TsallisINFCB < handle

    properties
        grid
        disc
        action
    end

    methods

        function obj = TsallisINFCB(grid, m, a, b)
            for i = 1:m
                handles(i) = TsallisINF(grid, 0);
            end
            obj.grid = handles;
            obj.disc = a + (b-a) * linspace(.5/m, 1.-.5/m, m);
        end

        function out = predict(obj, context)
            [~, i] = min(abs(obj.disc - context));
            out = obj.grid(i).predict();
            obj.action = i;
        end

        function update(obj, loss)
            obj.grid(obj.action).update(loss);
        end

    end
end