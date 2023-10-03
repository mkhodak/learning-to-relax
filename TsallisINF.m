classdef TsallisINF < handle
    
    properties
        grid
        d
        t
        k
        eta
        index
        prob
    end
    
    methods
        function obj = TsallisINF(grid, T)
            obj.grid = grid;
            obj.d = length(grid);
            obj.t = 1;
            obj.k = zeros(obj.d, 1);
            obj.eta = 1. / sqrt(T);
        end
        
        function out = predict(obj)

            if isinf(obj.eta)
                eta = 2. / sqrt(obj.t);
            else
                eta = obj.eta;
            end

            x = -1.;
            for i = 1:20
                probs = 4 * (eta*(obj.k - x)).^(-2.);
                x = x - (sum(probs) - 1) / (eta * sum(probs.^1.5));
            end
            
            try
                obj.index = randsample(1:obj.d, 1, true, probs);
            catch
                obj.index = randsample(1:obj.d, 1, true);
                obj.prob = 1. / obj.d;
            end
            obj.prob = probs(obj.index);
            out = obj.grid(obj.index);
        end

        function update(obj, loss)
            
            obj.k(obj.index) = obj.k(obj.index) + loss / obj.prob;
            obj.t(1) = obj.t(1) + 1;

        end
    end
end