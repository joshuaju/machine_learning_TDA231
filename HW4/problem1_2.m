x = [ 2 2; 4 4; 4 0;
    0 0; 2 0; 0 2];
t = [1; 1; 1;
    -1; -1; -1];

%% Primal
% Contraint is Aw >= 1. 'fmincon' expects Ax <= b, so we need to multiply
% by (-1)
A = (-1) * t .* [x ones(6,1)];
b = (-1) * ones(6,1);

primal = fmincon(@(x) 0.5 * (x'*x), zeros(3,1), A, b)
%% Dual
K = x*x'; % linear kernel function
H = (t*t').* K;
f = -1 * ones(6, 1);
Aeq = t'; 
beq=0;
LB = zeros(6,1);
UB =  repmat(inf, 6, 1);
alphas = quadprog(H, f, [], [], Aeq, beq, LB, UB)