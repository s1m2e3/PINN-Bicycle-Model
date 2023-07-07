
%% Define Jacobian
jacobians = jacobian(losses,betas);
%% Test Function
weights = randn([nodes 1]);
eval = subs(losses,W,weights);
bias = randn([nodes 1]);
eval = subs(eval,b,bias);
times = randn([1 discretizedTime]);
eval = subs(eval,t,times);
init_betas = randn([nodes*(numberStates+numberCostates) 1]);

