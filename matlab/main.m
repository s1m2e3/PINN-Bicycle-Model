%% Define mass, car length and desired speed
m = 1500;
cL = 4.9;
tS = 20;
%% Define total number of seconds, number of states and co-states
time = 2;
numberStates = 4;
numberCostates = 4;

%% Define Hamiltonian with respect to co-states and 
[V,Phi] = controlsDefinition (m,tS,cL);
%% get differential equations
[xDot, yDot, thetaDot, deltaDot, lambdaXDot, lambdaYDot, lambdaThetaDot, lambdaDeltaDot]= dynamicEquations(V,Phi,m,tS,cL);
%% Get predictions and add constraints
filename = "../left_turn_through.csv";
vehicleMovement = "left turning";
% vehicleMovement = "through";
discretizedTime = time*10;
nodes = floor(discretizedTime/2);
W = randn([nodes 1]);
b = randn([nodes 1]);
t = linspace(0,time,discretizedTime);
c = (1+1)/(t(end)-t(1));
t = -1+c*(t-t(1));

[predX, predY, predTheta, predDelta, predLambdaX, predLambdaY, predLambdaTheta, predLambdaDelta, dPredX, dPredY, dPredTheta, dPredDelta, dPredLambdaX, dPredLambdaY, dPredLambdaTheta, dPredLambdaDelta,betas]=predictions(numberStates,numberCostates,filename,vehicleMovement,discretizedTime,nodes,W,b,c,t);
%% Define operations and predictions
losses = getLosses(dPredX, dPredY, dPredTheta, dPredDelta, dPredLambdaX, dPredLambdaY, dPredLambdaTheta, dPredLambdaDelta, xDot, yDot, thetaDot, deltaDot, lambdaXDot, lambdaYDot, lambdaThetaDot, lambdaDeltaDot);
stored_jacobian = jacobian(losses,betas);
%% Train loop