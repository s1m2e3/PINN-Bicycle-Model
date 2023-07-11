%% Define Hamiltonian
function [xDot yDot thetaDot deltaDot lambdaXDot lambdaYDot lambdaThetaDot lambdaDeltaDot]=dynamicEquations(V,Phi,theta,delta,lambdaX,lambdaY,lambdaTheta,lambdaDelta,m,tS,cL)
size(V)
size(Phi)
size(lambdaX)
size(lambdaDelta)
H =  1/2*Phi.^2 + 1/2*m*V.^2+(V-tS).^2+lambdaX.*V.*cos(theta)+lambdaY*V*sin(theta)+lambdaTheta/cL.*V.*tan(delta) ...
+lambdaDelta*Phi;
H = sym(H);
%% Define dynamic equations for states and co - states
xDot = diff(H,lambdaX);
yDot = diff(H,lambdaY);
thetaDot = diff(H,lambdaTheta);
deltaDot = diff(H,lambdaDelta);
lambdaXDot= diff(-H,x);
lambdaYDot = diff(-H,y);
lambdaThetaDot = diff(-H,theta);
lambdaDeltaDot = diff(-H,delta);
end
