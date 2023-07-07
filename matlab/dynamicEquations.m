%% Define Hamiltonian
function [xDot yDot thetaDot deltaDot lambdaXDot lambdaYDot lambdaThetaDot lambdaDeltaDot]=dynamicEquations(V,Phi,m,tS,cL)
syms V(theta, delta,lambdaX,lambdaY,lambdaTheta,lambdaDelta) Phi(lambdaDelta) x y ;
H= @(theta, delta,lambdaX,lambdaY, lambdaTheta,lambdaDelta) 1/2*Phi(lambdaDelta)^2 + 1/2*m*V(theta, delta,lambdaX,lambdaY,lambdaTheta,lambdaDelta)^2 ...
+(V(theta, delta,lambdaX,lambdaY,lambdaTheta,lambdaDelta)-tS)^2+lambdaX*V(theta, delta,lambdaX,lambdaY,lambdaTheta,lambdaDelta)*cos(theta) ...
+lambdaY*V(theta, delta,lambdaX,lambdaY,lambdaTheta,lambdaDelta)*sin(theta)+lambdaTheta/cL*V(theta, delta,lambdaX,lambdaY,lambdaTheta,lambdaDelta)*tan(delta) ...
+lambdaDelta*Phi(lambdaDelta);
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
