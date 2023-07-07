%% Parameters: mass => m, targetSpeed => tS, carLength => cL, define controls in term of states and co-states
function [V,Phi] = controlsDefinition (m,tS,cL)
syms V(theta, delta,lambdaX,lambdaY,lambdaTheta,lambdaDelta) Phi(lambdaDelta) ;
V(theta, delta,lambdaX,lambdaY,lambdaTheta,lambdaDelta) = -(-2*tS+lambdaX*cos(theta)+lambdaY*sin(theta)+lambdaTheta/cL*tan(delta))/(m+2); 
Phi(lambdaDelta) =  - lambdaDelta;
end