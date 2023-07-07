function losses = getLosses(dPredX, dPredY, dPredTheta, dPredDelta, dPredLambdaX, dPredLambdaY, dPredLambdaTheta, dPredLambdaDelta, xDot, yDot, thetaDot, deltaDot, lambdaXDot, lambdaYDot, lambdaThetaDot, lambdaDeltaDot)

%% Define losses
lossesX = dPredX - xDot; 
lossesY = dPredY - yDot;
lossesTheta = dPredTheta- thetaDot;
lossesDelta = dPredDelta - deltaDot;
lossesLambdaX = dPredLambdaX - lambdaXDot;
lossesLambdaY = dPredLambdaY - lambdaYDot;
lossesLambdaTheta = dPredLambdaTheta - lambdaThetaDot;
lossesLambdaDelta = dPredLambdaDelta - lambdaDeltaDot;
losses=cat(1, lossesX, lossesY, lossesTheta, lossesDelta, lossesLambdaX, lossesLambdaY, lossesLambdaTheta, lossesLambdaDelta);

end 