function losses = getLosses(dPredXThrough, dPredYThrough, dPredThetaThrough, dPredDeltaThrough, dPredLambdaXThrough, dPredLambdaYThrough, dPredLambdaThetaThrough, dPredLambdaDeltaThrough, xDotThrough, yDotThrough, thetaDotThrough, deltaDotThrough, lambdaXDotThrough, lambdaYDotThrough, lambdaThetaDotThrough, lambdaDeltaDotThrough,dPredXTurning, dPredYTurning, dPredThetaTurning, dPredDeltaTurning, dPredLambdaXTurning, dPredLambdaYTurning, dPredLambdaThetaTurning, dPredLambdaDeltaTurning, xDotTurning, yDotTurning, thetaDotTurning, deltaDotTurning, lambdaXDotTurning, lambdaYDotTurning, lambdaThetaDotTurning, lambdaDeltaDotTurning)

%% Define losses
lossesXThrough = dPredXThrough - xDotThrough; 
lossesYThrough = dPredYThrough - yDotThrough;
lossesThetaThrough = dPredThetaThrough - thetaDotThrough;
lossesDeltaThrough = dPredDeltaThrough - deltaDotThrough;
lossesLambdaXThrough  = dPredLambdaXThrough - lambdaXDotThrough;
lossesLambdaYThrough  = dPredLambdaYThrough - lambdaYDotThrough;
lossesLambdaThetaThrough  = dPredLambdaThetaThrough - lambdaThetaDotThrough;
lossesLambdaDeltaThrough  = dPredLambdaDeltaThrough - lambdaDeltaDotThrough;
lossesXTurning = dPredXTurning - xDotTurning; 
lossesYTurning = dPredYTurning - yDotTurning;
lossesThetaTurning = dPredThetaTurning - thetaDotTurning;
lossesDeltaTurning = dPredDeltaTurning - deltaDotTurning;
lossesLambdaXTurning  = dPredLambdaXTurning - lambdaXDotTurning;
lossesLambdaYTurning  = dPredLambdaYTurning - lambdaYDotTurning;
lossesLambdaThetaTurning  = dPredLambdaThetaTurning - lambdaThetaDotTurning;
lossesLambdaDeltaTurning  = dPredLambdaDeltaTurning - lambdaDeltaDotTurning;

losses=cat(1, lossesXThrough, lossesYThrough, lossesThetaThrough, lossesDeltaThrough, lossesLambdaXThrough, lossesLambdaYThrough, lossesLambdaThetaThrough, lossesLambdaDeltaThrough,lossesXTurning, lossesYTurning, lossesThetaTurning, lossesDeltaTurning, lossesLambdaXTurning, lossesLambdaYTurning, lossesLambdaThetaTurning, lossesLambdaDeltaTurning);

end 