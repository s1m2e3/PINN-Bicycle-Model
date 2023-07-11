function [predXTurning,predYTurning,dPredXTurning,dPredYTurning] = couple(predXThrough,dPredXThrough,predYThrough,dPredYThrough,predXTurning,dPredXTurning,predYTurning,dPredYTurning,distanceThreshold,k,conflictX,conflictY)
    
    evalXTurning = double(subs(predXTurning,symvar(predXTurning),randn(size(symvar(predXTurning)))));
    evalYTurning = double(subs(predYTurning,symvar(predYTurning),randn(size(symvar(predYTurning)))));
    
    lowerXBound = evalXTurning-0.5;
    upperXBound = evalXTurning+0.5;
    lowerYBound = evalYTurning-0.5;
    upperYBound = evalYTurning+0.5;

    lowerPathX = lowerXBound - predXTurning;
    lowerPathY = lowerYBound - predYTurning;
    heavisideLowerPathX = getHeaviside(lowerXBound-predXTurning,k);
    heavisideLowerPathY = getHeaviside(lowerYBound-predYTurning,k);
    upperPathX = upperXBound - predYTurning;
    upperPathY = upperYBound - predYTurning;
    heavisideUpperPathX = getHeaviside(predXTurning-upperXBound,k);
    heavisideUpperPathY = getHeaviside(predYTurning-upperYBound,k);

    predXTurning = predXTurning + (lowerPathX).*heavisideLowerPathX + (upperPathX).*heavisideUpperPathX;
    dPredXTurning = dPredXTurning + (-dPredXTurning).*heavisideLowerPathX + (-dPredXTurning).*heavisideUpperPathX;
    predYTurning = predYTurning + (lowerPathY).*heavisideLowerPathY + (upperPathY).*heavisideUpperPathY;
    dPredXTurning = dPredYTurning + (-dPredYTurning).*heavisideLowerPathX + (-dPredYTurning).*heavisideUpperPathX;

    minDist = -predXThrough+predXTurning-predYThrough+predYTurning;
    heavisideMinDist = getHeaviside(distanceThreshold-minDist,k);
    region = getHeaviside(predXTurning-conflictX,k);
    region = region.*heavisideMinDist;
    
    yChange = distanceThreshold + predXThrough + predYThrough - predXTurning;
    dYChange = dPredXThrough + dPredYThrough - dPredXTurning;
    xChange = distanceThreshold + predXThrough + predYThrough - predYTurning;
    dXChange = dPredXThrough + dPredYTurning-dPredYTurning;

    predXTurning = predXTurning + (xChange-predXTurning).*region;
    dPredXTurning = dPredXTurning + (dXChange-dPredXTurning).*region;
    predYTurning = predYTurning + (yChange-predYTurning).*region;
    dPredXTurning = dPredYTurning + (dYChange-dPredYTurning).*region;

end
function heaviside = getHeaviside(x,k)
    heaviside = 1/2 +1/2*tanh(k*x);
end