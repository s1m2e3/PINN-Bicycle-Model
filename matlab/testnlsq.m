%% 40 works well
nodes = 40;
numberStates = 4;
numberCoStates = 4;
initBetas = randn(nodes,numberStates+numberCoStates);
W= randn(nodes,1);
b= randn(nodes,1);
t= linspace(-1,1,100);
c = 2/10;
file = "../left_turn_through.csv";
vehicleMovement = "through";
discretizedTime = size(t);
discretizedTime = discretizedTime(2);
%% Read tablecollocollocationPointsXcationPointsX
table = readtable(file);
rows = table.path == vehicleMovement;
vehicleTable = table(rows,:); 

xCar = vehicleTable.utmX - 504000;
yCar = vehicleTable.utmY - 3566000;
dxCar = vehicleTable.vx;
dyCar = vehicleTable.vy;
thetaCar = vehicleTable.heading;
indices = int32(vehicleTable.relative_proportion*discretizedTime);
indices(1) = indices(1)+1;
%% Count collocation points
countPosX = 0;
countSpeedX = 0;
countPosY = 0;
countSpeedY = 0;
countPosTheta = 0;
for i = 1:size(indices)
    if xCar(i) ~= 0
        countPosX = countPosX + 1;
    end
    if yCar(i) ~= 0
        countPosY = countPosY + 1;
    end
    if dxCar(i) ~= 0
        countSpeedX = countSpeedX + 1;
    elseif i == 1 
        countSpeedX = countSpeedX + 1;
    end
    if dyCar(i) ~= 0
        countSpeedY = countSpeedY + 1;
    elseif i == 1
        countSpeedY = countSpeedY + 1;
    end
    countPosTheta = countPosTheta + 1; 
end 

%% Find switching functions for collocation points for X and Y 

collocationPointsX = countPosX + countSpeedX;
collocationPointsY = countPosY + countSpeedY;
collocationPointsTheta = countPosTheta;

syms supportT [collocationPointsX collocationPointsX] 
syms freeSupportFunction [discretizedTime collocationPointsX]
syms dFreeSupportFunction [discretizedTime collocationPointsX]

for i = 1:collocationPointsX
    supportT(:,i) = supportT(:,i).^(i-1);
    freeSupportFunction(:,i) = freeSupportFunction(:,i).^(i-1);
end

for i=1:collocationPointsX
    for j=countPosX+1:countPosX+countSpeedX
        supportT(j,i) = diff(supportT(j,i));
    end
    for j=1:discretizedTime
        dFreeSupportFunction(j,i) = diff(freeSupportFunction(j,i));
    end
    
end

for i=1:countPosX
    for j =1:collocationPointsX
        supportT(i,j) = subs(supportT(i,j),t(indices(i)));
    end
end
for i=1:countSpeedX
    for j=1:collocationPointsX
        supportT(countPosX+i,j) = subs(supportT(countPosX+i,j),t(indices(i)));
    end
end

for i=1:discretizedTime
    for j=1:collocationPointsX
        freeSupportFunction(i,j)=subs(freeSupportFunction(i,j),t(i));
        dFreeSupportFunction(i,j)=subs(dFreeSupportFunction(i,j),t(i));
        
    end
end

supportT = double(supportT);
coefficientsMatrix = inv(supportT);
freeSupportFunction = double(freeSupportFunction);
dFreeSupportFunction = double(dFreeSupportFunction);
phi = freeSupportFunction*coefficientsMatrix;
dPhi = dFreeSupportFunction*coefficientsMatrix;


%% Define operations and predictions for X and Y

xSubstractionsPhi = getPosSubstractions(0,W,b,t,indices,phi,countPosX);
dXSubstractionsPhi = getDPosSubstractions(0,W,b,t,indices,phi,countPosX,countSpeedX);

xAdditionsPhi = getPosAdditions(0,xCar,phi,countPosX);
dXAdditionsPhi = getDPosAdditions(0,dxCar,phi,countPosX,countSpeedX)/c;


xSubstractionsDPhi = getPosSubstractions(0,W,b,t,indices,dPhi,countPosX);
dXSubstractionsDPhi = getDPosSubstractions(0,W,b,t,indices,dPhi,countPosX,countSpeedX);

xAdditionsDPhi = getPosAdditions(0,xCar,dPhi,countPosX);
dXAdditionsDPhi = getDPosAdditions(0,dxCar,dPhi,countPosX,countSpeedX)/c;

ySubstractionsPhi = getPosSubstractions(0,W,b,t,indices,phi,countPosY);
dYSubstractionsPhi = getDPosSubstractions(0,W,b,t,indices,phi,countPosY,countSpeedY);

yAdditionsPhi = getPosAdditions(0,yCar,phi,countPosY);
dYAdditionsPhi = getDPosAdditions(0,dyCar,phi,countPosY,countSpeedY)/c;

ySubstractionsDPhi = getPosSubstractions(0,W,b,t,indices,dPhi,countPosY);
dYSubstractionsDPhi = getDPosSubstractions(0,W,b,t,indices,dPhi,countPosY,countSpeedY);

yAdditionsDPhi = getPosAdditions(0,yCar,dPhi,countPosY);
dYAdditionsDPhi = getDPosAdditions(0,dyCar,dPhi,countPosY,countSpeedY)/c;

%% Find switching functions for theta 

syms supportT [collocationPointsTheta collocationPointsTheta] 
syms freeSupportFunction [discretizedTime collocationPointsTheta]
syms dFreeSupportFunction [discretizedTime collocationPointsTheta]


for i = 1:collocationPointsTheta
    supportT(:,i) = supportT(:,i).^(i-1);
    freeSupportFunction(:,i) = freeSupportFunction(:,i).^(i-1);
end

for i=1:collocationPointsTheta
    
    for j=1:discretizedTime
        dFreeSupportFunction(j,i) = diff(freeSupportFunction(j,i));
    end
    
end

for i=1:countPosTheta
    for j =1:collocationPointsTheta
        supportT(i,j) = subs(supportT(i,j),t(indices(i)));
    end
end

for i=1:discretizedTime
    for j=1:collocationPointsTheta
        freeSupportFunction(i,j)=subs(freeSupportFunction(i,j),t(i));
        dFreeSupportFunction(i,j)=subs(dFreeSupportFunction(i,j),t(i));
        
    end
end

supportT = double(supportT);
coefficientsMatrix = inv(supportT);
freeSupportFunction = double(freeSupportFunction);
dFreeSupportFunction = double(dFreeSupportFunction);
phi = freeSupportFunction*coefficientsMatrix;
dPhi = dFreeSupportFunction*coefficientsMatrix;


thetaSubstractionsPhi = getPosSubstractions(0,W,b,t,indices,phi,countPosTheta);
thetaAdditionsPhi = getPosAdditions(0,thetaCar,phi,countPosTheta);

thetaSubstractionsDPhi = getPosSubstractions(0,W,b,t,indices,dPhi,countPosTheta);
thetaAdditionsDPhi = getPosAdditions(0,thetaCar,dPhi,countPosTheta);

options = optimoptions(@lsqnonlin, "Algorithm","levenberg-marquardt");
options.Display = "iter";
for i=1:3
    [finalBetas, resnorm] = lsqnonlin(@(betas) newgetLosses(betas,W,b,t,xSubstractionsPhi,xSubstractionsDPhi,dXSubstractionsPhi,dXSubstractionsDPhi,xAdditionsPhi,dXAdditionsPhi,xAdditionsDPhi,dXAdditionsDPhi,ySubstractionsPhi,ySubstractionsDPhi,dYSubstractionsPhi,dYSubstractionsDPhi,yAdditionsPhi,dYAdditionsPhi,yAdditionsDPhi,dYAdditionsDPhi,thetaSubstractionsPhi,thetaSubstractionsDPhi,thetaAdditionsPhi,thetaAdditionsDPhi),initBetas,[],[],options);
    initBetas= finalBetas;
end
function losses=newgetLosses(betas,W,b,t,xSubstractionsPhi,xSubstractionsDPhi,dXSubstractionsPhi,dXSubstractionsDPhi,xAdditionsPhi,dXAdditionsPhi,xAdditionsDPhi,dXAdditionsDPhi,ySubstractionsPhi,ySubstractionsDPhi,dYSubstractionsPhi,dYSubstractionsDPhi,yAdditionsPhi,dYAdditionsPhi,yAdditionsDPhi,dYAdditionsDPhi,thetaSubstractionsPhi,thetaSubstractionsDPhi,thetaAdditionsPhi,thetaAdditionsDPhi)

h = tanh(W*t+b);
dh = (1-(tanh(W*t+b)).^2).*W;
c = 2/10;
discretizedTime = size(t);
discretizedTime = discretizedTime(2);

l=5;
m=1500;

betasX = betas(:,1);
betasY = betas(:,2);
betasTheta = betas(:,3);
betasDelta = betas(:,4);
betasLambdaX = betas(:,5);
betasLambdaY = betas(:,6);
betasLambdaTheta = betas(:,7);
betasLambdaDelta = betas(:,8);

x = (h-xSubstractionsPhi-dXSubstractionsPhi)'*betasX+xAdditionsPhi+dXAdditionsPhi;
y = (h-ySubstractionsPhi-dYSubstractionsPhi)'*betasY+yAdditionsPhi+dYAdditionsPhi;
dX = c*((dh-xSubstractionsDPhi-dXSubstractionsDPhi)'*betasX+xAdditionsDPhi+dXAdditionsDPhi);
dY = c*((dh-ySubstractionsDPhi-dYSubstractionsDPhi)'*betasY+yAdditionsDPhi+dYAdditionsDPhi);


theta= (h-thetaSubstractionsPhi)'*betasTheta+thetaAdditionsPhi;
dTheta = c*((dh-thetaSubstractionsDPhi)'*betasTheta+thetaAdditionsDPhi);
delta = h'*betasDelta;
lambdaX = h'*betasLambdaX;
lambdaY = h'*betasLambdaY;
lambdaTheta = h'*betasLambdaTheta;
lambdaDelta = h'*betasLambdaDelta;

dDelta = dh'*betasDelta;
dLambdaX = dh'*betasLambdaX;
dLambdaY = dh'*betasLambdaY;
dLambdaTheta = dh'*betasLambdaTheta;
dLambdaDelta = dh'*betasLambdaDelta;


dPredX = dX+lambdaX.*cos(theta).^2/m+lambdaY.*cos(theta).*sin(theta)/m-lambdaTheta.*cos(theta).*tan(delta)/(l*m);
dPredY = dY+lambdaY.*sin(theta).^2/m+lambdaX.*cos(theta).*sin(theta)/m-lambdaTheta.*sin(theta).*tan(delta)/(l*m);
dPredTheta = dTheta+lambdaTheta.*tan(delta).^2/(l^2*m)+lambdaX.*cos(theta).*tan(delta)/(l*m)-lambdaY.*tan(delta).*sin(theta)/(l*m);
dPredDelta = dDelta + lambdaDelta;
dPredLambdaX = dLambdaX;
dPredLambdaY = dLambdaY;
dPredLambdaTheta = dLambdaTheta -(lambdaX.*cos(theta).*(lambdaY.*cos(theta)-lambdaX.*sin(theta))/m ...
    -(lambdaY.*cos(theta)-lambdaX.*sin(theta)).*(lambdaX.*cos(theta)+lambdaY.*sin(theta)+lambdaTheta.*tan(delta)/l)/m ...
    +lambdaY.*sin(theta).*(lambdaY.*cos(theta)-lambdaX.*sin(theta))/m + lambdaY.*cos(theta).*(lambdaX.*cos(theta)+lambdaY.*sin(theta)+lambdaTheta.*tan(delta)/l)/m ...
    - lambdaX.*sin(theta).*(lambdaX.*cos(theta)+lambdaY.*sin(theta)+lambdaTheta.*tan(delta)/l)/m ...
    + lambdaTheta.*tan(delta).*(lambdaY.*cos(theta)-lambdaX.*sin(theta))/(l*m)); 
dPredLambdaDelta = dLambdaDelta - (lambdaTheta.^2.*tan(delta).*(tan(delta).^2+1)/(l^2*m)+ ...
    lambdaX.*lambdaTheta.*cos(theta).*(tan(delta).^2+1)/(l*m)+ ...
    lambdaY.*lambdaTheta.*sin(theta).*(tan(delta).^2+1)/(l*m));

losses= [dPredX dPredY dPredTheta dPredDelta dPredLambdaX dPredLambdaY dPredLambdaTheta dPredLambdaDelta];

% losses= mean(mean(losses.^2));

end
%% Extra functions
function h=getH(W,b,t)
h = tanh(W*t+b);
end
%%
function dh=getDh(W,b,t)
dh = (1-getH(W,b,t).^2).*W;
end
%%
function PosSubstractions = getPosSubstractions(PosSubstractions,W,b,t,indices,phiMult,countPos)
    for i=1:countPos
        PosSubstractions= getH(W,b,t(indices(i)))*phiMult(:,i)' + PosSubstractions;
    end
end
%%
function DPosSubstractions = getDPosSubstractions(DPosSubstractions,W,b,t,indices,phiMult,countPos,countDPos)
    for i=countPos+1:countPos+countDPos
        DPosSubstractions = getDh(W,b,t(indices(i-countPos)))*phiMult(:,i)' + DPosSubstractions;
    end
end

%%
function PosAdditions= getPosAdditions(PosAdditions,pos,phiMult,countPos)
    for i=1:countPos
        PosAdditions = pos(i)*phiMult(:,i) + PosAdditions;
    end
end
%%
function dPosAdditions = getDPosAdditions(dPosAdditions,dPos,phiMult,countPos,countDPos)
    for i=countPos+1:countPos+countDPos
        dPosAdditions = dPos(i-countPos)*phiMult(:,i) + dPosAdditions;
    end
end