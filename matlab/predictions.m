function [predX predY predTheta predDelta predLambdaX predLambdaY predLambdaTheta predLambdaDelta dPredX dPredY dPredTheta dPredDelta dPredLambdaX dPredLambdaY dPredLambdaTheta dPredLambdaDelta betas]=predictions(numberStates,numberCostates,file,vehicleMovement,discretizedTime,nodes,W,b,c,t)
%% Define predictions number of nodes => n, time in seconds => t , discretized time => dT in 0.1 s each point
syms betas [nodes*(numberStates+numberCostates) 1] 

%% Read tablecollocollocationPointsXcationPointsX
table = readtable(file);
rows = table.path == vehicleMovement;
vehicleTable = table(rows,:); 

x = vehicleTable.utmX;
y = vehicleTable.utmY;
dx = vehicleTable.vx;
dy = vehicleTable.vy;
ddx = vehicleTable.ax;
ddy = vehicleTable.ay;
theta = vehicleTable.heading;
indices = int32(vehicleTable.relative_proportion*discretizedTime);
indices(1) = indices(1)+1;
%% Count collocation points
countPosX = 0;
countSpeedX = 0;
countAccelX = 0;
countPosY = 0;
countSpeedY = 0;
countAccelY = 0;
countPosTheta = 0;
for i = 1:size(indices)
    if x(i) ~= 0
        countPosX = countPosX + 1;
    end
    if y(i) ~= 0
        countPosY = countPosY + 1;
    end
    if dx(i) ~= 0
        countSpeedX = countSpeedX + 1;
    elseif i == 1 
        countSpeedX = countSpeedX + 1;
    end
    if dy(i) ~= 0
        countSpeedY = countSpeedY + 1;
    elseif i == 1
        countSpeedY = countSpeedY + 1;
    end
    if ddx(i) ~= 0
        countAccelX = countAccelX + 1;
    elseif i == 1
        countAccelX = countAccelX + 1;
    end
    if ddy(i) ~= 0
        countAccelY = countAccelY + 1;
    elseif i == 1
        countAccelY = countAccelY + 1;
    end
    countPosTheta = countPosTheta + 1; 
end 

%% Define main predictions
h =  getH(W,b,t);
dh = getDh(W,b,t);
dhh = getDdh(W,b,t);
%% Find switching functions for collocation points for X and Y 

collocationPointsX = countPosX + countSpeedX + countAccelX;
collocationPointsY = countPosY + countSpeedY + countAccelY;
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
    
    for j=countPosX+countSpeedX+1:countPosX+countSpeedX+countAccelX
        supportT(j,i) = diff(supportT(j,i),2);
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
for i=1:countAccelX
    for j=1:collocationPointsX
        supportT(countPosX+countSpeedX+i,j) = subs(supportT(countPosX+countSpeedX+i,j),t(indices(i)));
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
ddXSubstractionsPhi = getDDPosSubstractions(0,W,b,t,indices,phi,countPosX,countSpeedX,countAccelX);

xAdditionsPhi = getPosAdditions(0,x,phi,countPosX);
dXAdditionsPhi = getDPosAdditions(0,dx,phi,countPosX,countSpeedX)/c;
ddXAdditionsPhi = getDDPosAdditions(0,ddx,phi,countPosX,countSpeedX,countAccelX)/c^2;

xSubstractionsDPhi = getPosSubstractions(0,W,b,t,indices,dPhi,countPosX);
dXSubstractionsDPhi = getDPosSubstractions(0,W,b,t,indices,dPhi,countPosX,countSpeedX);
ddXSubstractionsDPhi = getDDPosSubstractions(0,W,b,t,indices,dPhi,countPosX,countSpeedX,countAccelX);

xAdditionsDPhi = getPosAdditions(0,x,dPhi,countPosX);
dXAdditionsDPhi = getDPosAdditions(0,dx,dPhi,countPosX,countSpeedX)/c;
ddXAdditionsDPhi = getDDPosAdditions(0,ddx,dPhi,countPosX,countSpeedX,countAccelX)/c^2;


ySubstractionsPhi = getPosSubstractions(0,W,b,t,indices,phi,countPosY);
dYSubstractionsPhi = getDPosSubstractions(0,W,b,t,indices,phi,countPosY,countSpeedY);
ddYSubstractionsPhi = getDDPosSubstractions(0,W,b,t,indices,phi,countPosY,countSpeedY,countAccelY);

yAdditionsPhi = getPosAdditions(0,y,phi,countPosY);
dYAdditionsPhi = getDPosAdditions(0,dy,phi,countPosY,countSpeedY)/c;
ddYAdditionsPhi = getDDPosAdditions(0,ddy,phi,countPosY,countSpeedY,countAccelY)/c^2;

ySubstractionsDPhi = getPosSubstractions(0,W,b,t,indices,dPhi,countPosY);
dYSubstractionsDPhi = getDPosSubstractions(0,W,b,t,indices,dPhi,countPosY,countSpeedY);
ddYSubstractionsDPhi = getDDPosSubstractions(0,W,b,t,indices,dPhi,countPosY,countSpeedY,countAccelY);

yAdditionsDPhi = getPosAdditions(0,y,dPhi,countPosY);
dYAdditionsDPhi = getDPosAdditions(0,dy,dPhi,countPosY,countSpeedY)/c;
ddYAdditionsDPhi = getDDPosAdditions(0,ddy,dPhi,countPosY,countSpeedY,countAccelY)/c^2;

predX = (h-xSubstractionsPhi-dXSubstractionsPhi-ddXSubstractionsPhi)'*betas(1:nodes)+xAdditionsPhi+dXAdditionsPhi+ddXAdditionsPhi;

for i = indices
    h(i)-xSubstractionsPhi(i)-dXSubstractionsPhi(i)-ddXSubstractionsPhi(i)
end

predY = (h-ySubstractionsPhi-dYSubstractionsPhi-ddYSubstractionsPhi)'*betas(nodes+1:2*nodes)+yAdditionsPhi+dYAdditionsPhi+ddYAdditionsPhi;
dPredX = c*(dh-xSubstractionsDPhi-dXSubstractionsDPhi-ddXSubstractionsDPhi)'*betas(1:nodes)+xAdditionsDPhi+dXAdditionsDPhi+ddXAdditionsDPhi;

dPredY = c*(dh-ySubstractionsDPhi-dYSubstractionsDPhi-ddYSubstractionsDPhi)'*betas(nodes+1:2*nodes)+yAdditionsDPhi+dYAdditionsDPhi+ddYAdditionsDPhi;

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
thetaAdditionsPhi = getPosAdditions(0,theta,phi,countPosTheta);

thetaSubstractionsDPhi = getPosSubstractions(0,W,b,t,indices,dPhi,countPosTheta);
thetaAdditionsDPhi = getPosAdditions(0,theta,dPhi,countPosTheta);

predTheta = (h-thetaSubstractionsPhi)'*betas(nodes*2+1:3*nodes)+thetaAdditionsPhi;
dPredTheta = (dh-thetaSubstractionsDPhi)'*betas(nodes*2+1:3*nodes)+thetaAdditionsDPhi;

predDelta = h'*betas(nodes*3+1:4*nodes);
predLambdaX = h'*betas(nodes*4+1:5*nodes);
predLambdaY = h'*betas(nodes*5+1:6*nodes);
predLambdaTheta = h'*betas(nodes*6+1:7*nodes);
predLambdaDelta = h'*betas(nodes*7+1:8*nodes);


dPredDelta = dh'*betas(nodes*3+1:4*nodes);
dPredLambdaX = dh'*betas(nodes*4+1:5*nodes);
dPredLambdaY = dh'*betas(nodes*5+1:6*nodes);
dPredLambdaTheta = dh'*betas(nodes*6+1:7*nodes);
dPredLambdaDelta = dh'*betas(nodes*7+1:8*nodes);

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
function ddh= getDdh(W,b,t)
ddh = -getDh(W,b,t).*W;
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
function DDPosSubstractions = getDDPosSubstractions ( DDPosSubstractions,W,b,t,indices,phiMult,countPos,countDPos,countDDPos)
    for i=countPos+countDPos+1:countPos+countDPos+countDDPos
        DDPosSubstractions= getDdh(W,b,t(indices(i-countPos-countDPos)))*phiMult(:,i)' + DDPosSubstractions;
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
%%
function dDPosAdditions = getDDPosAdditions( dDPosAdditions,dDPos,phiMult,countPos,countDPos,countDDPos)
    for i=countPos+countDPos+1:countPos+countDPos+countDDPos
        dDPosAdditions= dDPos(i-countPos-countDPos)*phiMult(:,i) + dDPosAdditions;
    end
end


