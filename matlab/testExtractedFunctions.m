lossesThroughNArguments = nargin("lossesThrough");
lossesTurningNArguments = nargin("lossesTurning");
%  
%  for i =1:lossesThroughNArguments
%      eval(['betas',int2str(i),'=0;'])
%  end
%  for i=1:1000
%      evalLossesThrough = lossesThrough(betas1,betas2,betas3,betas4,betas5,betas6,betas7,betas8,betas9,betas10,betas11,betas12,betas13,betas14,betas15,betas16,betas17,betas18,betas19,betas20,betas21,betas22,betas23,betas24,betas25,betas26,betas27,betas28,betas29,betas30,betas31,betas32,betas33,betas34,betas35,betas36,betas37,betas38,betas39,betas40,betas41,betas42,betas43,betas44,betas45,betas46,betas47,betas48);
%      evalJacobianThrough = jacobianThrough(betas13,betas14,betas15,betas16,betas17,betas18,betas19,betas20,betas21,betas22,betas23,betas24,betas25,betas26,betas27,betas28,betas29,betas30,betas31,betas32,betas33,betas34,betas35,betas36,betas37,betas38,betas39,betas40,betas41,betas42);
%      delta = pinv(evalJacobianThrough)*evalLossesThrough;
%      for j =1:lossesThroughNArguments
%          eval(['betas',int2str(j),'=-0.01*delta(',int2str(j),')','+betas',int2str(j),';']);
%      end
%      mean(evalLossesThrough.^2)
% end


for i =1:lossesTurningNArguments
    eval(['betas',int2str(i),'=1;'])
end
for i=1:10
evalLossesTurning = lossesTurning(betas1,betas2,betas3,betas4,betas5,betas6,betas7,betas8,betas9,betas10,betas11,betas12,betas13,betas14,betas15,betas16,betas17,betas18,betas19,betas20,betas21,betas22,betas23,betas24,betas25,betas26,betas27,betas28,betas29,betas30,betas31,betas32,betas33,betas34,betas35,betas36,betas37,betas38,betas39,betas40,betas41,betas42,betas43,betas44,betas45,betas46,betas47,betas48);
evalJacobianTurning = jacobianTurning(betas1,betas2,betas3,betas4,betas5,betas6,betas7,betas8,betas9,betas10,betas11,betas12,betas13,betas14,betas15,betas16,betas17,betas18,betas19,betas20,betas21,betas22,betas23,betas24,betas25,betas26,betas27,betas28,betas29,betas30,betas31,betas32,betas33,betas34,betas35,betas36,betas37,betas38,betas39,betas40,betas41,betas42);
delta = pinv(evalJacobianTurning)*evalLossesTurning;
    for j =1:lossesTurningNArguments
        eval(['betas',int2str(j),'=-0.001*delta(',int2str(j),')','+betas',int2str(j),';']);
    end
    mean(evalLossesTurning.^2)
end