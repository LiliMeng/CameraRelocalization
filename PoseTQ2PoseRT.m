function PoseRT=PoseTQ2PoseRT(PoseTQ)
% PoseTQ2PoseRT: get a 4x4 Pose transformation matrix for 7Scenes format
% from 7x1 Pose transformation from TUM dataset format
% PoseRT=PoseTQ2PoseRT(PoseTQ)
% IN: 
%     PoseTQ - 7x1 Pose transformation from TUM dataset format 
% 
% OUT:
%     R -  4x4 Pose transformation matrix for 7Scenes format
%     
% Author: Lili Meng, July 8th, 2016
%close all;
%clear all;
%PoseTQ=load('/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.pose.quaternion.txt');
Q=PoseTQ(4:7);
R=quat2rotm(Q);


T=transpose(PoseTQ(1:3));
PoseRT(:,:)=R;
PoseRT(:,4)=T;

for i = 1:3
    PoseRT(4,i)=0;
end

PoseRT(4,4)=1;
