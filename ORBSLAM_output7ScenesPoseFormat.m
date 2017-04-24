clear all;
close all;

CamTrajectory= load('/home/lci/workspace/ORB_SLAM2/CameraTrajectory.txt');

N=length(CamTrajectory)



for i = 1:N
  PoseTQ=CamTrajectory(i,2:8)
  PoseRT=PoseTQ2PoseRT(PoseTQ)
  poseRTFileName = sprintf('/media/lci/storage/Thesis/RealData/trial1/pose/frame-%06d.pose.txt', i-1);
  fileID = fopen(poseRTFileName,'w');
  for j=1:4
    fprintf(fileID, '%.10f %.10f %.10f %10f \n', PoseRT(j,1), PoseRT(j,2), PoseRT(j,3), PoseRT(j,4));
  end
  fclose(fileID);
end

