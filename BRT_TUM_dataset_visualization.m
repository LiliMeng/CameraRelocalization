clear all;
close all;



cameraSize=0.01;

interval=5;
count=1;
for i = 1:interval:944
  
  poseRTFileName = sprintf('/media/lci/storage/Thesis/TUM_data/pose/sitting_halfsphere/BT1_estimated_poses_point/camera_%06d.txt', i-1);
 
  
  fid_poseRT=fopen(poseRTFileName); 
  PoseRT_tmp = textscan(fid_poseRT, '%f %f %f %f ',4,'HeaderLines',3, 'delimiter', '\n')


  PoseRT=cell2mat(PoseRT_tmp)
 
  
  for j=1:3
      for k=1:3
          R(j,k)=PoseRT(j,k);
      end
  end
  
  for m=1:3
      
      T(m)=PoseRT(m,4);
  end    
  
  totalT(count,:)=T;
  
  if count>=2

    pts = [totalT(count-1,:); totalT(count,:)];
     
    line(pts(:,1), pts(:,2), pts(:,3),'Color','r','LineWidth',1.5, 'LineStyle','-')

  end
  
   cam = plotCamera('Location',T,'Orientation',R,'Size', cameraSize, ...
     'Color', 'r', 'Opacity', 0);
   grid on
   axis equal
   axis manual
   xlabel('X (m)');
   ylabel('Y (m)');
   zlabel('Z (m)');

  hold on;
  count=count+1
  
 % pause(0.5);
  drawnow();
  

end

count1=1;
for i1 = 1:interval:944
  poseRTFileName1 = sprintf('/media/lci/storage/Thesis/TUM_data/rgbd_dataset_freiburg3_sitting_halfsphere_validation/pose/frame-%06d.pose.txt', i1-1);
  PoseRT1=load(poseRTFileName1);
 
  
  for j=1:3
      for k=1:3
          R1(j,k)=PoseRT1(j,k);
      end
  end
  
  for m=1:3
      T1(m)=PoseRT1(m,4);
  end    
  
  totalT1(count1,:)=T1;
  
  if count1>=2

    pts1 = [totalT1(count1-1,:); totalT1(count1,:)];
     
    line(pts1(:,1), pts1(:,2), pts1(:,3),'Color','b','LineWidth',1.5, 'LineStyle','-')

  end
  
   cam1 = plotCamera('Location',T1,'Orientation',R1,'Size', cameraSize, ...
     'Color', 'b', 'Opacity', 0.3);
   grid on
   axis equal
   axis manual
   xlabel('X (m)');
   ylabel('Y (m)');
   zlabel('Z (m)');

   hold on;
   count1=count1+1
 % pause(0.5);
   drawnow();
  
end
