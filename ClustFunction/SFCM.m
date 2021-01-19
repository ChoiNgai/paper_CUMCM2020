function [U,center] = SFCM(data, cluster_n,data_label)
data_n = size(data, 1); % ���data�ĵ�һά(rows)��,����������
in_n = size(data, 2);   % ���data�ĵڶ�ά(columns)����������ֵ����
% Ĭ�ϲ�������
options = [2; % �����Ⱦ���U��ָ��
   100;                 % ����������
   1e-5;               % ��������С�仯��,������ֹ����
   1];                 % ÿ�ε����Ƿ������Ϣ��־

%��options �еķ����ֱ�ֵ���ĸ�����;
expo = options(1);          % �����Ⱦ���U��ָ��
max_iter = options(2);  % ����������
min_impro = options(3);  % ��������С�仯��,������ֹ����
display = options(4);  % ÿ�ε����Ƿ������Ϣ��־

obj_fcn = zeros(max_iter, 1); % ��ʼ���������obj_fcn
%��ʼ��F
[~,F] = initcenter(data_label,data,cluster_n);
%��ʼ��center(���ѡȡ������Ϊ��ʼ��������)
rn = randperm(size(F,2));
rn = rn(1:cluster_n);
center0 = data(rn,:);


% ��ʼ��ģ���������,ʹU�����������Ϊ1,cluster_n=2,�û�����ȥ��������c=cluster_n
a=5;%5��6��

% �����ʼ�������Ⱦ���
U = rand(size(F));
U = U./sum(U);

% Main loop  ��Ҫѭ��
for i = 1:max_iter
   if i==1
       dist = distfcm(center0, data); 
       mf = U.^expo; 
       obj_fcn(1)=sum(sum((dist.^2).*mf))+a*sum(sum((dist.^2).*((U-F).^expo)));
   %�ڵ�k��ѭ���иı��������ceneter,�ͷ��亯��U��������ֵ;
   else
       [U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo,a,F);
   end
   if display
      fprintf('SFCM:Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
   end
% ��ֹ�����б�
   if i>1
     if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro
           break;
     end
   end
end

%iter_n = i; % ʵ�ʵ�������
%obj_fcn(iter_n+1:max_iter) = [];
%[~,j]=find(U<0);
%U(:,j)=[];
%[V_pc,~,V_pe,V_xb] = V_pcpexb(U,data,center)
end


function [center,F] = initcenter(data_label,data,cluster_n)%Ĭ��Ϊ��3��
center=zeros(cluster_n,size(data, 2));%����Ҫ��
F=zeros(cluster_n,size(data, 1));%����Ҫ��
for k=1:cluster_n
 for i=1:size(data_label,1)%center��һ�У���һ��
   if data_label(i,1)==k
       F(k,i)=1;
       for j=2:size(data_label,2)-1
          center(k,j)=(data_label(i,j)+center(k,j))/i;
       end
   end
 end
end

% for i=1:size(data_label,1)%center��2�У���2��
%     if data_label(i,1)==2
%         F(2,i)=1;
%         for j=2:size(data_label,2)-1
%            center(2,j)=(data_label(i,j)+center(2,j))/i;
%         end
%     end
% end
%    
% for i=1:size(data_label,1)%center��3�У���3��
%     if data_label(i,1)==3
%         F(3,i)=1;
%         for j=2:size(data_label,2)-1
%            center(3,j)=(data_label(i,j)+center(3,j))/i;
%         end
%     end
% end
  
end


% �Ӻ���
function U = initfcm(cluster_n,data,center0,F,a,expo)%a=6
% ��ʼ��fcm�������Ⱥ�������
% ����:
%   cluster_n   ---- �������ĸ���
%   data_n      ---- ��������
% �����
%   U           ---- ��ʼ���������Ⱦ���
% U = rand(cluster_n, data_n);
% col_sum = sum(U);                      
% U = U./col_sum(ones(cluster_n, 1), :);%��һ��
dist = distfcm(center0, data); 
tmp = dist.^(-2/(expo-1)); 
%U= tmp./(ones(cluster_n, 1)*sum(tmp));
U_fcm= tmp./(ones(cluster_n, 1)*sum(tmp));
U_3 =(a/(1+a))* U_fcm.*(ones(cluster_n,1)*sum(F));
U=U_fcm+(a/(1+a))*F-U_3;
end


% �Ӻ���
function [U_new, center, obj_fcn] = stepfcm(data,U,cluster_n, expo,a,F)
% ģ��C��ֵ����ʱ������һ��
% ���룺
%   data        ---- nxm����,��ʾn������,ÿ����������m��ά����ֵ
%   U           ---- �����Ⱦ���
%   cluster_n   ---- ����,��ʾ�ۺ�������Ŀ,�������
%   expo        ---- �����Ⱦ���U��ָ��                     
% �����
%   U_new       ---- ������������µ������Ⱦ���
%   center      ---- ������������µľ�������
%   obj_fcn     ---- Ŀ�꺯��ֵ
mf = U.^expo;       % �����Ⱦ������ָ��������
center = mf*data./((ones(size(data, 2), 1)*sum(mf'))'); % �¾�������(7)ʽ
     % ����������
dist = distfcm(center, data);
tmp = dist.^(-2/(expo-1));    
U_fcm= tmp./(ones(cluster_n, 1)*sum(tmp));
U_3 =(a/(1+a))* U_fcm.*((ones(cluster_n,1)*sum(F)));
U_new=U_fcm+(a/(1+a))*F-U_3;
obj_fcn =sum(sum((dist.^2).*mf))+a*sum(sum((dist.^2).*((U-F).^expo)));  % ����Ŀ�꺯��ֵ (4)ʽ

end


% �Ӻ���
function out = distfcm(center, data)
% �������������������ĵľ���
% ���룺
%   center     ---- ��������
%   data       ---- ������
% �����
%   out        ---- ����
out = zeros(size(center, 1), size(data, 1));
 for k = 1:size(center, 1) % ��ÿһ����������
   % ÿһ��ѭ��������������㵽һ���������ĵľ���
   out(k, :) = sqrt(sum(((data-ones(size(data,1),1)*center(k,:)).^2)',1));
 end
end