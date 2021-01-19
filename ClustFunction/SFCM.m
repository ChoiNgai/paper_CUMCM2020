function [U,center] = SFCM(data, cluster_n,data_label)
data_n = size(data, 1); % 求出data的第一维(rows)数,即样本个数
in_n = size(data, 2);   % 求出data的第二维(columns)数，即特征值长度
% 默认操作参数
options = [2; % 隶属度矩阵U的指数
   100;                 % 最大迭代次数
   1e-5;               % 隶属度最小变化量,迭代终止条件
   1];                 % 每次迭代是否输出信息标志

%将options 中的分量分别赋值给四个变量;
expo = options(1);          % 隶属度矩阵U的指数
max_iter = options(2);  % 最大迭代次数
min_impro = options(3);  % 隶属度最小变化量,迭代终止条件
display = options(4);  % 每次迭代是否输出信息标志

obj_fcn = zeros(max_iter, 1); % 初始化输出参数obj_fcn
%初始化F
[~,F] = initcenter(data_label,data,cluster_n);
%初始化center(随机选取样本作为初始聚类中心)
rn = randperm(size(F,2));
rn = rn(1:cluster_n);
center0 = data(rn,:);


% 初始化模糊分配矩阵,使U满足列上相加为1,cluster_n=2,用户填上去的种类数c=cluster_n
a=5;%5比6好

% 随机初始化隶属度矩阵
U = rand(size(F));
U = U./sum(U);

% Main loop  主要循环
for i = 1:max_iter
   if i==1
       dist = distfcm(center0, data); 
       mf = U.^expo; 
       obj_fcn(1)=sum(sum((dist.^2).*mf))+a*sum(sum((dist.^2).*((U-F).^expo)));
   %在第k步循环中改变聚类中心ceneter,和分配函数U的隶属度值;
   else
       [U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo,a,F);
   end
   if display
      fprintf('SFCM:Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
   end
% 终止条件判别
   if i>1
     if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro
           break;
     end
   end
end

%iter_n = i; % 实际迭代次数
%obj_fcn(iter_n+1:max_iter) = [];
%[~,j]=find(U<0);
%U(:,j)=[];
%[V_pc,~,V_pe,V_xb] = V_pcpexb(U,data,center)
end


function [center,F] = initcenter(data_label,data,cluster_n)%默认为分3类
center=zeros(cluster_n,size(data, 2));%可能要改
F=zeros(cluster_n,size(data, 1));%可能要改
for k=1:cluster_n
 for i=1:size(data_label,1)%center第一行，第一类
   if data_label(i,1)==k
       F(k,i)=1;
       for j=2:size(data_label,2)-1
          center(k,j)=(data_label(i,j)+center(k,j))/i;
       end
   end
 end
end

% for i=1:size(data_label,1)%center第2行，第2类
%     if data_label(i,1)==2
%         F(2,i)=1;
%         for j=2:size(data_label,2)-1
%            center(2,j)=(data_label(i,j)+center(2,j))/i;
%         end
%     end
% end
%    
% for i=1:size(data_label,1)%center第3行，第3类
%     if data_label(i,1)==3
%         F(3,i)=1;
%         for j=2:size(data_label,2)-1
%            center(3,j)=(data_label(i,j)+center(3,j))/i;
%         end
%     end
% end
  
end


% 子函数
function U = initfcm(cluster_n,data,center0,F,a,expo)%a=6
% 初始化fcm的隶属度函数矩阵
% 输入:
%   cluster_n   ---- 聚类中心个数
%   data_n      ---- 样本点数
% 输出：
%   U           ---- 初始化的隶属度矩阵
% U = rand(cluster_n, data_n);
% col_sum = sum(U);                      
% U = U./col_sum(ones(cluster_n, 1), :);%归一化
dist = distfcm(center0, data); 
tmp = dist.^(-2/(expo-1)); 
%U= tmp./(ones(cluster_n, 1)*sum(tmp));
U_fcm= tmp./(ones(cluster_n, 1)*sum(tmp));
U_3 =(a/(1+a))* U_fcm.*(ones(cluster_n,1)*sum(F));
U=U_fcm+(a/(1+a))*F-U_3;
end


% 子函数
function [U_new, center, obj_fcn] = stepfcm(data,U,cluster_n, expo,a,F)
% 模糊C均值聚类时迭代的一步
% 输入：
%   data        ---- nxm矩阵,表示n个样本,每个样本具有m的维特征值
%   U           ---- 隶属度矩阵
%   cluster_n   ---- 标量,表示聚合中心数目,即类别数
%   expo        ---- 隶属度矩阵U的指数                     
% 输出：
%   U_new       ---- 迭代计算出的新的隶属度矩阵
%   center      ---- 迭代计算出的新的聚类中心
%   obj_fcn     ---- 目标函数值
mf = U.^expo;       % 隶属度矩阵进行指数运算结果
center = mf*data./((ones(size(data, 2), 1)*sum(mf'))'); % 新聚类中心(7)式
     % 计算距离矩阵
dist = distfcm(center, data);
tmp = dist.^(-2/(expo-1));    
U_fcm= tmp./(ones(cluster_n, 1)*sum(tmp));
U_3 =(a/(1+a))* U_fcm.*((ones(cluster_n,1)*sum(F)));
U_new=U_fcm+(a/(1+a))*F-U_3;
obj_fcn =sum(sum((dist.^2).*mf))+a*sum(sum((dist.^2).*((U-F).^expo)));  % 计算目标函数值 (4)式

end


% 子函数
function out = distfcm(center, data)
% 计算样本点距离聚类中心的距离
% 输入：
%   center     ---- 聚类中心
%   data       ---- 样本点
% 输出：
%   out        ---- 距离
out = zeros(size(center, 1), size(data, 1));
 for k = 1:size(center, 1) % 对每一个聚类中心
   % 每一次循环求得所有样本点到一个聚类中心的距离
   out(k, :) = sqrt(sum(((data-ones(size(data,1),1)*center(k,:)).^2)',1));
 end
end