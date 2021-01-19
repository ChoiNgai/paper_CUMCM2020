clear all
close all
addpath data
addpath ClustFunction
addpath evaluate
data = xlsread('P1_paper_data.xlsx');
label = xlsread('P1_paper_label.xlsx');
cluster_n = max(label);
data = (data - min(data)) ./ (max(data) - min(data));

for i = 1:30        %标记样本数
    for j = 1:10    %循环次数
    [c] = kmeans(data,cluster_n);
    [~,kmeans_RI(j,i),~,~]=RandIndex(c,label);
    
    [~,U] = fcm(data,cluster_n);
    [~,c] = max(U);
    [~,FCM_RI(j,i),~,~]=RandIndex(c,label);
    
    [U,center] = SFCM(data,cluster_n,label(1:i));
    [~,c] = max(U);
    [~,SFCM_RI(j,i),~,~]=RandIndex(c,label);
    
    [U,center] = sSFCM(data,cluster_n,label(1:i));
    [~,c] = max(U);
    [~,sSFCM_RI(j,i),~,~]=RandIndex(c,label);
    
    [U,center] = SMUC(data,cluster_n,label(1:i));
    [~,c] = max(U);
    [~,SMUC_RI(j,i),~,~]=RandIndex(c,label);
    
    [U,center] = SMUKC(data,cluster_n,label(1:i));
    [~,c] = max(U);
    [SMUKC_AR(i),SMUKC_RI(j,i),SMUKC_MI(i),SMUKC_HI(i)]=RandIndex(c,label);
    end
end

figure(1)
hold on 
kmeans_RI = zeros( length(kmeans_RI),1 ) + mean(mean(kmeans_RI));
plot(kmeans_RI,'linewidth',1)
plot(mean(FCM_RI),'linewidth',1)
plot(mean(SFCM_RI),'linewidth',1)
plot(mean(sSFCM_RI),'linewidth',1)
plot(mean(SMUC_RI),'linewidth',1)
plot(mean(SMUKC_RI),'-o','color',[1 0 0],'linewidth',2)
legend('kmeans','FCM','SFCM','sSFCM','SMUC','SMUKC')
xlabel("标记样本数")
ylabel("RI")
h = legend("kmeans","FCM","SFCM","sSFCM","SMUC","SMUKC",'location','SouthOutside','NumColumns',3);
set(h,'Box','off')
