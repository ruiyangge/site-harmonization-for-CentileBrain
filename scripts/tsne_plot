clear;clc;

%% load data and site labels
labels = readtable('Classification_site_labels.csv');
labels = labels{:,:};
data = readtable('SVC_data.csv');

%% tSNE with the default values of the matlab function "tsne" 
InitY = generate_init_gry(data{:,:},2); % initialization with the automatic target generation process (ATGP)
Y = tsne(data{:,:},'Algorithm','barneshut','Distance','euclidean','NumPCAComponents',0,...
'verbose',1,'InitialY',InitY,'Perplexity',30,'Standardize',false);

%% plot the 2d tsne results
customColors = [0.2 0 1; 0.8 0.5 0; 0.8 0.25 0; 1 0 0; 1 0.2 1;...
    0 1 1; 0.3 0.6 1; 0.2 1 0; 0 1 0.6; 0 0.4 0.2];
figure;gscatter(Y(:,1),Y(:,2),[labels],customColors,[],10);
axis square;
