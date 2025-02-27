clear;clc;

%% load data and site labels
labels = readtable('Classification_site_labels.csv');
labels = labels{:,:};
data = readtable('morphometry_data.csv');

%% tSNE with the default values of the matlab function "tsne" 
InitY_tSNE = generate_init_gry(data{:,:},2); % initialization with the automatic target generation process (ATGP)
Y_tSNE = tsne(data{:,:},'Algorithm','barneshut','Distance','euclidean','NumPCAComponents',0,...
'verbose',1,'InitialY',InitY_tSNE,'Perplexity',30,'Standardize',false);
%% plot the 2d tsne results
customColors = [0.2 0 1; 0.8 0.5 0; 0.8 0.25 0; 1 0 0; 1 0.2 1;...
    0 1 1; 0.3 0.6 1; 0.2 1 0; 0 1 0.6; 0 0.4 0.2];
figure;gscatter(Y_tSNE(:,1),Y_tSNE(:,2),[labels],customColors,[],10);
axis square;


%% UMAP with the default values of the matlab function "run_umap" provided by https://www.mathworks.com/matlabcentral/fileexchange/71902-uniform-manifold-approximation-and-projection-umap 
[Y_UMAP, umap, clusterIdentifiers, extras] = run_umap(data{:,:},'randomize','FALSE');
%% plot the 2d UMAP results
customColors = [0.2 0 1; 0.8 0.5 0; 0.8 0.25 0; 1 0 0; 1 0.2 1;...
    0 1 1; 0.3 0.6 1; 0.2 1 0; 0 1 0.6; 0 0.4 0.2];
figure;gscatter(Y_UMAP(:,1),Y_UMAP(:,2),[labels],customColors,[],10);
axis square;
