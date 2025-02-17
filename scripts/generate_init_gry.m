function init = generate_init_gry(data,NumDim)
%%%%%%%  codes of initialization with automatic target generation process (ATGP)

t = zeros(size(data,1));
b = zeros(1,size(data,2));
c = zeros(1,size(data,2));
for j = 1:size(data,2)
    c(j) = data(:,j)'*data(:,j); 
end
index = find(c==max(c));
index = index(1);
t(:,1) = data(:,index);
U = t(:,1); 
% PU = eye(26)- U*inv(U'*U)*U';

for i = 2:NumDim
    PU = eye(size(data,1))- U*pinv(U'*U)*U';
    data2 = PU * data;
    for k = 1:size(data,2)
        b(k) = data2(:,k)'*data2(:,k); 
    end
    index = find(b==max(b));
    t(:,i)= data(:,index);
    clear data2;
    U = [U t(:,i)];
end

init = U;
