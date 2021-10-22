clear
clc

f = @(x,y,z) [0.1250.*(1-x).*(1-y).*(1-z),...
              0.1250.*(1+x).*(1-y).*(1-z),...
              0.1250.*(1-x).*(1+y).*(1-z),...  % <--- 4th shape function
              0.1250.*(1+x).*(1+y).*(1-z),...  % <--- 3th shape function
              0.1250.*(1-x).*(1-y).*(1+z),...
              0.1250.*(1+x).*(1-y).*(1+z),...
              0.1250.*(1-x).*(1+y).*(1+z),...  % <--- 8th shape function
              0.1250.*(1+x).*(1+y).*(1+z)];    % <--- 7th shape function
syms x y z

dfx = diff(f,x);
dfy = diff(f,y);
dfz = diff(f,z);

fx = matlabFunction(dfx);
fy = matlabFunction(dfy);
fz = matlabFunction(dfz);

P=2;
Q=2;
[B1d, D1d, W, qref1d] = FEcreateBasis(P,Q, 'GAUSS');
gs = abs(qref1d(1));
% NOTE the ordering:      
%       1  2   4  3  5  6  8 7  <--lexographic ordering 
x = gs*[-1  1 -1  1 -1  1 -1 1]';
y = gs*[-1 -1  1  1 -1 -1  1 1]';
z = gs*[-1 -1 -1 -1  1  1  1 1]';

% analytical evaluation
D0= fx(y,z);
D1= fy(x,z);
D2= fz(x,y);

B = reshape(B1d,Q,P)';
D = reshape(D1d,Q,P)';

disp('B @ B @ D - D0:')
kron(B,kron(B,D)) - D0

ne = 1;
dof = 1;
tensorI = eye(P*P*P);
tensorY = 0 * tensorI;
for i=1:size(tensorI,2)
    X = tensorI(:,i);
    tensorY(:,i) = tensorloop(ne,dof,P,Q,B1d,B1d,D1d,X);
end
disp('tensorloop(B, B, D)- D0')
tensorY - D0



