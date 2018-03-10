function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%%%%%%%%%%%CALCULATE layer 1 %%%%%%%%%%%%
%add base unit to X=m*401

X=[ones(size(X,1),1),X];
%m*401
a1=X;

%%%%%%%%%%%CALCULATE layer 2 %%%%%%%%%%%%
%Theta1=25*401; z2= m*25
z2=a1*Theta1';

a2=sigmoid(z2);

%add base unit a2=m*26
a2=[ones(size(a2,1),1),a2];
%%%%%%%%%%%CALCULATE layer 3%%%%%%%%%%%%
%Theta2= 10*26
z3=a2*Theta2';

%a3=m*10
a3=sigmoid(z3);

[value,indices]=max(a3,[],2);

p=indices;
% =========================================================================


end
