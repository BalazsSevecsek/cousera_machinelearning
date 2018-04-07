function [J_test] = testSetError(X_test,y_test,theta)
  J_test = linearRegCostFunction(X_test,y_test,theta,0);
end