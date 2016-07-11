#include "HiddenLayer.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

HiddenLayer::HiddenLayer(MTRand& random, int nIn, int nH) 
{
    
    n_in = nIn;
    n_h = nH;

    W.setZero(n_h,n_v);
    
    double bound = 4.0 * sqrt(6.0/(n_h + n_in));
    double r;

    for (int i=0; i<n_h; i++) {
        
        for (int j=0; j<n_in;j++) {

            r = random.rand(bound);
            W(i,j) = 2.0 * r - bound;
        }
    }
}



//*****************************************************************************
// Forward Pass 
//*****************************************************************************

VectorXd HiddenLayer::forward_pass(MatrixXd input) {
    
    VectorXd pre_activation(n_h);
    VectorXd activation(n_h);

    pre_activation = v_state * W.transpose();
    activation = sigmoid(pre_activation);
    
    return activation;
}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void HiddenLayer::loadParameters(string& modelName) 
{
        
    ifstream file(modelName);

    for (int i=0; i<n_h;i++) {
        for (int j=0; j<n_in; j++) {
            file >> W(i,j);
        }
    }
}

//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void HiddenLayer::saveParameters(string& modelName) 
{

    ofstream file(modelName);

    for (int i=0; i<n_h;i++) {
        for (int j=0; j<n_in; j++) {
            file << W(i,j) << " ";
        }
        file << endl;
    }
    
    file.close(); 

}



//*****************************************************************************
// Sigmoid function
//*****************************************************************************

MatrixXd HiddenLayer::sigmoid(VectorXd vector) {

    VectorXd X(vector.rows());
    
    for (int i=0; i< X.rows(); i++) {
            X(i) = 1.0/(1.0+exp(-vector(i)));
        }
    }

    return X;

}


