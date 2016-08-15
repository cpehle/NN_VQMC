#include "PsiLayer.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

PsiLayer::PsiLayer(MTRand& random, int nIn) 
{
    
    n_in = nIn;

    Z.setZero(n_in);
    
    double bound = 4.0 * sqrt(6.0/(1 + n_in));
    double r;

    for (int j=0; j<n_in;j++) {

        r = random.rand(bound);
        Z.real()(j) = 2.0 * r - bound;
            
        r = random.rand(bound);
        Z.imag()(j) = 2.0 * r - bound;

    }
}


VectorXcd PsiLayer::flattenHiddenMatrix(MatrixXcd & h_matrix) {

    VectorXcd h_vector(h_matrix.rows()*h_matrix.cols());
    
    for (int i=0;i<h_matrix.rows(); i++) {
        for (int k=0; k<h_matrix.cols(); k++) {
            h_vector(i*h_matrix.cols()+k) = h_matrix(i,k);
        }
    }
    
    return h_vector;

}


//*****************************************************************************
// Forward Pass 
//*****************************************************************************

complex<double> PsiLayer::forward_pass(VectorXcd & h_vector) {
    
    complex<double> pre_activation;
    complex<double> psi;
    
    //cout << "Total Input Dimension  " <<input.rows()*input.cols() << endl;
    for (int j = 0; j<h_vector.rows(); j++) {
        pre_activation += Z(j) * h_vector(j);
    }

    //
    psi = 1.0/(1.0+exp(-pre_activation));

    return psi;

}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void PsiLayer::loadParameters(string& modelName) 
{
        
    ifstream file(modelName);

    for (int j=0; j<n_in; j++) {
        file >> Z.real()(j);
        file >> Z.imag()(j);
    }
}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void PsiLayer::saveParameters(string& modelName) 
{

    ofstream file(modelName);

    for (int j=0; j<n_in; j++) {
        file << Z.real()(j) << " ";
        file << Z.imag()(j) << " ";
 
    }
    
    file.close(); 

}


//*****************************************************************************
// Sigmoid function
//*****************************************************************************

//complex<double> PsiLayer::sigmoid(complex<double> pre_act) {
////
//    //VectorXcd X(vector.rows());
//    
//    //for (int i=0; i< X.rows(); i++) {
//    //        X(i) = 1.0/(1.0+exp(-vector(i)));
//    //}
//
//    //return X;
//    return 1.0/(1.0+exp(-pre_act));
//}
//
