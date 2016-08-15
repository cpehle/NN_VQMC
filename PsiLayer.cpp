#include "PsiLayer.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

PsiLayer::PsiLayer(MTRand& random, int nIn, int nF) 
{
    
    n_in = nIn;
    n_f = nF;

    Z.setZero(n_f,n_in);
    
    double bound = 4.0 * sqrt(6.0/(1 + n_in));
    double r;

    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_f; i++) {

            r = random.rand(bound);
            Z.real()(i,k) = 2.0 * r - bound;
                
            r = random.rand(bound);
            Z.imag()(i,k) = 2.0 * r - bound;
        }
    }
}


//VectorXcd PsiLayer::flattenHiddenMatrix(MatrixXcd & h_matrix) {
//
//    VectorXcd h_vector(h_matrix.rows()*h_matrix.cols());
//    
//    for (int i=0;i<h_matrix.rows(); i++) {
//        for (int k=0; k<h_matrix.cols(); k++) {
//            h_vector(i*h_matrix.cols()+k) = h_matrix(i,k);
//        }
//    }
//    
//    return h_vector;
//
//}


//*****************************************************************************
// Forward Pass 
//*****************************************************************************

complex<double> PsiLayer::forward_pass(MatrixXcd & input) {
    
    complex<double> pre_activation;
    complex<double> psi;
    
    //cout << "Total Input Dimension  " <<input.rows()*input.cols() << endl;
    
    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_f; i++) {

            pre_activation += Z(i,k) * input(k,i);
        }
    }


    //for (int j = 0; j<h_vector.rows(); j++) {
    //    pre_activation += Z(j) * h_vector(j);
    //}

    //
    psi = 1.0/(1.0+exp(-pre_activation));

    return psi;

}


////*****************************************************************************
//// Save the Network Parameters
////*****************************************************************************
//
//void PsiLayer::loadParameters(string& modelName) 
//{
//        
//    ifstream file(modelName);
//
//    for (int j=0; j<n_in; j++) {
//        file >> Z.real()(j);
//        file >> Z.imag()(j);
//    }
//}
//
//
////*****************************************************************************
//// Save the Network Parameters
////*****************************************************************************
//
//void PsiLayer::saveParameters(string& modelName) 
//{
//
//    ofstream file(modelName);
//
//    for (int j=0; j<n_in; j++) {
//        file << Z.real()(j) << " ";
//        file << Z.imag()(j) << " ";
// 
//    }
//    
//    file.close(); 
//
//}


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
