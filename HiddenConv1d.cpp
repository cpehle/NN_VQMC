#include "HiddenConv1d.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

HiddenLayer::HiddenLayer(MTRand& random, int nF, int N_, double B) 
{
    
    n_c = N_-2;
    n_f = nF;
    f_size = 3;
    bound = B;
    
    vector<double> temp;

    double r;

    for (int i=0; i<n_f; i++) {
        for (int j=0; j<f_size;j++) {
            r = random.rand();
            temp.push_back(bound*(2.0 * r - 1.0));
        }
        W.push_back(temp);
        temp.clear();
    }
    
    for (int i=0; i<n_f; i++) {
        for (int k=0; k<n_c;k++) {
            r = random.rand();
            temp.push_back(bound*(2.0 * r - 1.0));
        }
        b.push_back(temp);
        temp.clear();
    } 

    for (int i=0; i<n_c; i++) {
         temp.push_back(i);
         temp.push_back(i+1);
         temp.push_back(i+2);
         LocalityTable.push_back(temp);
         temp.clear();
    }
}

//*****************************************************************************
// Forward Pass 
//*****************************************************************************

vector<vector<double> > HiddenLayer::forward_pass(vector<int> & input) {
    
    vector<vector<double> > h;
    vector<double> temp;
    double acc;

    for (int i=0; i<n_f; i++) {
        for (int k=0; k<n_c; k++) {
            acc = 0.0;
            for (int j=0; j<f_size; j++) {
                acc += W[i][j] * input[LocalityTable[k][j]];
            }
            acc += b[i][k];
            temp.push_back(1.0/(1.0+exp(-acc)));
        }
        h.push_back(temp);
        temp.clear();
    }
    
    return h;
}


////*****************************************************************************
//// Save the Network Parameters
////*****************************************************************************
//
//void ConvLayer::loadParameters(string& modelName) 
//{
//        
//    ifstream file(modelName);
//
//    for (int i=0; i<n_h;i++) {
//        for (int j=0; j<n_in; j++) {
//            file >> W.real()(i,j);
//            file >> W.imag()(i,j);
//        }
//    }
//}
//
//
////*****************************************************************************
//// Save the Network Parameters
////*****************************************************************************
//
//void ConvLayer::saveParameters(string& modelName) 
//{
//
//    ofstream file(modelName);
//
//    for (int i=0; i<n_h;i++) {
//        for (int j=0; j<n_in; j++) {
//            file << W.real()(i,j) << " ";
//            file << W.imag()(i,j) << " ";
// 
//        }
//        file << endl;
//    }
//    
//    file.close(); 
//
//}
//
//
////*****************************************************************************
//// Sigmoid function
////*****************************************************************************

