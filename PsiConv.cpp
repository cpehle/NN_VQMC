#include "PsiConv.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

PsiLayer::PsiLayer(MTRand& random, int nIn, int nF, double B) 
{
    
    n_in = nIn;
    n_f = nF;
    bound = B;

    vector<double> temp;
    double r;

    for (int i=0; i<n_f; i++) {
        for (int k=0; k<n_in; k++) {
            r = random.rand();
            temp.push_back(bound*(2.0 * r - 1.0));
        }
        Z.push_back(temp);
        temp.clear();
    }
    c = 0.0;
}


//*****************************************************************************
// Forward Pass 
//*****************************************************************************

double PsiLayer::forward_pass(vector<vector<double> > & input) {
    
    double activation= 0.0;
    double psi;
    
    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_f; i++) {
            activation += Z[i][k] * input[k][i] + c; 
        }
    }
    psi = 1.0/(1.0+exp(-activation));

    return psi;
}

//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void PsiLayer::loadParameters(ofstream & file) 
{
    
    for (int i=0; i<n_f; i++) {
        for (int k=0; k<n_in; k++) {
            file >> Z[i][k];
        }
    }
}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void PsiLayer::saveParameters(ofstream & file) 
{

    for (int i=0; i<n_f; i++) {
        for (int k=0; k<n_in; k++) {
            file << Z[i][k] << "  ";
        }
        file << endl;
    }
    file << endl << endl;
}

