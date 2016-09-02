#ifndef HIDDENCONV1D_H
#define HIDDENCONV1D_H

#include "PsiConv.cpp"

using namespace std;

class HiddenLayer {
    
    public:
        
        //Network Parameters
        int n_f;          //M
        int f_size;        //l
        int n_c;
        int N;
        double bound;

        vector<vector<double> > W;
        vector<vector<double> > b;

        vector<vector<double> > LocalityTable; 

        // Constructor
        HiddenLayer(MTRand & random, int nF, int N_, double B);
         
        // Functions
        vector<vector<double> > forward_pass(vector<int> & input);
        
        void loadParameters(string& modelName);
        void saveParameters(string& modelName);
        
};

#endif
