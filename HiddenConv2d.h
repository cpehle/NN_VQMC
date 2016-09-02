#ifndef HIDDENCONV2D_H
#define HIDDENCONV2D_H

#include "PsiFC.cpp"
using namespace std;

class HiddenLayer {
    
    public:
        
        //Network Parameters
        int n_f;          //M
        int f_size;        //l
        int n_c;
        double bound;
        int L;

        vector<vector<double> > W;
        vector<vector<double> > b;

        vector<vector<int> > LocalityTable; 

        // Constructor
        HiddenLayer(MTRand & random, int nF,int L_, double B);
         
        // Functions
        vector<vector<double> > forward_pass(vector<int> & input);
        void buildTable_NNsquare();
        void loadParameters(string& modelName);
        void saveParameters(string& modelName);
        int index(int x, int y); 
};

#endif
