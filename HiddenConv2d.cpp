#include "HiddenConv2d.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

HiddenLayer::HiddenLayer(MTRand& random, int nF, int L_, double B) 
{
    L = L_;
    n_c = L*L;
    n_f = nF;
    f_size = 9;
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
        for (int k=0; k<n_c; k++) {
            r = random.rand();
            temp.push_back(bound*(2.0 * r - 1.0));
        }
        b.push_back(temp);
        temp.clear();
    }

}

//*****************************************************************************
// Forward Pass 
//*****************************************************************************

vector<vector<double> > ConvLayer::forward_pass(vector<int> & input) {
    
    vector<vector<double> > h;
    vector<double> temp;
    double acc;
    
    for (int i=0; i<n_f; i++) {
        for (int k=0; k<n_in; k++) {
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



void HiddenLayer::buildTable_NNsquare() {
    
    
    int counter[2] = {0,0};  
    vector<int> temp;

    vector<vector<int> > Coordinates;
    Coordinates.resize(n_c,vector<int>(2));
    vector<vector<int> > Neighbors;
    Neighbors.resize(n_c,vector<int>(4));

    for (int i=1;i<n_c;i++){
        
        if ( (counter[0]+1) % L == 0){ //end of x-row
            
            counter[0]=0;
            
                if ( (counter[1]+1) % L == 0)
                    counter[1] = 0; //reset
                else{
                    counter[1]++;
                    //break;
                }
        }//if
        else {
            counter[0]++;
        }
        
        Coordinates[i][0]=counter[0];
        Coordinates[i][1]=counter[1];
    }
    
    //Neighbours
    for(int i=0;i<n_c;i++) {
        
        Neighbors[i][0]=index(Coordinates[i][0]+1,Coordinates[i][1]);
        Neighbors[i][1]=index(Coordinates[i][0]  ,Coordinates[i][1]+1);
        Neighbors[i][2]=index(Coordinates[i][0]-1,Coordinates[i][1]);
        Neighbors[i][3]=index(Coordinates[i][0]  ,Coordinates[i][1]-1);
    }
     
    //for(int i=0;i<n_c;i++) {
    //    for (int j=0;j<4;j++) {        
    //        cout << Neighbors[i][j] << "  ";
    //    }
    //    cout << endl;
    //}
    //cout << endl << endl;
    
    //for (int k=0; k<n_c; k++) {

    //    temp.push_back(k);
    //    temp.push_back(Neighbors[k][0]);
    //    temp.push_back(Neighbors[Neighbors[k][0]][0]);
    //    temp.push_back(Neighbors[k][1]);
    //    temp.push_back(Neighbors[Neighbors[k][1]][0]);
    //    temp.push_back(Neighbors[Neighbors[Neighbors[k][1]][0]][0]);
    //    temp.push_back(Neighbors[Neighbors[k][1]][1]);
    //    temp.push_back(Neighbors[Neighbors[Neighbors[k][1]][1]][0]);
    //    temp.push_back(Neighbors[Neighbors[Neighbors[Neighbors[k][1]][1]][0]][0]);
    //    
    //    LocalityTable.push_back(temp);
    //    temp.clear();
    //}

    //
    //for (int k=0; k<n_c; k++) {
    //    for (int i=0;i<f_size;i++) {
    //        cout << LocalityTable[k][i] << "  ";
    //    }
    //    cout << endl;
    //}
    //cout << endl << endl;
     
}

//Indexing of coordinates
int HiddenLayer::index(int x, int y) {

    if (x<0) x+= L;
    if (x>=L) x-= L;
    if (y<0) y+= L;
    if (y>=L) y-= L;

    return L*y+x;

}


