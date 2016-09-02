#include <stdlib.h>
#include <time.h>
#include <fstream>
//#include "VMC_FC_1dTFIM.cpp"
#include "HiddenConv2d.cpp"

int main(int argc, char* argv[]) {
   
    
    map<string,string> Helper;
    map<string,float> Parameters;
    
    initializeParameters(Parameters);
    
    //string model = argv[1];
    //string network = argv[2];

    //get_option("h","Magnetic Field",argc,argv,Parameters,Helper);
    //get_option("L","System Size",argc,argv,Parameters,Helper);
    //get_option("lr","Learning Rate",argc,argv,Parameters,Helper);
    ////get_option("w","Initial parameters width distribution",argc,argv,Parameters,Helper);
    //Parameters["w"]   = 1.0;
    //Parameters["ep"]  = 10000;
    //Parameters["MCS"] = 100;
    //
    MTRand random(1234);
    HiddenLayer HL(random,1,4,1.0);
    HL.buildTable_NNsquare();
    //
    //PsiLayer PL(random,Parameters["nH"],Parameters["w"]);


    //if (network.compare("FCNet")==0) {
    //    
    //    get_option("nH","Number of Hidden Units",argc,argv,Parameters,Helper);
    //    
    //    HiddenLayer HL(random,Parameters["L"],Parameters["nH"],Parameters["w"]);
    //    
    //    VariationalMC VMC(random,Parameters["L"],Parameters["h"],
    //                         Parameters["MCS"],Parameters["lr"],
    //                         Parameters["ep"],PL,HL); 
 
    //    string rawFileName = buildOutputNameRaw(network,model,Parameters);
    //    ofstream fout(rawFileName);
    //    
    //    VMC.printNetwork(PL,HL); 
    //    VMC.train(random,fout,PL,HL);
 
    //}

    //if (network.compare("ConvNet")==0) {
    //    get_option("nF","Number of Filters",argc,argv,Parameters,Helper);
    //}
        
        
    
}
