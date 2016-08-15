#ifndef UTILITIES_H
#define UTITLITES_H

#include <Eigen/Core>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <map>
#include <boost/format.hpp>

using namespace std;

//*****************************************************************************
// Get command line options
//*****************************************************************************

void initializeParameters(map<string,float>& par) 
{
    par["nV"] = 0;
    par["nH"] = 0;
    par["lr"] = 0;
    par["ep"] = 0;

}

//*****************************************************************************
// Get command line options
//*****************************************************************************

void get_option(const string& arg, const string& description,
                int argc, char** argv, 
                map<string,float>& par, map<string,string>& helper)
{
    string flag = "--" + arg ;
    for (int i=3; i<argc; ++i) {
            
        if (flag.compare(argv[i]) ==0) {

            par[arg] = atof(argv[i+1]);
            break;
        }
    }

    helper[arg] = description;
}


//*****************************************************************************
// Generate the Base Name of the simulation
//*****************************************************************************

string buildBaseName(const string& network, const string& model,
                     map<string,float>& par,
                     const string& CD_id, const string& Reg_id) 
{

    string baseName = network;
     
    baseName += "_nH";
    baseName += boost::str(boost::format("%.0f") % par["nH"]);
    baseName += "_ep";
    baseName += boost::str(boost::format("%.0f") % par["ep"]);
    baseName += "_lr";
    baseName += boost::str(boost::format("%.3f") % par["lr"]);
    baseName += "_";
    baseName += model;
    baseName += "_L";
    baseName += boost::str(boost::format("%d") % par["nV"]);
    
    return baseName;
}


//*****************************************************************************
// Generate the Name of the model file
//*****************************************************************************

string buildModelName(const string& network, const string& model,
                     map<string,float>& par,
                     const string& CD_id, const string& Reg_id) 
{
    
    string modelName = "data/networks/";
    modelName += buildBaseName(network,model,par,CD_id,Reg_id); 
    modelName += "_p";
    modelName += boost::str(boost::format("%.3f") % par["p"]);
    modelName += "_model.txt";
 
    return modelName;
}

//*****************************************************************************
// Generate the Name of the output file
//*****************************************************************************

string buildAccuracyName(const string& network, const string& model,
                     map<string,float>& par,string set,
                     const string& CD_id, const string& Reg_id) 
{
    
    int L = int(sqrt(par["nV"]/2));
    string accuracyName = "data/measurements/L";
    accuracyName += boost::str(boost::format("%d") % L);
    accuracyName += "/";
    accuracyName += buildBaseName(network,model,par,CD_id,Reg_id); 
    accuracyName += "_p";
    accuracyName += boost::str(boost::format("%.3f") % par["p"]);
    accuracyName += "_" + set + "_Accuracy.txt";
 
    return accuracyName;
}


//*****************************************************************************
// Print Matrix or Vector on the screen
//*****************************************************************************

template<typename T> 
ostream& operator<< (ostream& out, const Eigen::MatrixBase<T>& M)
{    
    for (size_t i =0; i< M.rows(); ++i) {
        
        for (size_t j =0; j< M.cols(); ++j) {
            
            out << M(i,j)<< " ";
        }
        
        if (M.cols() > 1) out << endl;
    }
    
    out << endl;

    return out;
}


//*****************************************************************************
// Write Matrix or Vector on file 
//*****************************************************************************

template<typename T> 
void write (ofstream& fout,const Eigen::MatrixBase<T>& M)
{
    for (size_t i =0; i< M.rows(); ++i) {
        
        for (size_t j =0; j< M.cols(); ++j) {
            
            fout << M(i,j)<< " ";
        }
        
        if (M.cols() > 1) fout << endl;
    }
    
    fout << endl;
}


//*****************************************************************************
// Apply sigmoid function to an array 
//*****************************************************************************

template<typename T> Eigen::ArrayXXd sigmoidTEMP(const Eigen::ArrayBase<T>& M)
{
    return M.exp()/(1.0+M.exp());

}


#endif
