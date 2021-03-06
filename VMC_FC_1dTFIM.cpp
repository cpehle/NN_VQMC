#include "VMC_FC_1dTFIM.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

VariationalMC::VariationalMC(MTRand& random, int L_, double h_,
                             int MCS_, double lr_, int ep,
                             PsiLayer& PL, HiddenLayer& HL) 
{
    
    lr = lr_;
    MCS = MCS_;
    L = L_;
    epochs = ep;
    eq = int(ep/10);

    J = 1.0;
    h = h_;
    
    for (int i=0; i<L; i++) {
        if (random.rand() > 0.5)
            spins.push_back(1);
        else
            spins.push_back(-1);
    }
   
    vector<double> temp;
    temp.assign(HL.n_h,0.0);
     
    for (int k=0; k< HL.n_in; k++) {
        dP_dW.push_back(temp);
        dP_dW_e.push_back(temp); 
    }

    dP_db.assign(HL.n_h,0.0);
    dP_db_e.assign(HL.n_h,0.0);
    
    dP_dZ.assign(PL.n_in,0.0);
    dP_dZ_e.assign(PL.n_in,0.0);
 
}

void VariationalMC::reset(PsiLayer& PL, HiddenLayer& HL) {

   
    E = 0.0;
    
    for (int i=0; i<HL.n_h; i++) {
        for (int k=0; k< HL.n_in; k++) {
            dP_dW[k][i]   = 0.0;
            dP_dW_e[k][i] = 0.0;
        }
        dP_db[i]   = 0.0;
        dP_db_e[i] = 0.0;
    }
    
    for (int i=0; i<PL.n_in; i++) {
        dP_dZ[i]   = 0.0;
        dP_dZ_e[i] = 0.0;
    }

    dP_dc   = 0.0;
    dP_dc_e = 0.0;
     
}

double VariationalMC::getPSI(PsiLayer& PL, HiddenLayer& HL) {
    
    vector<double> h;
    double psi;
    
    h = HL.forward_pass(spins);
    psi = PL.forward_pass(h);

    return psi;
}


double VariationalMC::getLocalEnergy(PsiLayer& PL, HiddenLayer& HL) {
    
    double e = 0.0;
    double psi;
    double psi_flip;
    
    psi = getPSI(PL,HL);

    for (int i=0; i<L-1; i++) {
        e += -J * spins[i]*spins[i+1];
        spins[i] *= -1;
        psi_flip = getPSI(PL,HL);
        e += -h*psi_flip/psi;
        spins[i] *= -1;
    }
    spins[L-1] *= -1;
    psi_flip = getPSI(PL,HL);
    e += -h*psi_flip/psi;
    spins[L-1] *= -1;
 
    return e;
}


void VariationalMC::updateObservables(PsiLayer& PL, HiddenLayer& HL) {

    vector<double> h;
    double psi;
    double eL;
    
    h   = HL.forward_pass(spins);
    psi = PL.forward_pass(h);
    eL  = getLocalEnergy(PL,HL);

    for (int i=0;i<PL.n_in; i++) {
        dP_dZ[i]   += (1.0-psi)*h[i];
        dP_dZ_e[i] += (1.0-psi)*h[i]*eL;
        
    }

    dP_dc   += (1.0-psi);
    dP_dc_e += (1.0-psi)*eL;

    for (int k=0; k<HL.n_in; k++) {
        for (int i=0; i<HL.n_h; i++) {
            dP_dW[k][i]   += (1.0-psi)*h[i]*(1.0-h[i])*PL.Z[i]*spins[k];
            dP_dW_e[k][i] += (1.0-psi)*h[i]*(1.0-h[i])*PL.Z[i]*spins[k]*eL;
        }
    }
    
    for (int i=0; i<HL.n_h; i++) {
        dP_db[i]   += (1.0-psi)*h[i]*(1.0-h[i])*PL.Z[i];
        dP_db_e[i] += (1.0-psi)*h[i]*(1.0-h[i])*PL.Z[i]*eL;
    }

    E += eL;

}

void VariationalMC::updateParameters(PsiLayer& PL, HiddenLayer& HL) {

    
    double mcs = 1.0*MCS;

    for (int i=0; i<PL.n_in; i++) {
            PL.Z[i] += -(2.0*lr)*(dP_dZ_e[i]/mcs-dP_dZ[i]*E/(mcs*mcs));
        }
        
        PL.c += -(2.0*lr)*(dP_dc_e/mcs-dP_dc*E/(mcs*mcs));
        
        for (int k=0; k<HL.n_in; k++) {
            for (int i=0; i<HL.n_h; i++) {
               HL.W[k][i] += -(2.0*lr)*(dP_dW_e[k][i]/mcs-dP_dW[k][i]*E/(mcs*mcs)); 
            }
        }

        for (int i=0; i<HL.n_h; i++) {
            HL.b[i] += -(2.0*lr)*(dP_db_e[i]/mcs-dP_db[i]*E/(mcs*mcs)); 
        }
        
}

void VariationalMC::MC_run(MTRand& random, PsiLayer& PL, HiddenLayer& HL){
    
    int site;
    double q;
    double psi;
    double psi_prime;
    
    for (int n=0; n<eq; n++) {
        for (int k = 0; k< L; k++) {
            site = random.randInt(L-1);
            psi = getPSI(PL,HL);
            spins[site] *= -1;
            psi_prime = getPSI(PL,HL);
            q = (psi_prime*psi_prime) / (psi*psi);
            
            if (random.rand() > q) {
                spins[site] *= -1;
            }
        }
    }

    for (int n=0; n<MCS; n++) {
        for (int k = 0; k< L; k++) {
            site = random.randInt(L-1);
            psi = getPSI(PL,HL);
            spins[site] *= -1;
            psi_prime = getPSI(PL,HL);
            q = (psi_prime*psi_prime) / (psi*psi);
             
            if (random.rand() > q) {
                spins[site] *= -1;
            }
        }
        updateObservables(PL,HL);
    }
}

void VariationalMC::train(MTRand & random, ofstream & file,
                          PsiLayer& PL, HiddenLayer& HL) 
{
    
    int lr_switch1 = int(epochs / 10);
    int lr_switch2 = int(epochs / 2);
    int record_frequency = 100;

    for (int k=1; k<lr_switch1+1; k++) {
        reset(PL,HL);
        MC_run(random,PL,HL);
        updateParameters(PL,HL);
        Energy = E/(1.0*MCS*L);
        if ((k%record_frequency) == 0)
            file << k << "      " << Energy << endl;
        //cout << "Epoch: " << k << "   Ground State Energy: " << Energy << endl; 
    }
    
    lr /= 2.0;

    for (int k=lr_switch1; k<lr_switch2+1; k++) {
        reset(PL,HL);
        MC_run(random,PL,HL);
        updateParameters(PL,HL);
        Energy = E/(1.0*MCS*L);
        if ((k%record_frequency) == 0)
            file << k << "      " << Energy << endl;
        //cout << "Epoch: " << k << "   Ground State Energy: " << Energy << endl; 
    }

    lr /= 5.0;

    for (int k=lr_switch2; k<epochs+1; k++) {
        reset(PL,HL);
        MC_run(random,PL,HL);
        updateParameters(PL,HL);
        Energy = E/(1.0*MCS*L);
        if ((k%record_frequency) == 0)
            file << k << "      " << Energy << endl;
        //cout << "Epoch: " << k << "   Ground State Energy: " << Energy << endl; 
    }


}

////*****************************************************************************
//// Print Network Informations
////*****************************************************************************
//
void VariationalMC::printNetwork(PsiLayer& PL, HiddenLayer& HL) 
{

    cout << "\n\n******************************\n\n" << endl;
    cout << "NEURAL NETWORK VARIATIONAL QUANTUM MONTE CARLO\n\n";
    cout << "Machine Parameter\n\n";
    cout << "\tNumber of Inputs Units: " << L << "\n";
    cout << "\nNumber of Hidden Units: " << HL.n_h << "\n";
    cout << "\tMagnetic Field: " << h << "\n";
    cout << "\nHyper-parameters\n\n";
    cout << "\tLearning Rate: " << lr << "\n";
    cout << "\tEpochs: " << epochs << "\n";
    cout << "\tMonte Carlo Steps: " << MCS << "\n";
    cout << "\tInitial distribution width: " << HL.bound << "\n";
    
    
 
}
