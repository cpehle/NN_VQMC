import gzip
import cPickle
import numpy as np
import argparse
import glob
import math as m

#-------------------------------------------------

def get_average(data,cutoff=None):

    """ Compute averages of data """
    
    if cutoff is None:
        cutoff = 0
    n_meas = len(data)-cutoff
    
    #if len(data.shape) != 1:
    #    n_obs  = len(data[0])
    #    avg = np.zeros((n_obs))
    #    for i in range(n_meas):
    #        for j in range(n_obs):
    #            avg[j] += data[i,j]
 
    #else:
    avg = 0.0
    for i in range(n_meas):
        avg += data[cutoff+i]

    avg /= 1.0*n_meas
    return avg

#------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='model',type=str)
    parser.add_argument('network',type=str)
    parser.add_argument('-L', help='Linear size of the model',type=int)
    parser.add_argument('-nH', help='Number of hidden units',type=int)
    parser.add_argument('-g', help='Magnetic field',type=float)
    parser.add_argument('-lr',help='learning rate',type=float)
    parser.add_argument('-MCS',default=1000,type=int)
    parser.add_argument('-w',default=1.0,type=float)
    parser.add_argument('-cut',type=int)

    args = parser.parse_args()

    outputName = 'data/'
    outputName += args.model + '/'
    outputName += '/Energy_'
    outputName += args.network
    outputName += '_L'
    outputName += str(args.L)
    outputName += '_nH' + str(args.nH)
    outputName += '_lr' + str(args.lr)
    outputName += '.dat'
    
    dmrgFileName = 'data/'
    dmrgFileName += args.model + '/dmrg/Energy_dmrg_'
    dmrgFileName += args.model
    dmrgFileName += '_L' + str(args.L)
    dmrgFileName += '.dat'
    
    dmrgFile = open(dmrgFileName,'r') 

    dmrg = np.loadtxt(dmrgFile)

    outputFile = open(outputName,'w')
    
    h = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9',
         '1.0','1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9','2.0',]
    
    if (args.lr == 0.1):  lr = '0.100'
    if (args.lr == 0.01): lr = '0.010'
    
    outputFile.write('#')
    outputFile.write('h     ')
    outputFile.write('VMC         ')
    outputFile.write('DMRG       ')
    outputFile.write('Err')
    outputFile.write('\n')

    for i in range(20):

        dataFileName = 'data/'
        dataFileName += args.model + "/raw/"
        dataFileName += args.network + '_' + args.model 
        dataFileName += '_' + str(args.L)
        dataFileName += '_nH' + str(args.nH)
        dataFileName += '_lr' + lr
        #dataFileName += '_lr0.100'
        dataFileName += '_w1.0'
        dataFileName += '_MCS' + str(args.MCS)
        dataFileName += '_h' + h[i]
        dataFileName += '_raw.txt'
        
        #print dataFileName
     
        dataFile = open(dataFileName,'r')
        data = np.loadtxt(dataFile)
        n_meas = len(data) - args.cut
        avg = get_average(data,args.cut)
        err = np.std(data,axis=0)/m.sqrt(n_meas)
        delta = abs(avg-dmrg[i,1])/abs(dmrg[i,1])

        #print('\nAverage Energy: %f  +-  %f\n' % (avg,err))
        outputFile.write('%s   ' % h[i])
        
        outputFile.write('%.6f   ' % avg)
        #outputFile.write('%f  ' % err)
            
        outputFile.write('%.6f   ' % dmrg[i,1])
        
        outputFile.write('%.6f   ' % delta)
        outputFile.write('\n') 
 



