import numpy as np
import stella.lib.rust_stella as rust
from tqdm import tqdm
class SNR:
    def __init__(self,Nc,Ns,Np=1,use_rust=True):
        """
            This function computes the Signal-to-Noise ratio between the traces
            and the intermediate values. It is ment to work on traces being 
            int16.

            Nc: Possible values for the intermediate values X
            Ns: Number of samples in a single traces
            Np: Number of intermediates variable to comptue the SNR on. Default
            to 1
        """
        if Nc >= (2**16):
            raise Exception("SNR can be computed on max 16 bit, {} given".format(Nc))
        self._Nc = Nc
        self._Ns = Ns
        self._Np = Np

        if use_rust:
            # Number of observed traces with given intermediate variable
            self._ns = np.zeros((Np,Nc),dtype=np.uint32)
            # Sum for each class
            self._sum = np.zeros((Np,Nc,Ns),dtype=np.int64)
            # Sum of squared traces for each class
            self._sum2 = np.zeros((Np,Nc,Ns),dtype=np.int64)
            # Mean of each class
            self._means = np.zeros((Np,Nc,Ns),dtype=np.float32)
            # Variance in each class
            self._vars= np.zeros((Np,Nc,Ns),dtype=np.float32)
            # SNR on each class
            self._SNR = np.zeros((Np,Ns),dtype=np.float32)
        else:
            # Number of observed traces with given intermediate variable
            self._ns = np.zeros((Np,Nc),dtype=np.uint32)
            # Sum for each class
            self._sum = np.zeros((Np,Nc,Ns),dtype=np.float64)
            # Sum of squared traces for each class
            self._sum2 = np.zeros((Np,Nc,Ns),dtype=np.float64)
            # Mean of each class
            self._means = np.zeros((Np,Nc,Ns),dtype=np.float64)
            # Variance in each class
            self._vars= np.zeros((Np,Nc,Ns),dtype=np.float64)
            # SNR on each class
            self._SNR = np.zeros((Np,Ns),dtype=np.float32)


        self._means[:,:] = np.nan

        self._i = 0
    def __del__(self):
        """
            Delet all the numpy array in the object
        """
        del self._ns,self._sum,self._sum2,self._means
        del self._SNR,self._vars

    def fit_u(self,traces,X,use_rust=True,nchunks=1):
        """
            Updates the SNR status to take the fresh samples into account

            traces: (?,Ns) int16 or int8 array containing the array.
            X: (Np,?) uint16 array coutaining
            use_rust: use low level rust
            nchunks: in how many chunks to // the snr
        """
        X = (X%self._Nc).astype(np.uint16)
        if self._Np == 1 and X.ndim == 1:
            X = X.reshape((1,len(X)))

        if use_rust:
            if not (traces.dtype == np.int16):
                raise Exception("Trace type not supported {}".format(Trace.dtype))
            elif len(X) != self._Np:
                raise Exception("Input X array does not match: Expected {} given {}".format((self._Np,len(traces)),X.shape))
 
            rust.update_snr(traces,X,self._sum,self._sum2,self._ns,self._means,self._vars,self._SNR,nchunks)
        else:
            n = len(traces[:,0])
            for v in range(self._Np):
                for i in tqdm(range(n),desc="SNR"):
                    self._ns[v,X[v,i]] += 1
                    t = traces[i,:]
                    self._sum[v,X[v,i],:] += t
                    self._sum2[v,X[v,i],:] += t*t.astype(np.float32)

                for c in range(self._Nc):
                    self._means[v,c,:] = (self._sum[v,c,:].T / self._ns[v,c]).T
                    self._vars[v,c,:] = (self._sum2[v,c,:].T/self._ns[v,c]).T - (self._means[v,c,:]**2)

            for v in range(self._Np):
                self._SNR[v,:] = np.var(self._means[v,:],axis=0)/np.mean(self._vars[v,:],axis=0)
        return self._SNR


class SNROrder:
    def __init__(self,Nc,Ns,Np=1,D=1):
        """
            This function computes the Signal-to-Noise ratio between the traces
            and the intermediate values. It is ment to work on traces being 
            int16.

            Nc: Possible values for the intermediate values X
            Ns: Number of samples in a single traces
            Np: Number of intermediates variable to comptue the SNR on. Default
            to 1
        """
        if Nc >= (2**16):
            raise Exception("SNR can be computed on max 16 bit, {} given".format(Nc))
        self._Nc = Nc
        self._Ns = Ns
        self._Np = Np
        self._D = D

        # Number of observed traces with given intermediate variable
        self._ns = np.zeros((Np,Nc),dtype=np.float64)
        # First order moment of each class
        self._M = np.zeros((Np,Nc,Ns),dtype=np.float64)
        # Centered moment of each class
        self._CS = np.zeros((Np,Nc,2*D,Ns),dtype=np.float64)
        # SNR on each class up to order D
        self._SNR = np.zeros((Np,D,Ns),dtype=np.float32)

    def fit_u(self,traces,X,use_rust=True,nchunks=1):
        """
            Updates the SNR status to take the fresh samples into account

            traces: (?,Ns) int16 or int8 array containing the array.
            X: (Np,?) uint16 array coutaining
            use_rust: use low level rust
            nchunks: in how many chunks to // the snr
        """
        X = (X%self._Nc).astype(np.uint16)
        if self._Np == 1 and X.ndim == 1:
            X = X.reshape((1,len(X)))

        if not (traces.dtype == np.int16):
            raise Exception("Trace type not supported {}".format(Trace.dtype))
        elif len(X) != self._Np:
            raise Exception("Input X array does not match: Expected {} given {}".format((self._Np,len(traces)),X.shape))
 
        rust.update_snrorder(traces,
                X,
                self._ns,
                self._CS,
                self._M,
                self._D,
                nchunks)

        for i in range(self._Np): # for each class
            
            CM = (self._CS[i].T/self._ns[i]).T # (Nc,D*2,Ns)
            for d in range(1,self._D+1):
                if d == 1:
                    u = self._M[i,:] #(Nc,Ns)
                    v = CM[:,1,:]
                elif d==2:
                    u = CM[:,1,:]
                    v = CM[:,3,:] - CM[:,1,:]**2
                else:
                    u = CM[:,d-1,:]/np.power(CM[:,1,:],d/2);
                    v = (CM[:,(d*2)-1,:] - CM[:,(d)-1]**2)/(CM[:,1,:]**d)
                self._SNR[i,d-1,:] = np.var(u,axis=0)/np.mean(v,axis=0)

    def get_sm(self,D):
        """
            returns the standardized moments to peform MCP-DPA

            returns:
            SM: standardized moments of order D
            s: standard deviation
            u: means

        """
        SM = np.zeros((self._Np,self._Nc,self._Ns))
        CM_all = np.zeros((self._Np,self._Nc,self._Ns))
        s = np.zeros((self._Np,self._Nc,self._Ns))
        u = self._M.copy()

        for i in range(self._Np): # for each class
            CM = (self._CS[i].T/self._ns[i]).T # (Nc,D*2,Ns)
            if D == 1:
                m = self._M[i,:] #(Nc,Ns)
            elif D==2:
                m = CM[:,1,:]
            else:
                m = CM[:,D-1,:]/np.power(CM[:,1,:],D/2);

            s[i,:] = np.sqrt(CM[:,1,:])
            SM[i,:,:] = m
            CM_all[i,:,:] = CM[:,D-1,:]
        return SM,u,s,CM_all


if __name__ == "__main__":
    Ns = 10
    Np = 1
    D = 2
    D_SNR = 6
    Nc = 4
    nt = 1000000
    v = np.random.randint(0,Nc,(nt,D),dtype=np.uint8)

    r = 0
    for d in range(D):
        r = np.bitwise_xor(v[:,d],r)

    X = r.reshape((1,nt))
    leakage = np.sum(np.random.normal(loc=v,scale=1),axis=1)
    leakage -= np.mean(leakage)
    leakage /= np.std(leakage)
    leakage *= 2**12
    leakage = leakage.astype(np.int16) + 2343
    tr = np.random.randint(0,2**12,(nt,Ns)).astype(np.int16)
    tr[:,1] = leakage 
    snr_o = SNROrder(Nc,Ns,Np,D_SNR)
    snr_o.fit_u(tr,X)

    for d in range(1,D_SNR+1):
        print("SNR at order %d"%(d))
        print(snr_o._SNR[0,d-1,:])
