import numpy as np
from stella.evaluation.snr import SNR
from stella.utils.DataReader import *
from stella.utils.Accumulator import *

def write_snr(TRACES_PREFIX,LABELS_PREFIX,FILE_SNR,
                n_files,
                labels,batch_size=-1,Nc=256,verbose=False):
    labels = np.array_split(labels,len(labels)//batch_size) if batch_size != -1 else np.array([labels])
    
    snrs_labels = []
    for l in labels:
        # prepare and start the DataReader
        file_read = [[(TRACES_PREFIX+"_%d.npy"%(i),None),(LABELS_PREFIX+"_%d.npz"%(i),["labels"])]   for i in range(n_files)]
        reader = DataReader(file_read,max_depth=2)
        reader.start()

        i = 0
        data = reader.queue.get()
        while data is not None:
            # load the file
            traces = data[0]
            labels_f = data[1][0]

            labels_f = list(filter(lambda x:x["label"] in l,labels_f))
            assert len(labels_f) == len(l)

            if i == 0:
                labels_0 = list(map(lambda x: x["label"],labels_f))
                classes = np.zeros((len(l),len(traces[:,0])),dtype=np.uint16)
                snr = SNR(Nc,len(traces[0,:]),Np=len(labels_f))
            else: #assert current file structured as the first one
                for la_0,la_f in zip(labels_0,labels_f): assert la_0 == la_f["label"]

            for j,la in enumerate(labels_f):
                classes[j,:] = la["val"]

            snr.fit_u(traces,classes)

            data = reader.queue.get()
            i += 1
        snrs_labels += [{"label":n["label"],"snr":snr._SNR[i,:]} for i,n in enumerate(labels_f)]

    for x in snrs_labels:
        print(x["label"] , "%.4f"%(np.max(x["snr"])))
    np.savez(FILE_SNR,snr=snrs_labels,allow_pickle=True)

def write_poi(FILE_SNR,FILE_POI,
                labels,
                selection_function):

    snrs_labels = np.load(FILE_SNR,allow_pickle=True)["snr"]
    pois_labels = list(map(lambda x: {"poi":selection_function(x["snr"]),"label":x["label"]},snrs_labels))
    np.savez(FILE_POI,poi=pois_labels,allow_pickle=True)

def build_model(TRACES_PREFIX,LABELS_PREFIX,FILE_POI,FILE_MODEL,
                n_files,
                labels,
                func,batch_size=-1):

    pois = np.load(FILE_POI,allow_pickle=True)["poi"]
    pois_l = list(map(lambda x:x["label"],pois))

    labels = np.array_split(labels,len(labels)//batch_size) if batch_size != -1 else np.array([labels])

    models = []
    for l in labels:
        # prepare and start the DataReader
        file_read = [[(TRACES_PREFIX+"_%d.npy"%(i),None),(LABELS_PREFIX+"_%d.npz"%(i),["labels"])]   for i in range(n_files)]
        reader = DataReader(file_read,max_depth=2)
        reader.start()

        i = 0
        data = reader.queue.get()
        while data is not None:

            # load the file
            traces = data[0]
            labels_f = data[1][0]

            labels_f = list(filter(lambda x:x["label"] in l,labels_f))
            assert len(labels_f) == len(l)

            if i == 0:
                models_l = labels_f
                for m_f in models_l:
                    index = pois_l.index(m_f["label"])
                    m_f["poi"] = pois[index]["poi"]
                    m_f["acc_val"] = Accumulator(len(traces[:,0])*n_files,1,dtype=np.uint16)
                    m_f["acc_traces"] = Accumulator(len(traces[:,0])*n_files,len(m_f["poi"]),dtype=np.int16)
            else: # check that all the files as structed in the same way as the first one
                for m,la_f in zip(models_l,labels_f): assert m["label"] == la_f["label"]

            for m,la_f in zip(models_l,labels_f): m["acc_val"].fit(la_f["val"].astype(np.uint16))
            for m in models_l: m["acc_traces"].fit(traces[:,m["poi"]].astype(np.uint16))

            i+= 1
            data = reader.queue.get()

        for m in models_l:
            t = m.pop("acc_traces").get()
            l = m.pop("acc_val").get()[:,0]
            m["model"] = func(t,l,m["label"])
            del t,l

        models += models_l
    
    np.savez(FILE_MODEL,model=models,allow_pickle=True)

