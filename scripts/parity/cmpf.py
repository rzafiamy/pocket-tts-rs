import sys, numpy as np, scipy.io.wavfile as w
def load(p):
    _, x = w.read(p)
    return x.astype(np.float64) / (32768 if x.dtype.kind == 'i' else 1)
a=load(sys.argv[1]); b=load(sys.argv[2])
n=min(len(a),len(b)); F=1920
print(" ".join("%d:%.3f" % (i, np.abs(a[i*F:(i+1)*F]-b[i*F:(i+1)*F]).max()) for i in range(n//F)))
