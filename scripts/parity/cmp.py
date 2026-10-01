import sys, numpy as np, scipy.io.wavfile as w
_, a = w.read(sys.argv[1]); _, b = w.read(sys.argv[2])
a = a.astype(np.float64); b = b.astype(np.float64)
if a.dtype.kind == 'i' or abs(a).max() > 2: a /= 32768
if abs(b).max() > 2: b /= 32768
print("len", len(a), len(b))
n = min(len(a), len(b))
d = a[:n]-b[:n]
print("max|diff| %.5f  rms diff %.5f  rms ref %.5f  corr %.5f" % (abs(d).max(), np.sqrt((d**2).mean()), np.sqrt((a[:n]**2).mean()), np.corrcoef(a[:n], b[:n])[0,1]))
