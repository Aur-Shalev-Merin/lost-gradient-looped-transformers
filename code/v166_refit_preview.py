import numpy as np, math
from scipy.optimize import curve_fit
F=[(0.8051,0.183,1),(0.6477,0.077,2),(0.3860,0.012,4),(0.8132,0.209,1),(0.6632,0.090,2),(0.4210,0.016,4),(0.1803,-0.005,8),
(0.8201,0.224,1),(0.6753,0.100,2),(0.4427,0.024,4),(0.2046,0.002,8),(0.8773,0.212,1),(0.7911,0.142,2),(0.6453,0.072,4),(0.4183,0.020,8),
(0.744,0.127,2),(0.646,0.076,3),(0.558,0.046,4),(0.423,0.015,6),(0.860,0.169,2),(0.809,0.137,3),(0.763,0.111,4),(0.679,0.073,6),
(0.5462,0.0479,3),(0.3071,0.0046,6)]
e=np.array([x[0] for x in F]);y=np.array([x[1] for x in F]);K=[x[2] for x in F]
f1=lambda x,c,p:c*x**p; f2=lambda x,c,p,d:c*x**p/(1+d*x**p)
p1,_=curve_fit(f1,e,y,p0=(.3,3),bounds=([0,.5],[10,12])); p2,_=curve_fit(f2,e,y,p0=(.3,3,0),bounds=([0,.5,0],[50,12,50]))
n=len(y); aic=lambda r,k:n*math.log(np.sum(r**2)/n)+2*k
a1=aic(y-f1(e,*p1),2);a2=aic(y-f2(e,*p2),3); print('pow',p1,a1,'| sat',p2,a2)
M=(lambda x:f1(x,*p1)) if a1<=a2 else (lambda x:f2(x,*p2))
byk={k:np.mean([y[i] for i in range(n) if K[i]==k]) for k in sorted(set(K))}; print('M_k',{k:round(v,4) for k,v in byk.items()})
ks=sorted(byk);L=lambda k:math.log2(k)
def g(k):
  if k in byk: return byk[k]
  if k>ks[-1]:
    a,b=ks[-2],ks[-1]; return max(0,byk[b]+(byk[b]-byk[a])/(L(b)-L(a))*(L(k)-L(b)))
  lo=max(x for x in ks if x<k);hi=min(x for x in ks if x>k);w=(L(k)-L(lo))/(L(hi)-L(lo));return (1-w)*byk[lo]+w*byk[hi]
C=[('A R64 a.3',8,.6022),('A R64 a.3',12,.4772),('A R64 a.3',16,.3886),('A R48 a.4',4,.6992),('A R48 a.4',8,.502),('A R48 a.4',12,.3808),('A R48 a.4',16,.279),
('B 370m a.5',2,.7894),('B 370m a.5',4,.6427),('B 370m a.5',8,.4171),('B 370m a1',2,.6739),('B 370m a1',4,.4513)]
for nm,k,x in C:
  me,mk=M(x),g(k); print('%-11s k%-2d err %.3f  M_err %.4f  M_k %.4f  diff %+.4f %s'%(nm,k,x,me,mk,me-mk,'SCORE' if abs(me-mk)>=.04 else ''))
