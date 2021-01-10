import numpy as np
from matplotlib import pyplot as plt
import derivatives 
import sys

TEST = 1;

edgeeffect = 0.1;

'''
% we are given one Gaussian
% faceDetector.MinSize =  [500 500];
% faceDetector.MaxSize =  [800 800];
% faceDetector.ScaleFactor = 100;
% faceDetector.MergeThreshold = 10;
'''
##########
# left image
##########
I = np.zeros((2000, 2000, 1))#imread('leftstart.png');
bbox = np.array([1000, 1100, 300, 400]) 

M, N, P = I.shape;

lfimcentx = N/2;
lfimcenty = M/2;

xlfimo = bbox[0] + 0.5*bbox[2];
ylfimo = bbox[1] + bbox[3];

#plt.plot(xlfimo,ylfimo,'r.')
#plt.show()

K1 = 1/300;
K2 = (K1*100)/N;
# % how much of sigma is the bbox dimension
K3 = 2;

xlfim = bbox[0] + 0.5*bbox[2] - lfimcentx;
ylfim = bbox[1] + bbox[3] - lfimcenty;
sigmaim = K3*max(bbox[2],bbox[3]);

d = 0; 
c = 0.5 - (K2*N)/2;

xlf = c - K2*xlfim
ylf = d - K2*ylfim
sigmal = sigmaim*K2

x = np.zeros((1,2))
y = np.zeros((1,2))

x[:,1] = xlf + edgeeffect;
y[:,1] = ylf;


##########
# right image
##########

bbox = np.array([1000, 1100, 300, 400]) 
# bbox = np.array([300, 400, 300, 400])

xlfimo = bbox[0] + 0.5*bbox[2];
ylfimo = bbox[1] + bbox[3];

xlfim = bbox[0] + 0.5*bbox[2] - lfimcentx;
ylfim = bbox[1] + bbox[3] - lfimcenty;
sigmaim = K3*max(bbox[2],bbox[3]);


b = 0; 
a = -0.5 + (K2*N)/2;

xrf = a - K2*xlfim
yrf = b - K2*ylfim
sigmar = sigmaim*K2

x[:,0] = xrf -edgeeffect ;
y[:,0] = yrf;

#############
# final right image
##############

bbox = np.array([100, 200, 300, 400]) 
# bbox = np.array([1000, 1100, 300, 400]) 

xlfimo = bbox[0] + 0.5*bbox[2];
ylfimo = bbox[1] + bbox[3];

xlfim = bbox[0] + 0.5*bbox[2] - lfimcentx;
ylfim = bbox[1] + bbox[3] - lfimcenty;
sigmaim = K3*max(bbox[2],bbox[3]);

b = 0; 
a = -0.5 + (K2*N)/2;

xrf = a - K2*xlfim
yrf = b - K2*ylfim
sigmar = sigmaim*K2

K4a = 1/np.abs(x[:,0]);
K4b = 1/np.abs(x[:,1]);

x[:,0] = x[:,0]*K4a
x[:,1] = x[:,1]*K4b

y[:,0] = y[:,0]*K4a
y[:,1] = y[:,1]*K4b


####################
# go through different random pairs of gaussians
#################### 
    
# if  TEST    
# % first gaussian parameters
# %     xlf = 0.3;   
# %     ylf = 0;

# variance in both directions
sigma1l = sigmal;
sigma2l = sigmal;

#% how elliptical it is 
rhol = 0;

# % second gaussian parameters
# %     xrf = -0.3
# %     yrf = -0.3;
     
sigma1r = sigmar;
sigma2r = sigmar;

#% how elliptical it is 
rhor = 0;

# plane pi on which we mark the two gaussians
pi_plane = np.zeros((100,100))

# Remember the whole struct, which is given to us
gaussstruct = {'xlf':xlf, 'ylf':ylf, 'xrf':xrf, 
                'yrf':yrf, 'sigma1l':sigma1l, 'sigma2l':sigma2l, 
                'sigma1r':sigma1r, 'sigma2r':sigma2r, 'rhol':rhol, 'rhor':rhor}



###############
# move between 0 and 1
###############
X, Y = np.meshgrid(np.arange(100), np.arange(100))
Y = np.flipud(Y)
mX = X.mean()
mY  = Y.mean()
mxX = X.max()
mxY  = Y.max()
X = X - mX
Y = Y - mY
X = X / mxX
Y = Y / mxY

############
# Calculate the gaussians in the image plane
############

C1p = 1/sigma1l**2;
C2p = 2 * rhol / (sigma1l * sigma2l)
C3p = 1 / sigma2l**2
C4 = 1 / (2 * (1 - rhol**2))

C1 = C1p / C4
C2 = C2p / C4
C3 = C3p / C4

C5 = 1/(2 * np.pi * sigma1l * sigma2l * np.sqrt(1 - rhol**2))


fgauss = C5*np.exp(-1*(C1*(X - xlf)**2 - C2*(X - xlf)*(Y - ylf) + C3*(Y - ylf)**2))


C1pr = 1/sigma1r**2;
C2pr = 2*rhor/(sigma1r*sigma2r)
C3pr = 1/sigma2r**2
C4r = 1/(2*(1-rhor**2))
C1r = C1pr / C4r
C2r = C2pr / C4r
C3r = C3pr / C4r
C5r = 1/(2*np.pi*sigma1r*sigma2r*np.sqrt(1 - rhor**2))


fgauss2 = C5r*np.exp(-1*(C1r*(X - xrf)**2 - C2r*(X - xrf)*(Y - yrf) + C3r*(Y - yrf)**2))
fgaussd = fgauss/np.max(fgauss)
fgauss2d = fgauss2/np.max(fgauss2)

f = fgauss + fgauss2
fd = fgaussd + fgauss2d


fmask1 = fgauss
fmask1[fmask1 > 0.1] = 1
fmask1[fmask1 != 1] = 0

fmask2 = fgauss2
fmask2[fmask2 > 0.1] = 1
fmask2[fmask2 != 1] = 0

   
# no need for ground truth, we just go from the original scan to the new one


#plt.figure(2)
#plt.imshow(fd)
#plt.show()

# % [x y] = ginput(2)

# hold
# % plot(x,y,'r-');
# % 
# %     y = 100 - y;
# % 
# %     x = (x - mX)./mxX;
# %     y = (y - mY)./mxY;
# % 
# % x = round(x);
# % y = round(y);


xrp = x[:,0]*mxX + mX
yrp = y[:,0]*mxY + mY
yrp = 100 - yrp

xop = x[:,1]*mxX + mX
yop = y[:,1]*mxY + mY
yop = 100 - yop


# plt.plot(np.array([xrp, xop]),np.array([yrp, yop]), '^');
# plt.plot(xrp,yrp,'*');
# plt.show()
   
thetas = np.arange(180)

# noise from the robot sensing
ncount = 1
noiseexp = 0
expcount = 1


# % We only accept an initialization which intersects the line
# % segment joining the maxima

# % we set this to the centers of the two frames using the PPT
# % scaling
# % we can set alpha to zero or pi/2 depending on the origin 


x1t = x[:,0]
y1t = y[:,0]
x2t = x[:,1]
y2t = y[:,1]
xr = x[:,0]
yr = y[:,0]
alpha = np.arctan((y2t - y1t)/(x2t - x1t))
    
L = np.sqrt((x[:,1] - x[:,0])**2 + (y[:,1] - y[:,0])**2)

E = np.exp(1)

Pi = np.pi


# calc derivs
struct = {'C1':C1, 'C2':C2, 'C3':C3, 'C4':C4, 'C5':C5, 
'xlf':xlf, 'ylf':ylf, 'E':E, 'xr':xr, 'yr':yr, 'alpha':alpha, 
'L':L, 'C1r':C1r, 'C2r':C2r, 'C3r':C3r, 'C4r':C4r, 'C5r':C5r, 
'xrf':xrf, 'yrf':yrf}
derivs = derivatives.calculate(struct)

#optimization params    
smallval = 0.1
lr = 0.13
oldnorm = np.linalg.norm(derivs) + 0.01

q_r = np.array([xr, yr, alpha])
Br = np.zeros((3,2))
u_r = np.zeros((2,1))
dt = 0.1

optcount = 1
MAXCOUNT = 500

#clear qrs;
#qrs(optcount,:) = q_r(:);
optcount = optcount + 1;

# Optimization running

while (np.linalg.norm(derivs) > smallval) and (optcount < MAXCOUNT):

    oldnorm = np.linalg.norm(derivs);
    oldqr = q_r;

    Br[0,0] = dt*np.cos(alpha);
    Br[1,0] = dt*np.sin(alpha);
    Br[2,1] = dt;

    u_r += lr*dt*(derivs.transpose()@Br).transpose()
    q_r += Br@u_r + dt*lr*derivs

    optcount += 1;
   
    # new mems location
    xr = q_r[0]
    yr = q_r[1]
    alpha = q_r[2]

    # new Pi location
    xo = xr + L*np.cos(alpha);
    yo = yr + L*np.sin(alpha);

    xrp = xr*mxX + mX;
    yrp = yr*mxY + mY;
    yrp = 100 - yrp;

    xop = xo*mxX + mX;
    yop = yo*mxY + mY;
    yop = 100 - yop;
    
    
    # plt.plot(xrp,yrp,'g-o')
    # plt.plot(xop, yop, 'r-*')
    # plt.show()
    # compute derivatives
    struct['xr'], struct['yr'], struct['alpha'] = xr, yr, alpha
    derivs = derivatives.calculate(struct)

    print(np.linalg.norm(derivs))
        

