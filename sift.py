import numpy as np
from scipy.signal import fftconvolve as conv2
from scipy.ndimage import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt

def gauss(sigma,num=None):
    if num is None:
        num=int(4*sigma+1)
        num += num%2+1#make odd
    x=np.arange(0,num)-(num-1)//2
    g = np.exp(-x**2/(2*sigma**2))
    G = np.einsum("x,y->xy",g,g)
    G /= np.sum(G)
    return G

def rot_mat(theta):
    R = np.array([[np.cos(theta),-np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R

def scale_space(im,s=3,sigma_0=1.6,sigma_n=0.5):
    G_sigma = gauss((sigma_0**2-sigma_n**2)**(1/2))
    L = []
    im = conv2(im,G_sigma,mode="same")
    num_octaves = int(np.floor(np.log2(min(im.shape))-4))# this is not opencv standard, but i dont get how they can make descriptors in octaves where the spatial dimensions are smaller than the descriptor???
    if num_octaves < 1:
        raise ValueError("Image is too small!")
    k = 2**(1/s)
    kernels = [None]+[gauss((k**2-1)**(1/2)*(k**scale)*sigma_0) for scale in range(1,s+3)]
    for octave in range(0,num_octaves):
        octave = np.zeros((s+3,)+im.shape)
        octave[0] = im
        for scale in range(1,s+3):
            octave[scale] = conv2(octave[scale-1],kernels[scale],mode="same")
        im = octave[s][::2,::2]#TODO: should it really be just nearest neighbour? most higher frequencies are gone but still
        L.append(octave)
    return L

def eliminate_edges(pts,D,r=10):
    H_11 = np.array([[0,0,0],
                    [1,-2,1],
                    [0,0,0]])
    H_12 = np.array([[1/2,0,-1/2],
                    [0,0,0],
                    [-1/2,0,1/2]])
    H_22 = H_11.T
    non_edge_pts = []
    T = (r+1)**2/r
    for o, octave_pts in enumerate(pts):
        octave_non_edge_pts = []
        for (s,y,x) in octave_pts:
            region = D[o][s,y-1:y+2,x-1:x+2]
            HD_11 = np.sum(H_11*region)
            HD_12 = np.sum(H_12*region)
            HD_22 = np.sum(H_22*region)
            det = HD_11*HD_22-HD_12**2
            tr = HD_11+HD_22
            edgeness = tr**2/det
            if edgeness < T:
                octave_non_edge_pts.append((s,y,x))
        non_edge_pts.append(octave_non_edge_pts)
    return non_edge_pts

def determine_orientation(pts,L,k=2**(1/3),sigma_0=1.6):
    dx = np.array([[1,0,-1]])
    dy = dx.T
    oriented_pts = []
    bins = np.linspace(0,2*np.pi,num=37)
    bin_centers = np.convolve(bins,np.array([1,1]),mode="valid")/2
    ind_to_center = {ind:ctr for ind,ctr in enumerate(bin_centers)}
    for o, octave_pts in enumerate(pts):
        octave_oriented_pts = []
        for (s,y,x) in octave_pts:
            sigma = 1.5*k**s*sigma_0
            width=int(2*sigma)+1
            G = gauss(sigma,2*width+1)
            region = L[o][s,y-width:y+width+1,x-width:x+width+1]
            if region.shape != G.shape:
                continue#Region is not wide enough to fit
            rx = conv2(region,dx,mode="same")
            ry = conv2(region,dy,mode="same")
            m = np.sqrt(rx**2+ry**2)*G
            theta = np.arctan2(ry,rx)+np.pi
            hist,_ = np.histogram(theta,bins=bins,weights=m)
            main_peak = np.argmax(hist)
            #TODO: here quadratic interpolation should be used
            octave_oriented_pts.append((ind_to_center[main_peak],s,y,x))
            # Search for second peak if it exists
            main_val = hist[main_peak]
            peaks = hist == maximum_filter(hist,size=5,mode="wrap")#this should wrap since the histogram is 2pi periodic
            other_peaks = peaks*(main_val>hist)*(hist>0.8*main_val)#avoid the actual main peak, but over 0.8*main_val
            if np.any(other_peaks):
                second_val=np.max(hist[other_peaks])
                second_peak = np.argwhere(np.logical_and(hist==second_val, other_peaks))[0,0]# The "and" here is to make sure that we dont end up selecting a non-peak
                octave_oriented_pts.append((ind_to_center[second_peak],s,y,x))
        oriented_pts.append(octave_oriented_pts)
    return oriented_pts

def trilinear_interpolation(theta,y,x,half_hist_width,m):
    descriptor = np.zeros((4,4,8))
    y_bins,x_bins,theta_bins = np.linspace(-half_hist_width,half_hist_width,num=4),np.linspace(-half_hist_width,half_hist_width,num=4),np.linspace(0,2*np.pi,num=9)
    coord_interval = half_hist_width/1.5
    theta_interval = np.pi/4
    y_dist = ((y/coord_interval)%1).flatten()
    x_dist = ((x/coord_interval)%1).flatten()
    theta_dist = ((theta/theta_interval)%1).flatten()

    for y_box in range(2):
        for x_box in range(2):
            for theta_box in range(2):
                values = 1-(y_box*y_dist+(1-y_box)*(1-y_dist))*\
                          (x_box*x_dist+(1-x_box)*(1-x_dist))*\
                          (theta_box*theta_dist+(1-theta_box)*(1-theta_dist))
                values *= m.flatten()
                hist,bins = np.histogramdd(np.stack((y.flatten()+coord_interval*y_box,x.flatten()+coord_interval*x_box,(theta.flatten()+theta_dist*theta_box)%(2*np.pi))).T, bins = (y_bins+coord_interval*y_box,x_bins+coord_interval*x_box,theta_bins),weights=values)
                descriptor[y_box:3+y_box,x_box:3+x_box,:] = hist
    return descriptor.flatten()

def normalize_descriptor(descriptor,max_val=0.2):
    descriptor /= np.sqrt(np.sum(descriptor**2))
    descriptor = np.maximum(descriptor,0.2)
    descriptor /= np.sqrt(np.sum(descriptor**2))
    return descriptor

def calculate_descriptors(pts,L,k=2**(1/3),sigma_0=1.6):
    dx = np.array([[1,0,-1]])
    dy = dx.T
    oriented_pts = []
    bins = np.linspace(0,2*np.pi,num=37)
    bin_centers = np.convolve(bins,np.array([1,1]),mode="valid")/2
    ind_to_center = {ind:ctr for ind,ctr in enumerate(bin_centers)}
    #In the actual implementation, this integral is computed by visiting a rectangular area of the image that fully contains the keypoint grid (along with half a bin border to fully include the bin windowing function). Since the descriptor can be rotated, this area is a rectangle of sides \(m/2\sqrt{2} (N_x+1,N_y+1)\) (see also the illustration).
    # https://www.vlfeat.org/api/sift.html
    # as I understand it, the image frame should be large enough to cover the canonical frame
    # so i will use a width of TODO 
    descriptors = []
    for o, octave_pts in enumerate(pts):
        for (orientation,s,y,x) in octave_pts:
            half_width = int(8*k**s)#TODO: parameter
            half_hist_width = (half_width/np.sqrt(2))
            G = gauss(sigma=half_width,num=2*half_width+1)#TODO: parameter
            region = L[o][s,y-half_width:y+half_width+1,x-half_width:x+half_width+1]
            if region.shape != G.shape:
                continue #Region is not wide enough to fit
            rx = conv2(region,dx,mode="same")
            ry = conv2(region,dy,mode="same")
            m = np.sqrt(rx**2+ry**2)*G
            theta = np.arctan2(ry,rx)+np.pi
            grid = np.stack(np.mgrid[-half_width:half_width+1,-half_width:half_width+1])

            R = rot_mat(orientation)
            rot_grid = np.einsum("yx,xmn->ymn",R.T,grid)
            rot_theta = (theta-orientation) % (2*np.pi)

            descriptor = trilinear_interpolation(rot_theta,rot_grid[1],rot_grid[0],half_hist_width,m)
            descriptor = normalize_descriptor(descriptor)
            descriptors.append(descriptor)
    return descriptors

def SIFT(im):
    r = 10#from paper, ration between curvature
    L = scale_space(im)
    D = [np.diff(octave,axis=0) for octave in L]
    pts = []
    for D_o in D:
        M_o=maximum_filter(np.abs(D_o),size=3,mode="constant")
        max_inds = (M_o==D_o)&(np.abs(D_o)>0.03)#This theshold depends on the values in the image TODO: should be a parameter
        pts.append(list(zip(*np.nonzero(max_inds))))
    pts = eliminate_edges(pts,D)
    pts = determine_orientation(pts,L)
    descriptors = calculate_descriptors(pts,L)

    #TODO: Here a quadratic model should be fitted, but skipping it for now
    print("done")

im = Image.open("cameraman.jpg")
im = np.mean(np.asarray(im),axis=2)/255
SIFT(im)
