import numpy as np
from scipy.signal import fftconvolve as conv2
from scipy.ndimage import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt

def gauss(sigma,num=None):
    """
    Constructs a 2d-gaussian filter with a given standard deviation

    Parameters:
        sigma: The standard deviation of the gaussian
        num (optional): Number of coefficients in the filter, will otherwise be automatically determined
    """

    if num is None:
        num=int(4*sigma+1)
        num += num%2+1#make odd
    x=np.arange(0,num)-(num-1)//2
    g = np.exp(-x**2/(2*sigma**2))
    G = np.einsum("x,y->xy",g,g)
    G /= np.sum(G)
    return G

def rot_mat(theta):
    """
    Constructs a simple 2-d rotation matrix from a given angle
    
    Parameters:
        theta: The rotation angle
    Returns:
        R: The rotation matrix
    """
    R = np.array([[np.cos(theta),-np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R

def scale_space(im,s=3,sigma_0=1.6,sigma_n=0.5):
    """
    Constructs a scale space for an image
    
    Parameters:
        im: The image
        s: The number of scales per octave (the number of octaves is automatically set)
        sigma_0: The initial blurring in the first scale
        sigma_n: Presumed blurring of the original image (typically some anti-aliasing has been done beforehand)
    Returns:
        L: The scale space, a list of all the constructed octaves where each octave is a 3d numpy array (scale,width,height) 
    """
    #TODO: Should add option to first upsample the image
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
        im = octave[s,::2,::2]#TODO: should it really be just nearest neighbour? most higher frequencies are gone but still
        L.append(octave)
    return L

def taylor_expand(pts,D):
    """
    TODO
    """
    return pts # TODO: finish this

def eliminate_edges(pts,D,r=10):
    """
    Removes all points where the curvature ratio is too sharp, i.e. eliminates edges

    Parameters:
        pts: The points
        D: The DoG scale space
        r (optional): The maximum allowed ratio
    """
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
            if det < 0:
                continue #TODO: This works kind of poorly when not using a taylor expansion of the DoG scale space (a lot of points get rejected)
            if edgeness < T:
                octave_non_edge_pts.append((s,y,x))
        non_edge_pts.append(octave_non_edge_pts)
    return non_edge_pts

def quadratic_fit(Q,hist,peak,ind_to_center):
    """
    """
    h = hist.take(range(peak-1,peak+2),mode="wrap")#indexing needs to wrap since histogram is periodic
    (a,b,c) = np.linalg.solve(Q,h)
    theta_offset = -b/(2*a)
    theta_diff = ind_to_center[1]-ind_to_center[0]
    # theta_offset should always be between [-0.5,0.5] since peak is a peak
    theta = ind_to_center[peak]+theta_diff*theta_offset
    return theta

def determine_orientation(pts,L,k=2**(1/3),sigma_0=1.6):
    """
    Deterimines the main orientation of the points of interest.

    Parameters:
        pts: The points of interest
        L: The scale space
        k: The scaling factor
        sigma_0: The scale of the most detailed level
    Returns:
        pts: The points, now with orientation attached
    """
    dx = np.array([[1,0,-1]])
    dy = dx.T
    oriented_pts = []
    bins = np.linspace(0,2*np.pi,num=37)
    bin_centers = np.convolve(bins,np.array([1,1]),mode="valid")/2
    ind_to_center = {ind:ctr for ind,ctr in enumerate(bin_centers)}
    Q = np.array([[1,-1,1],[0,0,1],[1,1,1]])#Matrix for quadratic interpolation
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
            theta = quadratic_fit(Q,hist,main_peak,ind_to_center)
            octave_oriented_pts.append((theta,s,y,x))
            # Search for second peak if it exists
            main_val = hist[main_peak]
            peaks = hist == maximum_filter(hist,size=5,mode="wrap")#this should wrap since the histogram is 2pi periodic
            other_peaks = peaks*(main_val>hist)*(hist>0.8*main_val)#avoid the actual main peak, but over 0.8*main_val
            if np.any(other_peaks):
                second_val=np.max(hist[other_peaks])
                second_peak = np.argwhere(np.logical_and(hist==second_val, other_peaks))[0,0]# The "and" here is to make sure that we dont end up selecting a non-peak
                theta = quadratic_fit(Q,hist,second_peak,ind_to_center)
                octave_oriented_pts.append((theta,s,y,x))
        oriented_pts.append(octave_oriented_pts)
    return oriented_pts

def create_descriptor(theta,y,x,half_hist_width,m):
    """
    SIFT Descriptor construction by trilinear histogram binning

    Parameters:
        theta: The orientation of the gradients
        y: y-coordinates of the gradients
        x: x-coordinates of the gradients
        half_hist_width: width of the histogram in the spatial axii

    """
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
    """
    Normalize the descriptor as done by Lowe et al. (2004)

    Parameters:
        descriptor: The descriptor
        max_val(optional): A threshold value
    Returns:
        descriptor: The normalized descriptor
    """
    descriptor /= np.sqrt(np.sum(descriptor**2))
    descriptor = np.minimum(descriptor,0.2)
    descriptor /= np.sqrt(np.sum(descriptor**2))
    return descriptor

def calculate_descriptors(pts,L,k=2**(1/3),sigma_0=1.6):
    """
    Creates descriptors for the given points

    Parameters:
        pts: The points of interest
        L: The scale space
        k: The scale factor
        sigma_0: The smallest scale in the scale space
    Returns:
        descriptors: The SIFT descriptors
    """
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

            descriptor = create_descriptor(rot_theta,rot_grid[1],rot_grid[0],half_hist_width,m)
            descriptor = normalize_descriptor(descriptor)
            descriptors.append(((2**o*y,2**o*x),descriptor))#Store the coordinates together with the descriptor
    return descriptors

def SIFT(im):
    """
    Constructs and returns SIFT descriptors of a given image

    Parameters:
        im: The image
    Returns:
        descriptors: The SIFT descriptors
    """
    r = 10#from paper, ration between curvature
    L = scale_space(im)
    D = [np.diff(octave,axis=0) for octave in L]
    pts = []
    for D_o in D:
        M_o=maximum_filter(np.abs(D_o),size=3,mode=('reflect','constant','constant'),cval=np.infty)
        max_inds = (M_o==D_o)&(np.abs(D_o)>0.03)#This theshold depends on the values in the image TODO: should be a parameter
        pts.append(list(zip(*np.nonzero(max_inds))))
    pts = taylor_expand(pts,D)#TODO: Here a taylor expansion should be done
    pts = eliminate_edges(pts,D)
    pts = determine_orientation(pts,L)
    descriptors = calculate_descriptors(pts,L)
    return descriptors

def match_descriptors(im1_descriptors,im2_descriptors):
    rows,cols = len(im1_descriptors),len(im2_descriptors)
    match = np.zeros((rows,cols))
    for idx1,desc1 in enumerate(im1_descriptors):
        for idx2,desc2 in enumerate(im2_descriptors):
            match[idx1,idx2] = np.sum(desc1[1]*desc2[1])
    rows_maxes = set(zip(np.arange(rows),np.argmax(match,axis=0)))
    cols_maxes = set(zip(np.argmax(match,axis=1),np.arange(cols)))
    good_matches= rows_maxes.intersection(cols_maxes)
    for row,col in good_matches:
        y_0,x_0 = im1_descriptors[row][0]
        y_1,x_1 = im2_descriptors[col][0]
        plt.plot([x_0,x_1],[y_0,y_1])
    plt.show()
    print("hej")

Im = Image.open("cman.tif")
im = np.array(Im)/255.0
descriptors = SIFT(im)
rot_Im = Im.rotate(45)
plt.imshow(rot_Im)
plt.show()
rot_im = np.array(rot_Im)/255.0
descriptors_rot = SIFT(rot_im)
match_descriptors(descriptors,descriptors_rot)
