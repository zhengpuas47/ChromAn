import numpy as np

from scipy.ndimage import maximum_filter,minimum_filter,gaussian_filter
from scipy.spatial.distance import cdist
from scipy.optimize import leastsq

def sphere(center,radius,imshape=None):
    """Returns an int array (size: n x len(center)) with the xyz... coords of a sphere(elipsoid) of radius in imshape"""
    radius_=np.array(radius,dtype=int)
    if len(radius_.shape)==0:
        radius_ = np.array([radius]*len(center),dtype=int)
    xyz = np.array(np.indices(2*radius_+1),dtype=float)
    radius__=np.array(radius_,dtype=float)
    for i in range(len(xyz.shape)-1):
        radius__=np.expand_dims(radius__,axis=-1)
    xyz_keep = np.array(np.where(np.sum((xyz/radius__-1)**2,axis=0)<1))
    xyz_keep = xyz_keep-np.expand_dims(np.array(radius_,dtype=int),axis=-1)+np.expand_dims(np.array(center,dtype=int),axis=-1)
    xyz_keep = xyz_keep.T
    if imshape is not None:
        xyz_keep=xyz_keep[np.all((xyz_keep>=0)&(xyz_keep<np.expand_dims(imshape,axis=0)),axis=-1)]
    return xyz_keep

def gauss_ker(sig_xyz=[2,2,2],sxyz=16,xyz_disp=[0,0,0]):
    dim = len(xyz_disp)
    xyz=np.indices([sxyz+1]*dim)
    for i in range(len(xyz.shape)-1):
        sig_xyz=np.expand_dims(sig_xyz,axis=-1)
        xyz_disp=np.expand_dims(xyz_disp,axis=-1)
    im_ker = np.exp(-np.sum(((xyz-xyz_disp-sxyz/2.)/sig_xyz**2)**2,axis=0)/2.)
    return im_ker

def add_source(im_,pos=[0,0,0],h=200,sig=[2,2,2]):
    pos = np.array(pos)+0.5
    im=np.array(im_,dtype=float)
    pos_int = np.array(pos,dtype=int)
    xyz_disp = -pos_int+pos
    im_ker = gauss_ker(sig_xyz=sig,sxyz=int(np.max(sig)*5),xyz_disp=xyz_disp)
    im_ker_sz = np.array(im_ker.shape,dtype=int)
    pos_min = pos_int-im_ker_sz/2
    pos_max = pos_min+im_ker_sz
    im_shape = np.array(im.shape)
    def in_im(pos__):
        pos_=np.array(pos__,dtype=int)
        pos_[pos_>=im_shape]=im_shape[pos_>=im_shape]#-1
        pos_[pos_<0]=0
        return pos_
    pos_min_ = in_im(pos_min)
    pos_max_ = in_im(pos_max)
    pos_min_ker = pos_min_-pos_min
    pos_max_ker = im_ker_sz+pos_max_-pos_max
    #print zip(pos_min_ker,pos_max_ker),zip(pos_min_,pos_max_),zip(pos_min,pos_max)
    slices_ker = [slice(pm,pM)for pm,pM in zip(pos_min_ker,pos_max_ker)]
    slices_im = [slice(pm,pM)for pm,pM in zip(pos_min_,pos_max_)]
    im[slices_im]+=im_ker[slices_ker]*h
    return im
def subtract_source(im,pfit):
    return add_source(im,pos=pfit[1:4],h=-pfit[0],sig=pfit[-3:])
def plus_source(im,pfit):
    return add_source(im,pos=pfit[1:4],h=pfit[0],sig=pfit[-3:])

def get_seed_points_base(im,gfilt_size_min=1,gfilt_size_max=3,filt_size=3,th_seed=0.,th_std=0.,max_num=None,
                         use_snr=False,hot_pix_th=0,return_h=False):
    """Get the seed points in an image.
    #1 perform a gaussian filter
    #2 find local maxima within a radius 3 above th_seed from the minimum
    #3 remove hot pixels (high multiplicity of xy positions with different z)
    """
    im_plt=np.array(im,dtype=np.float32)

    max_filt = maximum_filter(im_plt,filt_size)
    min_filt = minimum_filter(im_plt,filt_size)
    g_filt = gaussian_filter(im_plt,gfilt_size_max)
    g_filt_sm = gaussian_filter(im_plt,gfilt_size_min)
    im_plt2 = (max_filt==im_plt)&(min_filt!=im_plt)
    z,x,y = np.where(im_plt2)
    h = g_filt_sm[z,x,y]-g_filt[z,x,y]
    if th_std>0:
        h_all = g_filt_sm-g_filt
        h_mn,h_std = np.mean(h_all),np.std(h_all)
        keep = (h-h_mn)>h_std*th_std
        x,y,z,h = x[keep],y[keep],z[keep],h[keep]
    snr = 1.*g_filt_sm[z,x,y]/g_filt[z,x,y]
    
    if use_snr:
        keep = snr>th_seed
    else:
        keep = h>th_seed
    x,y,z = x[keep],y[keep],z[keep]
    h,snr = h[keep],snr[keep]
    #get rid of hot pixels
    if hot_pix_th>0 and len(x)>0:
        xy = y*np.max(x)+x
        xy_,cts_ = np.unique(xy,return_counts=True)
        bad_xy = xy_[cts_>hot_pix_th]
        keep = np.array([xy_ not in bad_xy for xy_ in xy],dtype=bool)
        x,y,z = x[keep],y[keep],z[keep]
        snr=snr[keep]
        h = h[keep]
    centers = np.array([z,x,y])
    #sort by absolute brightness or signal to noise ratio (snr)
    if not use_snr:
        ind = np.argsort(h)[::-1]
    else:
        ind = np.argsort(snr)[::-1]
    centers = np.array([z[ind],x[ind],y[ind]])
    if return_h:
        centers = np.array([z[ind],x[ind],y[ind],h[ind]])
    if max_num is not None:
        centers = centers[:,:max_num]
    return centers

def fitsinglegaussian_fixed_width(data,center,radius=5,n_approx=10,width_zxy=[1.,1.,1.]):
    """Returns (height, x, y,z, width_x, width_y,width_z,bk)
    for the 3D gaussian fit for <radius> around a 3Dpoint <center> in the 3Dimage <data>
    <width_zxy> are the widths of the gaussian
    """
    data_=data
    dims = np.array(data_.shape)
    if center is  not None:
        center_z,center_x,center_y = center
    else:
        xyz = np.array([np.ravel(temp_) for temp_ in np.indices(data_.shape)])
        data__=data_[xyz[0],xyz[1],xyz[2]]
        args_high = np.argsort(data__)[-n_approx:]
        center_z,center_x,center_y = np.median(xyz[:,args_high],axis=-1)
    
    xyz = sphere([center_z,center_x,center_y],radius,imshape=dims).T
    if len(xyz[0])>0:
        data__=data_[xyz[0],xyz[1],xyz[2]]
        sorted_data = np.sort(data__)#np.sort(np.ravel(data__))
        bk = np.median(sorted_data[:n_approx])
        height = (np.median(sorted_data[-n_approx:])-bk)
            
        width_z,width_x,width_y = np.array(width_zxy)
        params_ = (height,center_z,center_x,center_y,bk)
        
        def gaussian(height,center_z, center_x, center_y,
                     bk=0,
                     width_z=width_zxy[0], 
                     width_x=width_zxy[1], 
                     width_y=width_zxy[2]):
            """Returns a gaussian function with the given parameters"""
            width_x_ = np.abs(width_x)
            width_y_ = np.abs(width_y)
            width_z_ = np.abs(width_z)
            height_ = np.abs(height)
            bk_ = np.abs(bk)
            def gauss(z,x,y):
                g = bk_+height_*np.exp(
                    -(((center_z-z)/width_z_)**2+((center_x-x)/width_x_)**2+
                      ((center_y-y)/width_y_)**2)/2.)
                return g
            return gauss
        def errorfunction(p):
            f=gaussian(*p)(*xyz)
            g=data__
            #err=np.ravel(f-g-g*np.log(f/g))
            err=np.ravel(f-g)
            return err
        p, success = leastsq(errorfunction, params_)
        p=np.abs(p)
        p = np.concatenate([p,width_zxy])
        return  p,success
    else:
        return None,None

def fast_local_fit(im,centers,radius=5,width_zxy=[1,1,1],return_good=False):
    """
    Given a set of seeds <centers> in a 3d image <im> iteratively 3d gaussian fit around the seeds for <radius> and with fixed <width_zxy>
    Retruns a numpy array of size Nx(height, x, y, z, width_x, width_y,width_z,background) where N~len(centers). Bad fits are disregarded.
    """
    ps=[]
    good = []
    im_=np.array(im)
    for center in centers:
        p,success = fitsinglegaussian_fixed_width(im_,center,radius=radius,n_approx=5,width_zxy=width_zxy)
        good.append(False)
        if p is not None:
            if np.max(np.abs(p[1:4]-center))<radius:
                ps.append(p)
                good[-1]=True
    if return_good:
        return np.array(ps),np.array(good,dtype=bool)
    return np.array(ps)

def fit_seed_points_base(im,centers,width_zxy=[1.,1.,1.],radius_fit=5,n_max_iter = 10,max_dist_th=0.25):
    """
    Given a set of seeds <centers> in a 3d image <im> iteratively 3d gaussian fit around the seeds (in order of brightness) and subtract the gaussian signal.
    Retruns a numpy array of size Nx(height, x, y, z, width_x, width_y,width_z,background) where N~len(centers). Bad fits are disregarded.
    Warning: Generally a bit slow. In practice, the faster version fast_local_fit is used.
    """
    #print "Fitting:" +str(len(centers[0]))+" points"
    z,x,y = centers
    
    if len(x)>0:
        #get height of the points and order by brightness
        h = [im[int(z_),int(x_),int(y_)] for z_,x_,y_ in zip(z,x,y)]
        inds = np.argsort(h)[::-1]
        z,x,y = z[inds],x[inds],y[inds]
        zxy = np.array([z,x,y],dtype=int).T
        
        #fit the points in order of brightness and at each fit subtract the gaussian signal
        ps = []
        im_subtr = np.array(im,dtype=float)
        for center in zxy:
            p,success = fitsinglegaussian_fixed_width(im_subtr,center,radius=radius_fit,n_approx=5,width_zxy=width_zxy)
            if p is not None:
                ps.append(p)
                im_subtr = subtract_source(im_subtr,p)

        im_add = np.array(im_subtr)

        max_dist=np.inf
        n_iter = 0
        while max_dist>max_dist_th:
            ps_1=np.array(ps)
            ps_1=ps_1[np.argsort(ps_1[:,0])[::-1]]
            ps = []
            ps_1_rem=[]
            for p_1 in ps_1:
                center = p_1[1:4]
                im_add = plus_source(im_add,p_1)
                p,success = fitsinglegaussian_fixed_width(im_add,center,radius=radius_fit,n_approx=5,width_zxy=width_zxy)
                if p is not None:
                    ps.append(p)
                    ps_1_rem.append(p_1)
                    im_add = subtract_source(im_add,p)
            ps_2=np.array(ps)
            ps_1_rem=np.array(ps_1_rem)
            dif = ps_1_rem[:,1:4]-ps_2[:,1:4]
            max_dist = np.max(np.sum(dif**2,axis=-1))
            n_iter+=1
            if n_iter>n_max_iter:
                break
        return ps_2
    else:
        return np.array([])

def fitmultigaussian(data,centers,radius=10,n_approx=10,width_zxy=[1.,1.,1.],min_width=0.5,fix_width=False):
    """Returns (height, x, y, z, width_x, width_y,width_z,background)
    for the 3D gaussian fit parameters (unconstrained, except for widths>min_width, height>0,background>0) for each point in <centers>
    A spherical neighbourhood of <radius> from the 3d image <data> is used.
    <n_approx> is the list of points in the neighbourhood for which to estimate the paramaters before optimizing for fitting.
    
    Warning: In practice this loosely constrained version is only used to estimate the widths. fitsinglegaussian_fixed_width behaves more robustly
    """
    data_=np.array(data,dtype=float)
    dims = np.array(data_.shape)
    xyz_unq=set()
    params=[]
    for center in centers:
        xyz = sphere(center,radius,imshape=dims)
        for xyz_ in xyz:
            xyz_unq.add(tuple(xyz_))
        data__=data_[xyz.T[0],xyz.T[1],xyz.T[2]]
        bk = np.median(np.sort(np.ravel(data__))[:n_approx])
        height = (np.median(np.sort(np.ravel(data__))[-n_approx:])-bk)
        center_z,center_x,center_y = center
        width_z,width_x,width_y = np.array(width_zxy)-min_width
        if fix_width:
            params_ = (height,center_z,center_x,center_y,bk)
        else:
            params_ = (height,center_z,center_x,center_y,width_z,width_x,width_y,bk)
        params.append(params_)
    params = np.array(params)
    xyz_unq = np.array([val for val in xyz_unq]).T
    dist_bk = cdist(xyz_unq.T,centers)
    dist_bk[dist_bk<1]=1
    weigh_bk = dist_bk/np.expand_dims(np.sum(dist_bk,axis=-1),-1)
    data_=data_[xyz_unq[0],xyz_unq[1],xyz_unq[2]]
    def gaussian(height,center_z, center_x, center_y, width_z=width_zxy[0]-min_width, 
                 width_x=width_zxy[1]-min_width, 
                 width_y=width_zxy[2]-min_width, 
                 bk=0,min_width=min_width):
        """Returns a gaussian function with the given parameters"""
        
        width_x_ = np.abs(width_x)+float(min_width)
        width_y_ = np.abs(width_y)+float(min_width)
        width_z_ = np.abs(width_z)+float(min_width)
        height_ = np.abs(height)
        bk_ = np.abs(bk)
        def gauss(z,x,y):
            g = bk_+height_*np.exp(
                -(((center_z-z)/width_z_)**2+((center_x-x)/width_x_)**2+
                  ((center_y-y)/width_y_)**2)/2.)
            return g
        return gauss
    def errorfunction(p):
        p_ = np.reshape(p,[len(centers),-1])
        bk_map = np.dot(weigh_bk,np.abs(p_[:,-1]))
        f=bk_map+np.sum([gaussian(*p__)(*xyz_unq) for p__ in p_[:,:-1]],0)
        g=data_
        #err=np.ravel(f-g-g*np.log(f/g))
        err=np.ravel(f-g)
        #print np.mean(err**2)
        return err
    p, success = leastsq(errorfunction, params)
    p = np.reshape(p,[len(centers),-1])
    p=np.abs(p)
    #p[:1:4]+=0.5
    if fix_width:
        p = np.concatenate([p[:,:-1],[width_zxy]*len(p),np.expand_dims(p[:,-1],-1)],axis=-1)
    else:
        p[:,4:7]+=min_width
    return  np.reshape(p,[len(centers),-1]),success