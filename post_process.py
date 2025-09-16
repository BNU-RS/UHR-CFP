
'''
 _______          __                      .__    ___________.__       .__       .___
 \      \ _____ _/  |_ __ ______________  |  |   \_   _____/|__| ____ |  |    __| _/
 /   |   \\__  \\   __\  |  \_  __ \__  \ |  |    |    __)  |  |/ __ \|  |   / __ | 
/    |    \/ __ \|  | |  |  /|  | \// __ \|  |__  |     \   |  \  ___/|  |__/ /_/ | 
\____|__  (____  /__| |____/ |__|  (____  /____/  \___  /   |__|\___  >____/\____ | 
        \/     \/                       \/            \/            \/           \/ 
__________                            .__                                           
\______   \_____ _______   ____  ____ |  |                                          
 |     ___/\__  \\_  __ \_/ ___\/ __ \|  |                                          
 |    |     / __ \|  | \/\  \__\  ___/|  |__                                        
 |____|    (____  /__|    \___  >___  >____/                                        
                \/            \/    \/                                              
___________         __                        __  .__                               
\_   _____/__  ____/  |_____________    _____/  |_|__| ____   ____                  
 |    __)_\  \/  /\   __\_  __ \__  \ _/ ___\   __\  |/  _ \ /    \                 
 |        \>    <  |  |  |  | \// __ \\  \___|  | |  (  <_> )   |  \                
/_______  /__/\_ \ |__|  |__|  (____  /\___  >__| |__|\____/|___|  /                
        \/      \/                  \/     \/                    \/                

@ Authorized by:
    /             
 __/ __  , ______ 
(_/_/ (_/_/ / / <_
       /          
      '           >> dymwan@gmail.com
      
@ Cursive in https://patorjk.com/software/taag
'''
import shutil
import numpy as np
import os
from osgeo import gdal
import cv2
from skimage.morphology import skeletonize, medial_axis
from post_pro.simplify import find_holes
from post_pro.basicTools import timeRecord
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import time
import os
import glob


# from post_pro.utils import eliminate_salt_in_bound_area_v2
from post_pro.basicTools import getIsolateCls, timeRecord, getIndivCls
from post_pro.utils import get_cross_end_points, get_indivdual_lines, \
    get_led_by_fmatrix, get_topo_relations, get_dangling_lines, get_dual,\
    reconstruct_dlout, get_nacked_field_bound, set_lines_width,\
    get_extended_dangling_lines, eliminate_dangling_lines, fill_skeleton, \
    fix_broken_in_neighbor5, add_coordinate_to_graph, add_direction_to_graph,\
    get_danglinglinemap, raster2poly

from post_pro.simplify import topo_simplify

cross_kernel_3 = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0],
], np.uint8)



DEBUG = False

if not DEBUG:
    warnings.filterwarnings("ignore")






def readIm(dir):
    ds=  gdal.Open(dir)
    im = ds.ReadAsArray()
    return im, ds

def get_splitting_map(w, h, stride=5000):
    # NO OVERLAP
    
    nw = w// stride if w % stride == 0 else w // stride + 1 
    nh = h// stride if h % stride == 0 else h // stride + 1 
    anchors = []
    for wi in range(nw):
        for hi in range(nh):
            ws = wi* stride
            we = ws + stride
            we = we if we < w else w
            
            hs = hi *stride
            he = hs + stride
            he = he if he < h else h
            anchors.append([ws, we, hs, he])
    return anchors


def enclose_exploding_field(fieldbin, boundbin, bound_buf=2):
    
    fieldbin_dilate = cv2.dilate(fieldbin, np.ones([3,3], np.uint8), iterations=bound_buf)
    fieldbin_dilate -=  fieldbin
    
    boundbin[ (boundbin != 1) & (fieldbin_dilate == 1)] = 1
    
    return boundbin

@timeRecord
def get_skeleton(binmap, downsample=None, method='zhang'):
    if downsample is not None:
        w, h = binmap.shape[-2:]
        # dsize = [int(e/downsample) for e in shape]
        dsize = [int(e/downsample) for e in [h, w]]
        
        sklt = skeletonize(
            cv2.resize(binmap, dsize=dsize, interpolation=cv2.INTER_NEAREST)
        ).astype(np.uint8)
        
        return sklt # TODO here return the downsmapled sklt
        # sklt =  cv2.resize(sklt.astype(np.uint8), dsize=shape, interpolation=cv2.INTER_NEAREST)
        # return skeletonize(sklt, method)
    else:
        return skeletonize(binmap, method=method).astype(np.uint8)
    


@timeRecord
def modify_areal_bound(bb, fb, min_area=667):

    
    # open bb
    bb = cv2.dilate(bb, np.ones([3,3], np.uint8), iterations=1)
    bb = cv2.erode(bb, np.ones([3,3], np.uint8), iterations=1)
    
    inverse_bb = bb.copy()
    inverse_bb += 1
    inverse_bb[inverse_bb == 2] = 0
    
    inverse_bb[fb == 1] = 0

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(inverse_bb, connectivity=4)

    filling_mums = []
    for i in range(retval):
        
        if i == 0:
            continue
        
        x,y,w,h,n = stats[i]
        
        if n > 10*min_area:
            continue
        
        if n <= min_area:
            filling_mums.append(i)
        else:
            xyr = abs(float(w/h))
            if xyr < 0.5 or xyr > 2:
                filling_mums.append(i)
            elif float(n / w* h) < 0.2:
                filling_mums.append(i)
            else:
                continue
        
    mask = np.isin(labels, filling_mums)
    
    bb[mask] = 1
    
    return bb


def progress(percent, msg, tag):

    print(percent, msg, tag)

def save_im(arr, ref_ds=None, dst_dir='', dt=gdal.GDT_Float32):
    
    assert len(arr.shape) <= 3
    
    if len(arr.shape) == 2:
        x,y = arr.shape
        nb=1
    else:
        nb, x, y = arr.shape
    
    
    drv:gdal.Driver = gdal.GetDriverByName('GTiff')
    
    if dt == gdal.GDT_Byte:
        ods:gdal.Dataset = drv.Create(dst_dir, y, x, nb, dt, options=["TILED=YES", "COMPRESS=LZW"])
    else:
        ods:gdal.Dataset = drv.Create(dst_dir, y, x, nb, dt)
    
    if ref_ds is not None:
        ods.SetProjection(ref_ds.GetProjection())
        ods.SetGeoTransform(ref_ds.GetGeoTransform())
    
    if nb==1:
        oband = ods.GetRasterBand(1)
        oband.WriteArray(arr)
    else:
        for bi in range(nb):
            oband = ods.GetRasterBand(bi)
            oband.WriteArray(arr[bi,:,:])
            
    ods.FlushCache()
    ods = None
    
def dilate(bin_img:torch.Tensor, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out




@timeRecord
def add_width_to_graph(graph, boundmap, indiv_sklt_map, width_thresh=6, device='cpu'):
    
    bound_dist = cv2.distanceTransform(
        np.where(
            boundmap == 1, 1, 0
        ).astype(np.uint8), cv2.DIST_L2, 5
        
    )
    
    
    indiv_sklt_map_t = torch.from_numpy(indiv_sklt_map).to(device)
    
    bound_dist = torch.from_numpy(bound_dist).to(device)
    
    line_ids = list(graph.lines.keys())
    
    width = torch.empty([3, len(line_ids)]).to(device=device)
    
    for idx, lineid in enumerate(line_ids):
        
        line_width = bound_dist[indiv_sklt_map_t == lineid]
        mean_width = torch.mean(line_width)
        min_width =  torch.min( line_width)
        
        width[ 0, idx] = lineid
        width[ 1, idx] = mean_width
        width[ 2, idx] = min_width
    
    # print(width_thresh)
    daul_cases = width[0,:][width[1,:] >= width_thresh].detach().cpu().numpy().astype(np.uint16)
    
    
    
    for dual_line_id in daul_cases:
        graph.lines[dual_line_id].set_daul()
        
        for nid in graph.lines[dual_line_id].linked:
            # 叠加判断
            graph.nodes[nid].set_daul()
            
    
    
    daul_line_ids = set()
    
    
    for idx, lineid in enumerate(line_ids):
        if graph.lines[lineid].is_daul():
            daul_line_ids.add(lineid)
            
        elif all([graph.nodes[_nid].is_daul() for _nid in graph.lines[lineid].linked]):
            daul_line_ids.add(lineid)
            
            
    
    daul_line_ids = torch.tensor(list(daul_line_ids)).to(device)
    
    
    daul_line_map = torch.isin(indiv_sklt_map_t, daul_line_ids)#, assume_unique=True)
    
    
    dilatemap = torch.zeros_like(indiv_sklt_map_t, dtype=torch.float32, device=device)
    
    
    
    dilatemap[daul_line_map] = bound_dist[daul_line_map]
    dilatemap -= width_thresh
    
    dilatemap[(dilatemap <= 1) & (dilatemap > -width_thresh)] = 1
    dilatemap[dilatemap == -width_thresh] = 0
    
    dilatemap = dilatemap.unsqueeze(0).unsqueeze(0)
    
    output = torch.zeros_like(dilatemap).long().to(device)
    for cur_dilate_width in torch.unique(dilatemap.long()):
        if cur_dilate_width == 0: continue
        dilatemap = dilate(dilatemap, ksize=3)
        
        output[(dilatemap>= cur_dilate_width) & (output==0)] = 1
    
    # print(output.shape)
    output = output.squeeze(0).squeeze(0)
    
    # print(output.shape)
    # exit()
    return output.detach().cpu().numpy()
        
      

@timeRecord 
def get_extended_dangling_line_map(dangling_line_map, boundbin=None, max_extend_length=50):
    
    W,H = dangling_line_map.shape[-2:]
    
    contours = []
    _contours, _ = cv2.findContours(dangling_line_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    nc = len(_contours)
    
    
    starts =  np.zeros([nc*2,2], np.int16)
    ends =    np.zeros([nc*2,2], np.int16)
    
    ic = 0
    for _c in _contours:

        _c = _c.squeeze()

        
        _len_c = len(_c)
        if _len_c < 2:
            continue
        
        _len_c = int(_len_c/2)+1
        _c = _c[:_len_c]
        
        
        pair1 = []
        pair2 = []
        
        start1 = _c[0]
        start2 = _c[-1]
        
        try:
            thrsh = abs( start1[0] - start2[0]) + abs(start1[1]-start2[1]) // 3

            thrsh = max([10, thrsh])
        except:
            continue
        
        # print(thrsh)
            
        find1=True #forward
        find2=True # backward
        
        step = 0
        
        _acc1 = 0
        _acc2 = 0
        
        last1 = start1.copy()
        last2 = start2.copy()
        
        while any([find1, find2]):
            
            
            if step + 2 >= _len_c:
            
                pair1.append(_c[step+1])
                pair2.append(_c[-2-step])
                find1 = False
                find2 = False
        
            else:
                
                #TODO judgement may rely on L2 dist rather than L1
                if find1:
                    cur_1 = _c[step+1]
                    _acc1 += abs( cur_1[0] - last1[0]) + abs(cur_1[1]-last1[1])
                    if _acc1 < thrsh:
                        last1 = cur_1
                        # step += 1
                    else:
                        pair1.append(cur_1)
                        find1 = False
                    
                if find2:
                    cur_2 = _c[-2-step]
                    _acc2 += abs( cur_2[0] - last2[0]) + abs(cur_2[1]-last2[1])
                    if _acc1 < thrsh:
                        last2 = cur_2
                        # step += 1
                    else:
                        pair2.append(cur_2)
                        find2 = False
                
                # if any([find1, find2]):
                step += 1
                
        else:
            pair1.append(start1)
            pair2.append(start2)
        
        starts[ic, :] = pair1[0]
        ends[ic, :] = pair1[1]
        starts[ic+nc, :] = pair2[0]
        ends[ic+nc, :] = pair2[1]

        ic += 1
    
    # construct enxtending lines
    
    _end_to_start = ends-starts
    
    _dist = np.linalg.norm(_end_to_start, axis=1)
    
    
    
    tos =  (ends + _end_to_start / _dist[:, np.newaxis] * max_extend_length).astype(np.int16)
    
    
    
    extentd_lines = np.linspace(ends, tos, num=max_extend_length+1, axis=1,dtype=np.int16) # (2043, 50, 2)
    
    # extended_dist = np.linalg.norm(tos-ends, axis=1) # L2
    extended_dist = np.sum(np.abs(tos-ends), axis=1) # L1
    #make sure _dist gt 0 and no strange pairs from cv2.findContour
    extentd_lines = extentd_lines[(extended_dist <= max_extend_length) | (_dist > 0),:,:]
    
    
    extentd_xs = extentd_lines[:,:,0]#.flatten()
    extentd_ys = extentd_lines[:,:,1]#.flatten()
    # print('a', extentd_xs.shape) #(654, 51, 2)
    # print('a', extentd_ys.shape) #(654, 51, 2)
    return extentd_xs, extentd_ys

@timeRecord
def walking_extend(extentd_xs, extentd_ys, skeleton, device='cpu'):
    _anchor_backup = skeleton[0,0]
    w,h = skeleton.shape[-2:]
    
    # walking_start = torch.LongTensor(walking_start).to(device)
    # helo_index = torch.LongTensor([[0],[1],[2],[5],[8],[7],[6],[3],[4]])#.unsqueeze(0).repeat(1,1, w*h).to(device)
    helo_index = torch.LongTensor([0,1,2,5,8,7,6,3,4]).to(device)#.unsqueeze(0).repeat(1,1, w*h).to(device)
    sklt_t = torch.from_numpy(skeleton).to(device).float().unsqueeze(0).unsqueeze(0)
    shift_window = torch.nn.Unfold(3,1,1,1)
    sklt_ats = shift_window(sklt_t).view(1,9,*sklt_t.shape[-2:])[0,helo_index,:,:]
    
    #TODO for debug
    walked = torch.zeros(skeleton.shape).float().to(device).unsqueeze(0).unsqueeze(0)
    
    
    nlins, nsteps = extentd_xs.shape #(654, 51)
    
    extentd_xs = torch.from_numpy(extentd_xs).to(device).long()
    extentd_ys = torch.from_numpy(extentd_ys).to(device).long()
    
    #
    extentd_xs[(extentd_xs >= h) | (extentd_xs < 0)] = 0
    extentd_ys[(extentd_ys >= w) | (extentd_ys < 0)] = 0
    
    
    
    last_xs = None
    last_ys = None
    
    cur_xs = extentd_xs[:, 0]
    cur_ys = extentd_ys[:, 0]
    
    sklt_prj = sklt_ats[:, cur_ys, cur_xs]
    sklt_halo = torch.sum(sklt_prj, dim=0)
    
    # print(extentd_ys.shape)
    extentd_ys = extentd_ys[sklt_halo == 2]
    extentd_xs = extentd_xs[sklt_halo == 2]
    # print(extentd_ys.shape)
    
    last_xs = extentd_xs[:, 0]
    last_ys = extentd_ys[:, 0]
    
    walked_ats = shift_window(walked).view(1,9,*sklt_t.shape[-2:])[0,helo_index,:,:]
    
    
    walking=True
    # steps = 1
    for steps in range(1, nsteps-1):
        
        cur_xs = extentd_xs[:, steps]
        cur_ys = extentd_ys[:, steps]
        
        
        
        pause_ys = cur_ys == last_ys
        pause_xs = cur_xs == last_xs
        
        pauses = torch.bitwise_and(pause_ys, pause_xs)
        
        cur_sklt_halo = sklt_ats[:, cur_ys, cur_xs]
        cur_walk_halo = walked_ats[:-1, cur_ys, cur_xs]
        cur_walk_halo -= torch.roll(cur_walk_halo, 1, dims=0)
        cur_walk_halo[cur_walk_halo == -1] = 0
        
        
        touched_sklt = (cur_sklt_halo[-1, :] == 1) & (torch.sum(cur_sklt_halo[:-1,:], dim=0) >=2)
        meet = torch.sum(cur_walk_halo, dim = 0) >= 2
        
        touched_sklt = torch.bitwise_or(touched_sklt, meet)
        
        keep_walkings = torch.bitwise_or(~touched_sklt, pauses)
        walk_thistime = torch.bitwise_and(~touched_sklt, ~pauses)
        
        #TODO for debug
        
        
        walked[:,:, cur_ys[walk_thistime], cur_xs[walk_thistime]] = 1
        
        sklt_t[:,:, cur_ys[walk_thistime], cur_xs[walk_thistime]] = 1
        
        #update sklt
        sklt_ats = shift_window(sklt_t).view(1,9,*sklt_t.shape[-2:])[0,helo_index,:,:]
        walked_ats = shift_window(walked).view(1,9,*sklt_t.shape[-2:])[0,helo_index,:,:]
        
        extentd_xs = extentd_xs[keep_walkings, :]
        extentd_ys = extentd_ys[keep_walkings, :]

        last_xs = extentd_xs[:, steps]
        last_ys = extentd_ys[:, steps]
        
        if extentd_xs.shape == torch.Size([0]):
            break
        
    else:
        # 
        pass
        
    sklt = sklt_t.view(w,h).detach().cpu().numpy().astype(np.uint8)
    
    return sklt

def eliminate_narrow_and_salt(skltmap:np.ndarray, widest_narrow=1.5, largest_salt=5, largest_poly=65.136718):
    '''
    widest_narrow:
        threshold for max wide of the narrow place (in distance transform)
    largest_salt:
        the pixel numbers for eliminating the floting salts, after narrow clean
    41.6875 = 667/16 which means the skeleton is derived in 4 times downsamples (1m)
    65.1367 = 667/16 which means the skeleton is derived in 4 times downsamples (0.8m)
    '''
    
    invert_sklt = np.where(skltmap == 1, 0, 1).astype(np.uint8)
    
    dist = cv2.distanceTransform(invert_sklt, cv2.DIST_L2, 5)
    
    skltmap[dist<=widest_narrow] = 1
    
    
    skltmap = get_skeleton(skltmap)
    
    
    # return skltmap
    
    
    n, indiv = cv2.connectedComponents(skltmap)
    
    count = np.bincount(indiv.flatten())
    # print(count)
    
    salt_nums = []
    for idx, v in enumerate(count):
        if v <= largest_salt:
            salt_nums.append(idx)
    
    
    mask = np.isin(indiv, salt_nums)
    skltmap[mask] = 0
    
    
    # elminate inner salt
    invert_sklt = np.where(skltmap == 1, 0, 1).astype(np.uint8)
    
    
    n, indiv = cv2.connectedComponents(invert_sklt, connectivity=4)
    count = np.bincount(indiv.flatten())
    salt_nums = []
    for idx, v in enumerate(count):
        if v <= largest_poly:
            salt_nums.append(idx)
    
    
    mask = np.isin(indiv, salt_nums)
    skltmap[mask] = 1
    
    skltmap = get_skeleton(skltmap)
    
    return skltmap


def fill_cfp(sklt, fbmerge):
    reverse_sklt = np.where(sklt==1, 0, 1).astype(np.uint8)
    
    dist_to_sklt = cv2.distanceTransform(reverse_sklt, cv2.DIST_L2, 5)
    
    seed = np.where(dist_to_sklt > 2, 1, 0).astype(np.uint8)
    unknown = seed == 0
    nsd, seed = cv2.connectedComponents(seed, connectivity=4)
    
    seed[seed > 0] += 1
    seed[unknown] = 0
    
    growmap = dist_to_sklt.copy()
    growmap[growmap> 255] = 255
    growmap = growmap.astype(np.uint8)
    growmap = cv2.cvtColor(growmap, cv2.COLOR_BGRA2RGB)
    
    markers = cv2.watershed(growmap, seed)
    
    markers[markers >= 1] += 1
    markers[markers == -1] = 1
    
    field_intersect_markers = np.where(fbmerge == 1, markers, 0)
    
    # return markers
    marker_ct = np.bincount(markers.flatten())
    inters_ct = np.bincount(field_intersect_markers.flatten(), minlength=len(marker_ct))
    
    ratio = inters_ct / marker_ct
    
    fieldblkid = []
    for i in range(2, len(ratio)):
        # if i < 2:
        #     continue
        # TODO this place should be reconstruct
        if ratio[i] > 0.2: #???????
            fieldblkid.append(i)
    
    mask = np.isin(markers, fieldblkid)
    bound = markers == 1
    
    markers[mask] = 2
    markers[~mask] = 3
    # markers[1] = 1
    
    return markers


def prepare_to_r2v(fieldmap, device='cpu', scaling=4):
    w, h = fieldmap.shape[-2:]
    maps = get_splitting_map(w, h, 5000)
    
    n, fieldmap = cv2.connectedComponents(fieldmap, connectivity=4)
    
    for ws, we, hs, he in maps:
        # print(ws, we, hs, he)
        sub_field = fieldmap[ws:we, hs:he]
        
        sub_field = torch.from_numpy(sub_field).to(device).float()
        
        # TODO this 
        sub_field = _eliminate_bound(sub_field.unsqueeze(0).unsqueeze(0), scaling//2)
    
        fieldmap[ws:we, hs: he] = sub_field
    
    return fieldmap



def _eliminate_bound(im, iter):
    shift_window = torch.nn.Unfold(3,1,1,1)
    
    w, h = im.shape[-2:]
    
    for i in range(iter):
        ats = shift_window(im).view(1,9, w, h)
        
        bound_ct = torch.sum(ats, dim=1)
        
        convert = (ats[0, 4,:,:] == 0) & (bound_ct[0,:,:] >= 2)
        # print(convert.shape, im.shape)
        max_, miter = torch.max(ats, dim=1)
        im[:,:,convert] = max_[0, convert]
    
    return im.squeeze(0).squeeze(0).detach().cpu().numpy()

'''
                                       _____                    __  .__               
  ____  ___________   ____           _/ ____\_ __  ____   _____/  |_|__| ____   ____  
_/ ___\/  _ \_  __ \_/ __ \   ______ \   __\  |  \/    \_/ ___\   __\  |/  _ \ /    \ 
\  \__(  <_> )  | \/\  ___/  /_____/  |  | |  |  /   |  \  \___|  | |  (  <_> )   |  \
 \___  >____/|__|    \___  >          |__| |____/|___|  /\___  >__| |__|\____/|___|  /
     \/                  \/                           \/     \/                    \/
'''

@timeRecord
def postpro_patch(i1m, i2m=None, working_scale=2,  min_length=10,  max_extend_length=150, device='cpu'):
    
    pad_size = 5
    
    
    w,h = i1m.shape[-2:]
    
    # protect_bound = 20
    
    
    #bb = boundbin
    bbmerge = np.zeros_like(i1m, dtype=np.uint8)
    fbmerge = np.zeros_like(i1m, dtype=np.uint8)
    bbmerge[i1m == 1]= 1
    fbmerge[i1m == 2] = 1
    
    if i2m is not None:
        bbmerge[i2m == 1] = 1
        fbmerge[i2m == 2] = 1
    
    
    
    # generate a bound for the exploding field area TODO
    bbmerge = enclose_exploding_field(fbmerge, bbmerge, bound_buf=2)
    
    # filling holes only based on bound image
    '''eliminate the narrow and smale holes inside the bound'''
    bbmerge = modify_areal_bound(bbmerge, fbmerge, min_area=667)

    
    
    if working_scale is not None:
        w, h = bbmerge.shape[-2:]
        bbmerge = cv2.resize(bbmerge, dsize=[int(e/working_scale) for e in [h,w]], interpolation=cv2.INTER_NEAREST)
    bbmerge = np.pad(bbmerge, ((pad_size,pad_size), (pad_size,pad_size)), mode='symmetric')
    
    
    

    sklt = get_skeleton(bbmerge)
    
    
    
    sklt[:, 0] = 1
    sklt[:, -1] = 1
    sklt[0, :] = 1
    sklt[-1, :] = 1
    
    # original
    
    
    
    point_map = get_cross_end_points(sklt, device=device)

    sklt = fix_broken_in_neighbor5(sklt, point_map, device=device)

    sklt = get_skeleton(sklt)
    
    # fix_broken_in_neigh5
    
    point_map = get_cross_end_points(sklt, device=device)
    num_lines, indivi_skeleton = get_indivdual_lines(sklt, point_map)
    nump_point, indiv_point_map = getIndivCls(point_map, 1,2)
    
    
    
    M_pl, N_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)
    topoG = get_topo_relations(M_pl, N_pl)
    add_coordinate_to_graph(topoG, indiv_point_map, device=device)
    add_direction_to_graph(topoG, indivi_skeleton, device=device)
    
    
    halo_map = add_width_to_graph(topoG, bbmerge, indivi_skeleton, width_thresh=10, device=device)
    
    
    
    bbmerge[halo_map == 1] = 0
    
    
    
    sklt = get_skeleton(bbmerge)
    
    # dualline
       
    point_map = get_cross_end_points(sklt, device=device)
    num_lines, indivi_skeleton = get_indivdual_lines(sklt, point_map)
    nump_point, indiv_point_map = getIndivCls(point_map, 1,2)
    M_pl, N_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)
    topoG = get_topo_relations(M_pl, N_pl)
    add_coordinate_to_graph(topoG, indiv_point_map, device=device)
    add_direction_to_graph(topoG, indivi_skeleton, device=device)
    
    # TODO ????
    # reconstruct_dlout(indivi_skeleton, indiv_point_map, lines=topoG.lines, nodes=topoG.nodes, device=device)
    
    
    danglinglinemap = get_danglinglinemap(topoG, indivi_skeleton, min_length=min_length)
    
    # extended_map, init_points, extentd_xs, extentd_ys = get_extended_dangling_line_map(danglinglinemap, max_extend_length=50)
    extentd_xs, extentd_ys = get_extended_dangling_line_map(danglinglinemap, max_extend_length=max_extend_length)
    
    sklt = walking_extend(extentd_xs, extentd_ys, sklt, device=device)
    
    
    
    torch.cuda.empty_cache()
    
    #may at the last
    sklt = eliminate_narrow_and_salt(sklt, widest_narrow=1.5, largest_salt=5, largest_poly=41.6875)
    
    
    
    if working_scale is not None:
        # w, h = fbmerge.shape[-2:]
        dsize = [int(e/working_scale) for e in [h,w]]
        
        fbmerge = cv2.resize(fbmerge, dsize=dsize, interpolation=cv2.INTER_NEAREST)
        
    fbmerge = np.pad(fbmerge, ((pad_size,pad_size), (pad_size,pad_size)), mode='symmetric')
    
    filled_sklt = fill_cfp(sklt, fbmerge)
    
    out = filled_sklt[pad_size:-pad_size, pad_size:-pad_size]
    if working_scale is not None:
        out = cv2.resize(out, dsize=[h,w], interpolation=cv2.INTER_NEAREST)
    
    return out, fbmerge[pad_size:-pad_size, pad_size:-pad_size], sklt[pad_size:-pad_size, pad_size:-pad_size]


def add_frame(sklt):
    sklt[:, 0] = 1
    sklt[:, -1] = 1
    sklt[0, :] = 1
    sklt[-1, :] = 1



def pre_skletonize(input_bin):
    
    sklt = get_skeleton(input_bin)
    add_frame(sklt)

    point_map = get_cross_end_points(sklt, device=device)
    sklt = fix_broken_in_neighbor5(sklt, point_map, device=device)
    sklt = get_skeleton(sklt)
    
    return sklt



def TRC(sklt=None, input_bin=None, device='cpu'):
    
    if sklt is None:
        assert input_bin is not None
        sklt = pre_skletonize(input_bin)
        
    point_map = get_cross_end_points(sklt, device=device)
    num_lines, indivi_skeleton = get_indivdual_lines(sklt, point_map)
    nump_point, indiv_point_map = getIndivCls(point_map, 1,2)
    
    M_pl, N_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)
    topoG = get_topo_relations(M_pl, N_pl)
    add_coordinate_to_graph(topoG, indiv_point_map, device=device)
    add_direction_to_graph(topoG, indivi_skeleton, device=device)
    
    return topoG, indivi_skeleton, indiv_point_map

def _draw_direction(topoG, isklt):
    line_ids = list(topoG.lines.keys())
    
    _direction=  np.zeros(isklt.shape, dtype=np.float32)#.to(isklt.device)
    
    for lid in line_ids:
        _direction[isklt==lid] = topoG.lines[lid].dir
    
    return _direction


def DLD(input_bin, topoG, indivi_skeleton, width_thresh):
    halo_map = add_width_to_graph(topoG, input_bin, indivi_skeleton, width_thresh=width_thresh, device=device)
    input_bin[halo_map == 1] = 0
    sklt = get_skeleton(input_bin)
    return sklt, input_bin

def DLE(sklt, min_length, max_extend_length, device='cpu'):
    point_map = get_cross_end_points(sklt, device=device)
    num_lines, indivi_skeleton = get_indivdual_lines(sklt, point_map)
    nump_point, indiv_point_map = getIndivCls(point_map, 1,2)
    M_pl, N_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)
    topoG = get_topo_relations(M_pl, N_pl)
    add_coordinate_to_graph(topoG, indiv_point_map, device=device)
    add_direction_to_graph(topoG, indivi_skeleton, device=device)

    danglinglinemap = get_danglinglinemap(topoG, indivi_skeleton, min_length=min_length)
    
    # extended_map, init_points, extentd_xs, extentd_ys = get_extended_dangling_line_map(danglinglinemap, max_extend_length=50)
    extentd_xs, extentd_ys = get_extended_dangling_line_map(danglinglinemap, max_extend_length=max_extend_length)
    
    sklt = walking_extend(extentd_xs, extentd_ys, sklt, device=device)
    return sklt


def postpro_patch_manuscript(i1m, i2m=None, working_scale=None,  min_length=10,  max_extend_length=150, width_thresh=10, device='cpu'):
    
    pad_size = 5
    
    
    w,h = i1m.shape[-2:]
    
    # protect_bound = 20
    
    
    #bb = boundbin
    bbmerge = np.zeros_like(i1m, dtype=np.uint8)
    fbmerge = np.zeros_like(i1m, dtype=np.uint8)
    bbmerge[i1m == 1]= 1
    fbmerge[i1m == 2] = 1
    
    if i2m is not None:
        bbmerge[i2m == 1] = 1
        fbmerge[i2m == 2] = 1
    
    
    
    # generate a bound for the exploding field area TODO
    bbmerge = enclose_exploding_field(fbmerge, bbmerge, bound_buf=2)
    
    # filling holes only based on bound image
    '''eliminate the narrow and smale holes inside the bound'''
    bbmerge = modify_areal_bound(bbmerge, fbmerge, min_area=667)

    
    
    if working_scale is not None:
        w, h = bbmerge.shape[-2:]
        bbmerge = cv2.resize(bbmerge, dsize=[int(e/working_scale) for e in [h,w]], interpolation=cv2.INTER_NEAREST)
    bbmerge = np.pad(bbmerge, ((pad_size,pad_size), (pad_size,pad_size)), mode='symmetric')
    
    bbmerge[0,:] = 1
    bbmerge[-1,:] = 1
    bbmerge[:,0] = 1
    bbmerge[:,-1] = 1
    
    
    sklt = pre_skletonize(bbmerge)
    
    if width_thresh is not None:
        topoG, indivi_skeleton, indivi_points = TRC(input_bin=bbmerge, device=device)
        sklt, bbmerge = DLD(bbmerge, topoG, indivi_skeleton, width_thresh)
    # return sklt, bbmerge
    # sklt = DLE(sklt, min_length, max_extend_length, device=device)
    
    
    torch.cuda.empty_cache()
    
    #may at the last
    sklt = eliminate_narrow_and_salt(sklt, widest_narrow=1.5, largest_salt=5, largest_poly=41.6875)
    
    
    
    if working_scale is not None:
        # w, h = fbmerge.shape[-2:]
        dsize = [int(e/working_scale) for e in [h,w]]
        
        fbmerge = cv2.resize(fbmerge, dsize=dsize, interpolation=cv2.INTER_NEAREST)
        
    fbmerge = np.pad(fbmerge, ((pad_size,pad_size), (pad_size,pad_size)), mode='symmetric')
    
    filled_sklt = fill_cfp(sklt, fbmerge)
    
    out = filled_sklt[pad_size:-pad_size, pad_size:-pad_size]
    if working_scale is not None:
        out = cv2.resize(out, dsize=[h,w], interpolation=cv2.INTER_NEAREST)
    
    return out, fbmerge[pad_size:-pad_size, pad_size:-pad_size], sklt[pad_size:-pad_size, pad_size:-pad_size]

    
    
    

'''
#  ________  _______   _______   ________        ________  ________      
# |\   ___ \|\  ___ \ |\  ___ \ |\   __  \      |\   __  \|\   ____\     
# \ \  \_|\ \ \   __/|\ \   __/|\ \  \|\  \     \ \  \|\  \ \  \___|_    
#  \ \  \ \\ \ \  \_|/_\ \  \_|/_\ \   ____\     \ \   _  _\ \_____  \   
#   \ \  \_\\ \ \  \_|\ \ \  \_|\ \ \  \___|      \ \  \\  \\|____|\  \  
#    \ \_______\ \_______\ \_______\ \__\          \ \__\\ _\ ____\_\  \ 
#     \|_______|\|_______|\|_______|\|__|           \|__|\|__|\_________\
#                                                            \|_________|

'''
@timeRecord
def postpro_untill_r2v(r1mdir, dst_folder, r2mdir=None, device='cpu', width_thresh=10):
    # print(r1mdir)
    # time.sleep(2)
    basename = '_'.join(os.path.basename(r1mdir).split('.')[:-1])
    
    dst_folder = os.path.join(dst_folder, basename)
    
    if os.path.isfile(os.path.join(dst_folder, basename + '.shp')):
        return
    
    if not os.path.isdir(dst_folder):
        os.makedirs(dst_folder)
    else:
        # return
        pass
    
    i1m, ds1m = readIm(r1mdir)
    
    #TODO lxy
    gt = list(ds1m.GetGeoTransform())

    # gt[-1] = -1
    gt[-1] = -gt[-5]
    ds1m.SetGeoTransform(gt)
    
    if r2mdir is not None:
        i2m, ds1m = readIm(r2mdir)
    # else: 
    #     i2m = None
    
    w, h = i1m.shape[-2:]
    maps = get_splitting_map(w, h, 5000)
    
    # outmap = np.zeros_like(i1m, dtype=np.uint8)
    #debug
    outmap = np.zeros_like(i1m, dtype=np.float32)
    
    #TODO for testing
    fbout = np.zeros_like(i1m, dtype=np.uint8)
    skltbout = np.zeros_like(i1m, dtype=np.uint8)
    
    # if globals().get('DEBUG', False):
    #     tbar = maps
    # else:
    #     tbar = tqdm(maps, total=len(maps), ncols=100)
    
    for ws, we, hs, he in maps:
        # print(ws, we, hs, he)
        i1mi = i1m[ws: we, hs: he]
        if r2mdir is not None:
            i2mi = i2m[ws: we, hs: he]
        else: 
            i2mi = None

        
        
        try:
            # isklt = postpro_patch(i1mi, i2mi, device=device, working_scale=2)
            # isklt, ifbmerge, iskltt = postpro_patch(i1mi, i2mi, device=device, working_scale=None)
            
            
            # TODO DEBUG
            isklt, ifbmerge, iskltt = postpro_patch_manuscript(i1mi, i2mi, device=device, width_thresh=width_thresh, working_scale=None)
            
            
            # sklt, bbmerge = postpro_patch_manuscript(i1mi, i2mi, device=device, width_thresh=width_thresh, working_scale=None)
            # save_im(sklt, ds1m, r'D:\CPWE\BaiduSyncdisk\manuscript_fig_2312\product_result\post_ablation\test\test_sklt.tif', gdal.GDT_Byte)
            # save_im(bbmerge, ds1m, r'D:\CPWE\BaiduSyncdisk\manuscript_fig_2312\product_result\post_ablation\test\test_bbmerge.tif', gdal.GDT_Byte)
            # exit()


            # save_im(_dirction, ds1m, r'D:\CPWE\BaiduSyncdisk\manuscript_fig_2312\product_result\post_ablation\test\_dirction.tif', gdal.GDT_Float32)
            # save_im(bbmerge, ds1m, r'D:\CPWE\BaiduSyncdisk\manuscript_fig_2312\product_result\post_ablation\test\bbmerge.tif', gdal.GDT_Byte)
            # save_im(indivi_skeleton, ds1m, r'D:\CPWE\BaiduSyncdisk\manuscript_fig_2312\product_result\post_ablation\test\indivi_skeleton.tif', gdal.GDT_UInt16)
            # save_im(i1mi, ds1m, r'D:\CPWE\BaiduSyncdisk\manuscript_fig_2312\product_result\post_ablation\test\i1mi.tif', gdal.GDT_Byte)
            # save_im(indivi_skeleton, ds1m, r'D:\CPWE\BaiduSyncdisk\manuscript_fig_2312\product_result\post_ablation\test\indivi_skeleton.tif', gdal.GDT_UInt16)
            # save_im(indiv_point_map, ds1m, r'D:\CPWE\BaiduSyncdisk\manuscript_fig_2312\product_result\post_ablation\test\indiv_point_map.tif', gdal.GDT_UInt16)
            
            # isklt, ifbmerge, iskltt = postpro_patch_manuscript(i1mi, i2mi, device=device, working_scale=None)
            
            
            # print('------------------------------------')
            # exit()
            
            
            
            outmap[ws: we, hs: he] = isklt #.transpose(1,0)#[::-1,::-1]
            fbout[ws: we, hs: he] = ifbmerge #.transpose(1,0)#[::-1,::-1]
            skltbout[ws: we, hs: he] = iskltt #.transpose(1,0)#[::-1,::-1]
        except:
            raise
    
    # dst_folder = r'D:\CPWE\BaiduSyncdisk\manuscript_fig_2312\product_result\testv2'
    
    # save_im(outmap, ds1m, os.path.join(dst_folder, 's7_indiv_filled.tif'), dt=gdal.GDT_Byte)
    # # save_im(outmap, ds1m, os.path.join(dst_folder, 's4_width.tif'), dt=gdal.GDT_Float32)
    field = np.where(outmap==3, 0, 1).astype(np.uint8)
    
    field_map = prepare_to_r2v(field, device=device, scaling=4)
    
    
    save_im(field_map, ds1m, os.path.join(dst_folder, basename + '_field_map.tif'), gdal.GDT_UInt16)
    save_im(fbout, ds1m, os.path.join(dst_folder, basename + '_fbout.tif'), gdal.GDT_Byte)
    save_im(skltbout, ds1m, os.path.join(dst_folder, basename + '_skltout.tif'), gdal.GDT_Byte)
    
    raster2poly(
                os.path.join(dst_folder, basename + '_field_map.tif'), 
                os.path.join(dst_folder, basename + '_raw.shp'), 
                'layer')
    
    # os.rmdir(os.path.join(dst_folder, basename + '_field_map.tif'))
    # os.rmdir(os.path.join(dst_folder, basename + '_field_map.tif'))
    # os.rmdir(os.path.join(dst_folder, basename + '_field_map.tif'))
    # os.rmdir(os.path.join(dst_folder, basename + '_field_map.tif'))
    
    topo_simplify(
        os.path.join(dst_folder, basename + '_raw.shp'),
        os.path.join(dst_folder, basename + '_simplified.shp'),
        tolerance = 1
    )
    
    
    # find_holes(
    #     os.path.join(dst_folder, basename + '_simplified.shp'),
    #     os.path.join(dst_folder, basename + '.shp')
    # )
    
    
'''
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
'''


if __name__ == '__main__':
    
    # 指定device 
    device='cuda:0'
    # device='cpu'
    
    src_folder = r'/home4/lxy/GUANGXI/Guangxi_uav_parcel/guangxi_uav_result'
    dst_folder = r'/home4/lxy/GUANGXI/Guangxi_uav_parcel/guangxi_uav_shp'
    

    mission_name = os.path.basename(src_folder)
    mission_dst_folder = os.path.join(dst_folder, mission_name)
    if not os.path.isdir(mission_dst_folder):
        os.makedirs(mission_dst_folder)
    else:
        pass
    
    # 单个影像
    # tifname = r'3_result.tif'
    # r1mdir = os.path.join(src_folder, tifname)
    # postpro_untill_r2v(r1mdir, mission_dst_folder, r2mdir=None, device=device, width_thresh=None)


    # 批量处理
    tif_files = glob.glob(os.path.join(src_folder, '*.tif'))
    for tif_file in tif_files:
        print(f'Processing {tif_file}...')
        postpro_untill_r2v(tif_file, mission_dst_folder, r2mdir=None, device=device, width_thresh=None)
        print(f'Finished processing {tif_file}')