import numpy as np
import topojson as tp
import geopandas as gpd
import pandas as pd
# import subprocess
import os
from shapely.geometry import Polygon, Point, LineString
from shapely import wkt
import pprint
from .basicTools import timeRecord



CASE2_ANGLE_MAPPING = {
    0: [0,1,2],
    1: [0,2,1],
    2: [1,2,0]
    # argmax: arc index pair and not pair
}
CASE3_ANGLE_MAPPING = {
    0: [0,1,2,3],
    1: [0,2,1,3],
    2: [0,3,1,2],
    3: [1,2,0,3],
    4: [1,3,0,2],
    5: [2,3,0,1],
    # argmax: arc index pair and not pair
}

@timeRecord
def topo_simplify(src_shp, dst_shp, tolerance=0.00001, simplify_algorithm='dp',
        simplify_with='shapely',):
    # simplification

    gdf = gpd.read_file(src_shp)


    # # 根据不同算法选择简化方法
    # if simplify_algorithm == 'vw':  # Visvalingam-Whyatt algorithm
    #     # 使用 `simplification` 库的 `vw_simplify` 方法
    #     from simplification.cutil import simplify_coords_vw
        
    #     def simplify_geometry(geom):
    #         if isinstance(geom, (Polygon, LineString)):
    #             return geom.simplify(tolerance, preserve_topology=True)
    #         return geom
       
    #     gdf['geometry'] = gdf['geometry'].apply(simplify_geometry)
    
    # elif simplify_algorithm == 'dp':  # Douglas-Peucker algorithm
    #     # 使用 Shapely 的 simplify 方法
    #     gdf['geometry'] = gdf['geometry'].simplify(tolerance, preserve_topology=True)
    
    # else:
    #     raise ValueError(f"Unknown simplification algorithm: {simplify_algorithm}")
    
    # # 将简化后的GeoDataFrame保存为新的Shapefile文件
    # gdf.to_file(dst_shp)
    
    
    topo = tp.Topology(gdf, topology=True, prequantize=False)
    
    # simplified = topo.toposimplify(
    #     epsilon=tolerance,
    #     simplify_algorithm='dp',
    #     simplify_with='shapely',
    #     prevent_oversimplify=False
    # ).to_gdf()
    simplified = topo.toposimplify(
        epsilon=tolerance,
        simplify_algorithm=simplify_algorithm,
        simplify_with=simplify_with,
        prevent_oversimplify=False
    ).to_gdf()

    simplified.to_file(dst_shp, driver="ESRI Shapefile")
    
    return dst_shp
    # cmd = f'ogr2ogr -r "ESRI Shapefile" {dst_shp} {tmp_json}'

def _get_angle(v1, v2):
    
    dot_product = np.dot(v1, v2)
    
    # magnitude
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    
    cosine_theta = dot_product / (m1 * m2)
    angle_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
    
    angle_deg = np.rad2deg(angle_rad)
    
    return angle_deg


def _get_length(v1, v2):
    return np.linalg.norm([v1[0]-v2[0], v1[1]-v2[1]])

def get_pair(gdf):
    
    # print(gdf)

    
    interior_polygons = gdf[gdf['interior'] == 1]
    
    touched_cfps = dict()
    
    for index, row in interior_polygons.iterrows():
        
        if row.area > 667 *15: continue
        
        adjacent_polygons = gdf[gdf.touches(row['geometry'])]
        touched_cfps[row.pid] = adjacent_polygons.pid.tolist()

    return touched_cfps


def get_topo_map(topo,  searching_field='pid'):
    topo_map = dict()
    for r in topo.output['objects']['data']['geometries']:
        searching_anchor = r["properties"][searching_field]
        arcs = r['arcs']
        
        topo_map[searching_anchor] = arcs
        
    return topo_map

def calc_abc_from_line_2d(x0, y0, x1, y1):
    a = y0 - y1
    b = x1 - x0
    c = x0*y1 - x1*y0
    return a, b, c

def find_intersection_point(a, b, c, d):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

    a0, b0, c0 = calc_abc_from_line_2d(x1, y1, x2, y2)
    a1, b1, c1 = calc_abc_from_line_2d(x3, y3, x4, y4)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    
    if x > max(x1, x2):
        # print('xmax')
        x = max(x1, x2)
    elif x < min(x1, x2):
        x = min(x1, x2)
        # print('xmmin')
    
    if y>max(y1, y2):
        y = max(y1,y2)
        # print('ymax')
    elif y<min(y1, y2):
        y = min(y1,y2)
        # print('ymin')
    
    return x, y

def point_to_line_projection(a, b, c):
    '''get the cross point where the c projected on line AB'''
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    
    ab_vector = b - a

    
    ac_vector = c - a

    
    proj_ac_ab = np.dot(ac_vector, ab_vector) / np.dot(ab_vector, ab_vector) * ab_vector

    
    intersection_point = a + proj_ac_ab
    x,y = intersection_point
    x1, y1 = a
    x2, y2 = b
    
    if x > max(x1, x2):
        x = max(x1, x2)
    elif x < min(x1, x2):
        x = min(x1, x2)
    
    if y>max(y1, y2):
        y = max(y1,y2)
    elif y<min(y1, y2):
        y = min(y1,y2)
    
    return [x, y]




def optimization_case1(topo, topo_map, interior_id, **kwargs):
    '''1 interior touched by 2 neighbor CFPs'''
    interior_arcs = topo_map[interior_id]
    undirected_arcs = [e if e > 0 else -e-1 for e in interior_arcs[0]]
    
    for arc in undirected_arcs:
        arc_coords = topo.output['arcs'][arc]
        
        topo.output['arcs'][arc] = [arc_coords[0], arc_coords[-1]]

def optimization_case2(topo, topo_map, interior_id, touched_ids):
    
    interior_arcs = topo_map[interior_id][0]
    
    # print(interior_arcs)
    
    if len(interior_arcs) == 2:
        optimization_case1(topo, topo_map, interior_id)
        return
    
    norm_inter_arcs = [e if e > 0 else -e-1 for e in interior_arcs]
    # print(norm_inter_arcs)
    tails = []
    for touched_id in touched_ids:
        
        _arcs = [e for e in topo_map[touched_id][0]]
        _arcs += [_arcs[0]]
        _norm_arcs = [e if e >= 0 else -e-1 for e in _arcs]
        # print('----')
        # print(_arcs)
        # print(_norm_arcs)
        
        for inter_arc in norm_inter_arcs:
            if inter_arc in _norm_arcs:
                index = _norm_arcs.index(inter_arc)
                tail_arc = _norm_arcs[index+1]
                # print(tail_arc, inter_arc)
                # tail_dir = 1 if _arcs[index + 1] >=0 else -1
                
                if _arcs[index + 1] >=0:
                    arc_coords = topo.output['arcs'][tail_arc][:2]
                else:
                    arc_coords = topo.output['arcs'][tail_arc][-2:][::-1]
                
                tails.append(arc_coords)
                
    tail_v =  [ [e[0][0] - e[1][0], e[0][1]- e[1][1] ] for e in tails]
    
    try:
        tail_angle = [
            _get_angle(tail_v[0], tail_v[1]),
            _get_angle(tail_v[0], tail_v[2]),
            _get_angle(tail_v[1], tail_v[2]),
        ]
    except:
        raise
    
    connect_ = CASE2_ANGLE_MAPPING.get(np.argmax(tail_angle)) # 0,2,1-> 0------2 1
    
    inter_p = find_intersection_point(
        tails[connect_[0]][0],
        tails[connect_[1]][0],
        tails[connect_[2]][0],
        tails[connect_[2]][1],
    )
    
    #update
    for inter_arc in norm_inter_arcs:
        arc_coords = topo.output['arcs'][inter_arc]
        topo.output['arcs'][inter_arc] = [
            arc_coords[0], inter_p, arc_coords[-1]
        ]
    
    # print(tail_angle)
    # print(tails[connect_[0]][0])
    # print(tails[connect_[1]][0])
    # print(tails[connect_[2]][0])
    # print(tails[connect_[2]][1])
    
    # print(inter_p)


def optimization_case3(topo, topo_map, interior_id, touched_ids):
    '''1 interior touched 4 CFPs'''
    interior_arcs = topo_map[interior_id][0]
    
    
    
    if len(interior_arcs) == 3:
        optimization_case2(topo, topo_map, interior_id, touched_ids)
        return
    
    if len(interior_arcs) != 4:
        print(topo.output['arcs'][interior_arcs[0]])
        raise
    else:
        a,b,c,d = interior_arcs
        na, nb, nc, nd = [e if e > 0 else -e-1 for e in [a,b,c,d]]
    
    
    # norm_inter_arcs = [e if e > 0 else -e-1 for e in interior_arcs]
    
    vertices = []
    
    
    start_a = topo.output['arcs'][na][-1 if a<0 else 0]
    start_b = topo.output['arcs'][nb][-1 if a<0 else 0]
    start_c = topo.output['arcs'][nc][-1 if a<0 else 0]
    start_d = topo.output['arcs'][nd][-1 if a<0 else 0]
    # [0, 1, 2, 3]
    # diagonal line from the start of a and c
    dist1 = _get_length(start_a, start_c)
    dist2 = _get_length(start_b, start_d)
    
    if dist1 >= dist2:
        p1 = point_to_line_projection(start_a, start_c, start_b)
        p2 = point_to_line_projection(start_a, start_c, start_d)
    
        b_coord = topo.output['arcs'][nb]
        topo.output['arcs'][nb]  = [b_coord[0], p1, b_coord[-1]]
    
        a_coord = topo.output['arcs'][na]
        topo.output['arcs'][na]  = [a_coord[0], p1, a_coord[-1]]
    
        c_coord = topo.output['arcs'][nc]
        topo.output['arcs'][nc]  = [c_coord[0], p2, c_coord[-1]]
    
        d_coord = topo.output['arcs'][nd]
        topo.output['arcs'][nd]  = [d_coord[0], p2, d_coord[-1]]
    
    else:
        p1 = point_to_line_projection(start_b, start_d, start_a)
        p2 = point_to_line_projection(start_b, start_d, start_c)
        
        a_coord = topo.output['arcs'][na]
        topo.output['arcs'][na]  = [a_coord[0], p1, a_coord[-1]]
        
        d_coord = topo.output['arcs'][nd]
        topo.output['arcs'][nd]  = [d_coord[0], p1, d_coord[-1]]
        
        b_coord = topo.output['arcs'][nb]
        topo.output['arcs'][nb]  = [b_coord[0], p2, b_coord[-1]]
        
        c_coord = topo.output['arcs'][nc]
        topo.output['arcs'][nc]  = [c_coord[0], p2, c_coord[-1]]
        
    










@timeRecord
def find_holes(input_shp, out_shp):
    
    
    gdf:gpd.GeoDataFrame = gpd.read_file(input_shp)
    
    gdf['geometry'] = gdf['geometry'].apply(lambda x: x.buffer(0))
    
    gdf['dissolve'] = 1
    
    gdf['interior'] = [0] * len(gdf)
    
    dissolved:gpd.GeoDataFrame = gdf.dissolve('dissolve').explode(index_parts=True)
    
    _all_interios = dissolved.interiors
    
    all_interiors = []
    
    pi = 0
    for i in _all_interios:
        if i == []: continue
        for ii in i:
            
            all_interiors.append({'geometry': Polygon(ii), 'polygon_id': pi})
            pi += 1
    try:
        all_interiors = gpd.GeoDataFrame(all_interiors, geometry='geometry')
    except:
        gdf.to_file(out_shp)
        return
    
    all_interiors['interior'] = [1] * len(all_interiors)
    
    adjacent_gdf = gdf[gdf.touches(all_interiors.unary_union)]
    
    left_gdf = gdf[~gdf.touches(all_interiors.unary_union)]
    
    
    adjacent_gdf = pd.concat([all_interiors, adjacent_gdf], ignore_index=True)
    adjacent_gdf = gpd.GeoDataFrame(adjacent_gdf, geometry='geometry')
    
    # adjacent_gdf.to_file(os.path.join(r'D:\data_transp\tmp\shp', 'adjacent_gdf.shp'))
    # adjacent_gdf = gpd.read_file(os.path.join(r'D:\data_transp\tmp\shp', 'adjacent_gdf.shp'))
    
    
    # print(adjacent_gdf)
    adjacent_gdf['area'] = adjacent_gdf.area
    adjacent_gdf['pid'] = list(range(len(adjacent_gdf)))
    
    
    
    touched_pairs = get_pair(adjacent_gdf)
    
    topo = tp.Topology(adjacent_gdf, topology=True, prequantize=False)
    topo_map = get_topo_map(topo, searching_field='pid')
    
    
    for interior_id, touched_ids in touched_pairs.items():
        if len(touched_ids) == 2:
            try:
                optimization_case1(topo, topo_map, interior_id)
            except:
                print('-->', 2)
        if len(touched_ids) == 3:
            try:
                optimization_case2(topo, topo_map, interior_id, touched_ids)
            except:
                print('-->', 3)
        if len(touched_ids) == 4:
            try:
                optimization_case3(topo, topo_map, interior_id, touched_ids)
            except:
                print('-->', 4)
        
    # exit()
    
    # left_gdf
    try:
        topo = topo.to_gdf()
        topo = topo[topo['interior'] != 1]
        gdf_new = pd.concat([left_gdf, topo], ignore_index=True)
        gdf_new = gpd.GeoDataFrame(gdf_new, geometry='geometry')
    except:
        gdf_new = gdf
        
    gdf_new.to_file(out_shp)
    # topo.to_file(out_shp.replace('_simplified.shp', '.shp'))
    
    
    
    # print(gdf.touches(all_interiors.unary_union))
    # exit()
    # concat_gdf = pd.concat([gdf, all_interiors], ignore_index=True)
    # concat_gdf = gpd.GeoDataFrame(concat_gdf, geometry='geometry')
    
    # exit()





def topo_simplify_v2(src_shp, dst_shp, tolerance=1):
    
    gdf = gpd.read_file(src_shp)

    topo = tp.Topology(gdf, topology=True, prequantize=False)
    
    '''
    # print(topo.output.keys())
    # dict_keys(['type', 'coordinates', 'objects', 'bbox', 'arcs'])
    [
        'topology', 
        [], 
        {
            'properties': {'value': 165}, 
            'type': 'Polygon', 
            'arcs': [[262, -284, -323, 263, -297, 264, -295, 265]], 
            'id': 89}]
        }
        (12794482.395435471, 4192480.1400969876, 12800846.34466334, 4197662.2705),
        ...
    ]
    '''
    print(topo.output['objects'])
    # exit()
    
    arcs = []
    for idx, arc in enumerate(topo.output['arcs']):
        arcs.append({'geometry': LineString(arc), 'aid': idx})
    
    arcs = gpd.GeoDataFrame(arcs, geometry='geometry')
    
    arcs.to_file(os.path.join(dst_shp, 'arcs_4u'), driver="ESRI Shapefile")
    
    
    


if __name__ == '__main__':
    
    test_out_dir = r'D:\data_transp\tmp'
    test_dir = os.path.join(test_out_dir, 'field_map_final1_simp8.shp')
    
    find_holes(test_dir)
    
    # test topo
    
    # test_out_dir = r'D:\data_transp\tmp'
    
    # topo_simplify_v2(
    #     os.path.join(test_out_dir, 'shp', 'topotest_4u.shp'),
    #     os.path.join(test_out_dir, 'shp'),
    #     tolerance = 8
    # )