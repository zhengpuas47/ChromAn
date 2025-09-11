import sys, os
import numpy as np
import geopandas as gpd
from fishtank.seg.convert import polygons_to_masks, masks_to_polygons

# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from segmentation_tools.segment import cellposeSegment

if __name__ == "__main__":
    fov = sys.argv[1]
    _fl= f'polygons_{fov}.json'
    print(_fl)
    nuc_polygon_folder = r'/lab/weissman_imaging/puzheng/MERFISH_analysis/SC/Postanalysis/20250430-Hela500k_SC1/nuclei_polygons/'
    cyto_polygon_folder = r'/lab/weissman_imaging/puzheng/MERFISH_analysis/SC/Postanalysis/20250430-Hela500k_SC1/cytoplasm_polygons/'

    merged_nuc_folder = r'/lab/weissman_imaging/puzheng/MERFISH_analysis/SC/Postanalysis/20250430-Hela500k_SC1/merged/nuclei_polygons'
    if not os.path.exists(merged_nuc_folder):
        os.makedirs(merged_nuc_folder)
    merged_cyto_folder = r'/lab/weissman_imaging/puzheng/MERFISH_analysis/SC/Postanalysis/20250430-Hela500k_SC1/merged/cytoplasm_polygons'
    if not os.path.exists(merged_cyto_folder):
        os.makedirs(merged_cyto_folder)
        
    
    nuc_polygons = gpd.read_file(os.path.join(nuc_polygon_folder, _fl))
    cyto_polygons = gpd.read_file(os.path.join(cyto_polygon_folder, _fl))
    
    nuc_masks = polygons_to_masks(nuc_polygons, bounds=[0,0,2304,2304], shape=[11,2304,2304])

    cyto_masks = polygons_to_masks(cyto_polygons, bounds=[0,0,2304,2304], shape=[11,2304,2304])

    merged_masks = cellposeSegment.match_cyto_nuclei_masks(cyto_masks, nuc_masks)

    merged_cyto_polygons = masks_to_polygons(np.abs(merged_masks))
    merged_cyto_polygons = cellposeSegment._append_polygon_features(
        merged_cyto_polygons, fov=np.unique(nuc_polygons['fov'])[0],
        z_offsets=np.arange(-5,6)*1.2, 
        stage_position=[np.unique(nuc_polygons['x_offset'])[0],np.unique(nuc_polygons['y_offset'])[0]]
    )

    merged_nuc_masks = -1 * merged_masks.copy()
    merged_nuc_masks[merged_nuc_masks < 0] = 0
    merged_nuc_polygons = masks_to_polygons(merged_nuc_masks)
    merged_nuc_polygons = cellposeSegment._append_polygon_features(
        merged_nuc_polygons, fov=np.unique(nuc_polygons['fov'])[0],
        z_offsets=np.arange(-5,6)*1.2, 
        stage_position=[np.unique(nuc_polygons['x_offset'])[0],np.unique(nuc_polygons['y_offset'])[0]],
    )
    
    # save
    merged_cyto_polygons.to_file(os.path.join(merged_cyto_folder, _fl))
    merged_nuc_polygons.to_file(os.path.join(merged_nuc_folder, _fl))
        