
# this version doesn't have ChromAn compiled, so directly load from src:
# modify to match the new structure from Nikon:
import sys
import numpy as np
import os, sys
import argparse
import tifffile
import shutil
# Parse the arguments
sys.path.append(r"/lab/weissman_imaging/puzheng/Softwares")
from ChromAn.src.file_io.data_organization import search_fovs_in_folders, Color_Usage
from ChromAn.src.file_io.data_organization import Data_Organization, search_fovs_in_folders
from ChromAn.src.file_io.dax_process import DaxProcesser
from ChromAn.src.correction_tools.illumination import illumination_correction

dw_path = "/lab/weissman_imaging/puzheng/Softwares/deconwolf/builddir/dw"
ref_path = "/lab/weissman_imaging/puzheng/Corrections/DW_PSFs"
# run deconwolf:
def deconwolf(img, channel, zstep=500,
              dw_path=dw_path, ref_path=ref_path,
              gpu=True, overwrite=False):

    channel_2_psf = {
        '748':'Alexa750',
        '637':'Alexa647',
        '545':'Atto565',
        '477':'beads',
        '488':'Alexa488',
        '405':'DAPI',
    }
    import tempfile
    import subprocess
    from skimage.io import imsave, imread
    # search excutive:
    if not os.path.exists(dw_path):
        raise FileNotFoundError(f"DeconWolf: {dw_path} not fuound")
    # search ref:
    matched_psf = [os.path.join(ref_path, _f) for _f in os.listdir(ref_path) 
                   if (channel_2_psf[str(channel)] in _f) and (str(zstep) in _f)
                   and ('psf' in _f) and ('tif'==_f.split(os.extsep)[-1])
                  ]
    #print(matched_psf)
    if len(matched_psf) == 0:
        raise FileNotFoundError(f"No PSF reference found given channel: {channel} in folder: {ref_path}")
    elif len(matched_psf) > 1:
        raise FileNotFoundError(f"Multiple PSF reference found given channel: {channel} in folder: {ref_path}")
    else:
        matched_psf = matched_psf[0]
        print(f"PSF: {matched_psf}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        imsave(f"{tmp_dir}/img.tif",img)
        gpu_str = " --gpu" if gpu else ""
        overwrite_str = " --overwrite" if overwrite else ""
        command = f"{dw_path} --out {tmp_dir}/decon.tif --iter 100 {gpu_str}{overwrite_str} --verbose 1 {tmp_dir}/img.tif {matched_psf}"
        print("DeconWolf command: ", command)
        subprocess.run(command, check=True,shell=True)
        decon_img = imread(f"{tmp_dir}/decon.tif")
        
    return decon_img


if __name__ == '__main__':
    # process args with deconwolf
    parser = argparse.ArgumentParser(description="DeconWolf processing for MERFISH datasets")
    # Add arguments
    parser.add_argument("--fov_id", type=int,help="Field of View")
    parser.add_argument("--output_fov_id", type=int, default=None, help="re-indexed fov id, None if not re-indexed")
    parser.add_argument("--data_path", type=str, help="Image Path")
    parser.add_argument("--output_path", type=str, help="Output Path")
    parser.add_argument("--correction_path", type=str, default = None, help="Path to image correction files")

    parser.add_argument("--color_usage", type=str, help="Location for color usage file")
    parser.add_argument("--data_organization", type=str, help="Location for data organization file")
    
    parser.add_argument("--gpu", type=bool, default=True, help="Use GPU (True/False)")
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite existing files (True/False)")
    
    args = parser.parse_args()
    # FOV_ID
    fov_id = args.fov_id
    output_fov_id = args.output_fov_id if args.output_fov_id is not None else fov_id

    #fov_id = 1
    data_path = args.data_path
    #data_path = r"/lab/weissman_imaging/puzheng/4T1Tumor/20240917-F320-5-0909_MF7_mch"
    output_path = args.output_path
    #output_path = r"/lab/weissman_imaging/puzheng/MERFISH_data/20240917-F320_MF7"
    # Load data
    # Color usage file marks the organization of imaging output folders:
    color_usage = Color_Usage(args.color_usage)
    #color_usage_filename = os.path.join(data_path, 'Analysis', 'color_usage_MF7_mch.csv')
    #color_usage = Color_Usage(color_usage_filename)
    # Data organization file marks the target organziation of MERlin inputs:
    data_organization = Data_Organization(args.data_organization)
    #data_organization_filename = f'/lab/weissman_imaging/puzheng/Softwares/Weissman_MERFISH_Scripts/merlin_parameters/dataorganization/20240917-MF7_20bit_v1.csv'
    #data_organization = Data_Organization(data_organization_filename,)
    # Correction folder stores illumination correction profiles:
    correction_path = args.correction_path
    #correction_path = r'/lab/weissman_imaging/puzheng/Corrections/Data/20240906-beads_correction/Corrections'
    # additional args, use GPU:
    use_gpu = getattr(args, 'gpu')
    #use_gpu = True
    overwrite = getattr(args, 'overwrite', True) # set default to True
    
    _folders, _fovs = search_fovs_in_folders(data_path)
    for _hyb, _row in color_usage.iterrows():
        # get information for this round:
        _round_channels, _round_infos = color_usage.get_channel_info_for_round(_hyb)
        # run deconwolf for valid channels:
        _valid_channels = [_ch for _ch, _info in zip(_round_channels, _round_infos) 
                        if len(_info) > 1 and _info[0]=='m' and _info[1:].isdigit()]
        _valid_ids = [int(_info[1:]) for _ch, _info in zip(_round_channels, _round_infos) 
                    if _ch in _valid_channels]
        # if deconwolf is needed:
        if len(_valid_ids) > 0: 
            # get save_file:
            _img_type = data_organization.loc[data_organization['bitNumber']==_valid_ids[0], 'imageType'].values[0]
            _img_round = data_organization.loc[data_organization['bitNumber']==_valid_ids[0], 'imagingRound'].values[0]
            _save_filename = os.path.join(output_path, f"{_img_type}_{output_fov_id}_{_img_round}.tif")
            if os.path.exists(_save_filename):
                print(f"file for hyb={_hyb}, fov={fov_id}: {_save_filename} already exists, skip")
                continue

            # then if doesn't work, start loading data:
            _daxp = DaxProcesser(os.path.join(os.path.join(data_path,_hyb), _fovs[fov_id]))
            _daxp._load_image()
            # run illumination correction:
            #_daxp._corr_illumination(correction_folder=correction_path, correction_channels=_valid_channels, rescale=False) # test illumination correction before deconwolf
            # run deconwolf:
            _dw_ims = [deconwolf(getattr(_daxp, f'im_{_ch}'), _ch, gpu=use_gpu, tile=tile, tile_size=tile_size) for _ch in _valid_channels ]
            _dw_ims = illumination_correction(_dw_ims, channels=_valid_channels, correction_folder=correction_path) # illumination correction after deconwolf
            # replace attrs:
            for _im, _ch in zip(_dw_ims, _valid_channels):
                setattr(_daxp, f"im_{_ch}", _im)
            # re-save to tiff:
            _im_sizes = _daxp._FindImageSize(_daxp.filename)
            print(f"Save to file: {_save_filename}")
            tifffile.imwrite(_save_filename, 
                            np.stack([getattr(_daxp, f"im_{_ch}") for _ch in _daxp.channels]).transpose((1,0,2,3)).reshape(len(_daxp.channels)*_im_sizes[0], _im_sizes[1], _im_sizes[2]),
                            imagej=True)
        # no step required, just copy:
        else:
            _img_type = np.concatenate([data_organization.loc[data_organization['channelName']==_ri, 'imageType'].values for _ri in _round_infos])
            _img_round = np.concatenate([data_organization.loc[data_organization['channelName']==_ri, 'imagingRound'].values for _ri in _round_infos])
            if len(_img_type) == 0:
                print(f"hyb={_hyb} doesn't have valid rows in data_organization, skip")
                continue
            else:
                _img_type = np.unique(_img_type)[-1]
                _img_round = np.unique(_img_round)[-1]
            _save_filename = os.path.join(output_path, f"{_img_type}_{output_fov_id}_{_img_round}.dax")
            if os.path.exists(_save_filename):
                print(f"file for hyb={_hyb}, fov={fov_id}: {_save_filename} already exists, skip")
            else:
                # copy:
                print(f"Copying file hyb={_hyb},fov={fov_id} to: {_save_filename}")
            # additonal files:
            shutil.copyfile(os.path.join(os.path.join(data_path,_hyb), _fovs[fov_id]), _save_filename)
            # copy .inf, .off, .power, .xml:
            shutil.copyfile(os.path.join(os.path.join(data_path,_hyb), _fovs[fov_id]).replace('.dax', '.inf'), _save_filename.replace('.dax', '.inf')) if not os.path.exists(_save_filename.replace('.dax', '.inf')) else None
            shutil.copyfile(os.path.join(os.path.join(data_path,_hyb), _fovs[fov_id]).replace('.dax', '.off'), _save_filename.replace('.dax', '.off')) if not os.path.exists(_save_filename.replace('.dax', '.off')) else None
            shutil.copyfile(os.path.join(os.path.join(data_path,_hyb), _fovs[fov_id]).replace('.dax', '.power'), _save_filename.replace('.dax', '.power')) if not os.path.exists(_save_filename.replace('.dax', '.power')) else None
            shutil.copyfile(os.path.join(os.path.join(data_path,_hyb), _fovs[fov_id]).replace('.dax', '.xml'), _save_filename.replace('.dax', '.xml')) if not os.path.exists(_save_filename.replace('.dax', '.xml')) else None
            
            
    # finish
    