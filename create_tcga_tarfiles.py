import sys,os,glob,shutil,json,tarfile,io,h5py,gc,pickle,time
import numpy as np
import pandas as pd
import openslide
import idr_torch


def split_into_subsets_with_indices(arr, num):
    # Create a list of tuples (value, index)
    indexed_arr = [(value, index) for index, value in arr.items()]
    
    # Sort the array in descending order based on value
    indexed_arr.sort(reverse=True, key=lambda x: x[0])
    
    # Initialize 8 subsets with empty lists
    subsets = [[] for _ in range(num)]
    subsets_indices = [[] for _ in range(num)]
    
    # Initialize the sum of elements in each subset
    subset_sums = [0] * num
    
    # Distribute elements into subsets
    for value, original_index in indexed_arr:
        # Find the subset with the smallest sum
        min_subset_index = subset_sums.index(min(subset_sums))
        # Add the current number and its index to this subset
        subsets[min_subset_index].append(value)
        subsets_indices[min_subset_index].append(original_index)
        # Update the sum of this subset
        subset_sums[min_subset_index] += value
    
    return subsets, subsets_indices, subset_sums


def get_count():
    patches_dir = '/data/zhongz2/tcga/TCGA-ALL2_256/patches'
    save_root = '/data/zhongz2/tcga_tars/'
    if idr_torch.rank == 0:
        os.makedirs(save_root, exist_ok=True)
    all_h5_filenames = sorted(glob.glob(patches_dir + '/*.h5'))
    index_splits = np.array_split(np.arange(len(all_h5_filenames)), indices_or_sections=idr_torch.world_size)

    arr = {}
    for svs_prefix_id in index_splits[idr_torch.rank]:
        h5filename = all_h5_filenames[svs_prefix_id]
        svs_prefix = os.path.basename(h5filename).replace('.h5', '')
        with h5py.File(h5filename, 'r') as h5file: 
            arr[svs_prefix_id] = h5file['coords'][:].shape[0]
    with open(os.path.join(save_root, f'count{idr_torch.rank}.pkl'), 'wb') as fp:
        pickle.dump(arr, fp)

def generate_fake_dataset():
    import sys,os,glob,pickle,json,io,tarfile,time,gc
    import numpy as np
    from PIL import Image
    save_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'fake_dataset')
    os.makedirs(save_root, exist_ok=True)
    num_tars = 10
    count_dict = {}
    for i in range(num_tars):
        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')
        save_filepath = '{}/{:05d}.tar.gz'.format(save_root, i)
        newsize = (256, 256)
        num_imgs = np.random.randint(low=10, high=15)
        label = 'c{:04d}'.format(i)
        count_dict[label] = num_imgs
        for j in range(num_imgs):
            patch = Image.fromarray(np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8))
            im_buffer = io.BytesIO()
            patch.save(im_buffer, format='JPEG')
            info = tarfile.TarInfo(name="{}/{:05d}.jpg".format(label, j))
            info.size = im_buffer.getbuffer().nbytes
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp.addfile(info, im_buffer)
        
        tar_fp.close()
        with open(save_filepath, 'wb') as fp:
            fp.write(fh.getvalue())
        del fh
        gc.collect()
    with open(os.path.join(save_root, 'count_dict.pkl'), 'wb') as fp:
        pickle.dump(count_dict, fp)
    print('total number of images: ', sum(list(count_dict.values())))

def main_tar():

    image_ext = '.svs'
    svs_dir = '/data/zhongz2/tcga/TCGA-ALL2_256/svs'
    patches_dir = '/data/zhongz2/tcga/TCGA-ALL2_256/patches'
    save_root = '/data/zhongz2/tcga_tars/'
    num_sub_splits = 8
    if idr_torch.rank == 0:
        os.makedirs(save_root, exist_ok=True)

    local_temp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], str(idr_torch.rank), str(idr_torch.local_rank))
    os.makedirs(local_temp_dir, exist_ok=True)

    all_h5_filenames = sorted(glob.glob(patches_dir + '/*.h5'))
    with open(os.path.join(save_root, '__count__.pkl'), 'rb') as fp:
        count_dict = pickle.load(fp)
    splits, index_splits, split_sums = split_into_subsets_with_indices(count_dict, idr_torch.world_size)

    if idr_torch.rank == 0:
        with open(os.path.join(save_root, 'all_h5_filenames.pkl'), 'wb') as fp:
            pickle.dump({'svs_dir': svs_dir, 
                'patches_dir': patches_dir, 
                'all_h5_filenames': all_h5_filenames, 
                'splits': splits, 'split_sums': split_sums,
                'index_splits': index_splits}, fp)

    count_dict1 = {i: count for i, count in zip(index_splits[idr_torch.rank], splits[idr_torch.rank])}

    splits1, index_splits1, split_sums1 = split_into_subsets_with_indices(count_dict1, num_sub_splits)

    for ii in range(num_sub_splits):
            
        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')
        save_filepath = '{}/{:05d}.tar.gz'.format(save_root, num_sub_splits*idr_torch.rank+ii)
        newsize = (256, 256)
        for svs_prefix_id in index_splits1[ii]:
            h5filename = all_h5_filenames[svs_prefix_id]
            svs_prefix = os.path.basename(h5filename).replace('.h5', '')

            svs_filename = os.path.join(svs_dir, svs_prefix + image_ext)
            local_svs_filename = os.path.join(local_temp_dir, os.path.basename(svs_filename))
            if not os.path.exists(local_svs_filename):
                os.system(f'cp -RL "{svs_filename}" "{local_svs_filename}"')
            slide = openslide.open_slide(local_svs_filename)

            with h5py.File(h5filename, 'r') as h5file: 
                coords = h5file['coords'][:]
                patch_level = h5file['coords'].attrs['patch_level']
                patch_size = h5file['coords'].attrs['patch_size']

            for ind, coord in enumerate(coords):
                patch = slide.read_region(location=(int(coord[0]), int(coord[1])), 
                    level=patch_level, size=(patch_size, patch_size)).resize(newsize).convert('RGB')
                im_buffer = io.BytesIO()
                patch.save(im_buffer, format='JPEG')
                info = tarfile.TarInfo(
                    name="{}/x{}_y{}.jpg".format(svs_prefix_id, int(coord[0]), int(coord[1])))
                info.size = im_buffer.getbuffer().nbytes
                info.mtime = time.time()
                im_buffer.seek(0)
                tar_fp.addfile(info, im_buffer)

            slide.close()
            if os.path.exists(local_svs_filename):
                os.system(f'rm -rf "{local_svs_filename}"')
            
        tar_fp.close()
        with open(save_filepath, 'wb') as fp:
            fp.write(fh.getvalue())
        del fh
        gc.collect()

if __name__ == '__main__':
    # get_count()
    main_tar()














