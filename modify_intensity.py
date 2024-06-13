import nibabel as nib
import numpy as np

dataset_dir = 'C:/Users/nguc4116/Desktop/artery_reconstruction/dataset/'
original_file_path = dataset_dir + 'BCW-1205-RES_0000.nii.gz'
# segment_file_path = dataset_dir + 'sub-581_run-1_mra_eICAB_CW.nii.gz'
# original_file_path = dataset_dir + 'sub-581_run-1_mra_resampled.nii.gz'
# centerline_file_path = dataset_dir + 'sub-9_run-1_mra_CircleOfWillis_centerline.nii.gz'

original_image = nib.load(original_file_path)

pos = np.array([
    [267, 300, 126],
    [268, 300, 126],
    [269, 300, 126],
    [267, 298, 126],
    [268, 298, 126],
    [269, 298, 126],
    [267, 297, 126],
    [268, 297, 126],
    [269, 297, 126],
    [266, 297, 126],

])



value = [

]

original_data = original_image.get_fdata()

# Generate random values between 600 and 650
random_values = np.random.randint(1000, 1200, size=pos.shape[0])

# Change the intensity values in the original_data array
for idx, position in enumerate(pos):
    original_data[tuple(position)] = original_data[tuple(position)] + 50

new_image = nib.Nifti1Image(original_data, affine=original_image.affine, header=original_image.header)
nib.save(new_image, dataset_dir + 'BCW-1205-RES_0000_2.nii.gz')