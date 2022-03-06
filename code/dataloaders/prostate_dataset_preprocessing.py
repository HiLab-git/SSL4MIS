# save images in slice level
import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())

# slice_num = 0
# mask_path = sorted(
#     glob.glob("/home/SENSETIME/luoxiangde.vendor/Desktop/SSL4MIS_5Fold/data/prostate_zonal_nii/*_lab.nii.gz"))
# for image_path in mask_path:
#     image_itk = sitk.ReadImage(image_path.replace("_lab", ""))
#     image = sitk.GetArrayFromImage(image_itk)

#     image = MedicalImageDeal(image, percent=0.99).valid_img
#     image = (image - image.min()) / (image.max() - image.min())
#     norm_img_itk = sitk.GetImageFromArray(image)
#     norm_img_itk.CopyInformation(image_itk)
#     sitk.WriteImage(norm_img_itk, image_path.replace("_lab", ""))


# saving images in slice level

slice_num = 0
mask_path = sorted(
    glob.glob("/home/SENSETIME/luoxiangde.vendor/Desktop/SSL4MIS_5Fold/data/CHAOS_NII/label/*.nii.gz"))
for case in mask_path:
    label_itk = sitk.ReadImage(case)
    label = sitk.GetArrayFromImage(label_itk)

    image_path = case.replace("/label/", "/image/")
    image_itk = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_itk)
    spacing = image_itk.GetSpacing()

    image = MedicalImageDeal(image, percent=0.99).valid_img
    image = (image - image.min()) / (image.max() - image.min())
    print(image.shape)
    image = image.astype(np.float32)
    item = case.split("/")[-1].split(".")[0].replace("_gt", "")
    if image.shape != label.shape:
        print("Error")
    print(item)
    
    f = h5py.File(
        '/home/SENSETIME/luoxiangde.vendor/Desktop/SSL4MIS_5Fold/data/CHAOS/all_volumes/{}.h5'.format(item), 'w')
    f.create_dataset(
        'image', data=image, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.create_dataset('spacing', data=np.array(spacing), compression="gzip")
    f.close()
print("Converted all ACDC volumes to 2D slices")
print("Total {} slices".format(slice_num))
# # saving images in volume level

# slice_num = 0
# mask_path = sorted(
#     glob.glob("/home/SENSETIME/luoxiangde.vendor/Desktop/SSL4MIS_5Fold/data/prostate_zonal_nii/*_lab.nii.gz"))
# for case in mask_path:
#     label_itk = sitk.ReadImage(case)
#     label = sitk.GetArrayFromImage(label_itk)

#     image_path = case.replace("_lab", "")
#     image_itk = sitk.ReadImage(image_path)
#     image = sitk.GetArrayFromImage(image_itk)
#     spacing = image_itk.GetSpacing()

#     image = image.astype(np.float32)
#     item = case.split("/")[-1].split(".")[0].replace("_lab", "")
#     if image.shape != label.shape:
#         print("Error")
#     print(item)
    
#     f = h5py.File(
#         '/home/SENSETIME/luoxiangde.vendor/Desktop/SSL4MIS_5Fold/data/ProstateX/all_volumes/{}.h5'.format(item), 'w')
#     f.create_dataset(
#         'image', data=image, compression="gzip")
#     f.create_dataset('label', data=label, compression="gzip")
#     f.create_dataset('spacing', data=np.array(spacing), compression="gzip")
#     f.close()
# print("Converted all Prostate volumes to 2D slices")
# print("Total {} slices".format(slice_num))