import os
import errno
import dicom
import numpy as np

def export_labels_to_dicom(final_labels, dicom_list, output_dir, series_description='Renal_Segmentation'):
    """
    final_labels: (nx, ny, nz) uint8, same orientation as dicom2hdf5 recon[:,:,z]
    dicom_list:   sorted list from dataset['dicom_list']
                  first nz entries = one full spatial volume (first time point)
    """
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    nz = final_labels.shape[2]
    # First time-point's slices (dicom_list sorted by trigger, then slice location)
    templates = dicom_list[:nz]
    if len(templates) < nz:
        raise ValueError('dicom_list has fewer than nz=%d files for one volume' % nz)

    series_uid = dicom.UID.generate_uid()

    for zz in range(nz):
        img = final_labels[:, :, zz]
        pixel_array = img.astype(np.uint16) * 50 + 250
        pixel_array[img == 0] = 0

        ds = dicom.read_file(templates[zz])
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDescription = series_description
        ds.InstanceNumber = zz + 1
        ds.SOPInstanceUID = dicom.UID.generate_uid()

        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'

        if pixel_array.dtype != np.uint16:
            pixel_array = pixel_array.astype(np.uint16)
        ds.PixelData = pixel_array.tostring()

        ds.SmallestImagePixelValue = 0
        ds.LargestImagePixelValue = int(pixel_array.max())
        ds[0x0028, 0x0106].VR = 'US'
        ds[0x0028, 0x0107].VR = 'US'

        out_path = os.path.join(output_dir, '%04d.dcm' % zz)
        ds.save_as(out_path)

    print('Wrote %d DICOM label slices to %s' % (nz, output_dir))