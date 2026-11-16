# Source/pipeline_kmeans_morph.py
"""
K-means + morphology renal tissue segmentation (cortex / medulla / pelvis).
"""
from __future__ import print_function

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.morphology import convex_hull_image
from scipy.ndimage import (
    binary_fill_holes,
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
    label as cc_label,
    uniform_filter,
)

import dicom
from Source.segmentation import renal_segment_v4
from Source.dicom_load_tolerant import load_dicom_dataset


def drop_trailing_high_trigger_frames(recon, dicom_list, gap_factor=2.5):
    nx, ny, nz, nt = recon.shape
    n_files = len(dicom_list)
    triggers = []
    for t in range(nt):
        idx = t * nz
        if idx >= n_files:
            break
        ds = dicom.read_file(dicom_list[idx])
        if hasattr(ds, 'TriggerTime') and ds.TriggerTime is not None:
            tr = float(ds.TriggerTime)
        elif hasattr(ds, 'AcquisitionTime') and ds.AcquisitionTime is not None:
            at = str(ds.AcquisitionTime)
            hh, mm = float(at[0:2]), float(at[2:4])
            ss = float(at[4:]) if len(at) > 4 else 0.0
            tr = hh * 3600 + mm * 60 + ss
        else:
            tr = float(t)
        triggers.append(tr)

    triggers = np.asarray(triggers, dtype=np.float64)
    nt_eff = len(triggers)
    if nt_eff < nt:
        recon = recon[..., :nt_eff]
        nt = nt_eff

    print('Trigger times (per phase):', np.round(triggers, 2))
    if nt < 3:
        return recon

    dt = np.diff(triggers)
    dt_pos = dt[dt > 1e-6]
    med_dt = float(np.median(dt_pos)) if dt_pos.size else 0.0
    print('Median Δtrigger = %.3f' % med_dt)

    keep = np.ones(nt, dtype=bool)
    if med_dt > 0:
        for t in range(nt - 1, 0, -1):
            gap = triggers[t] - triggers[t - 1]
            if gap > gap_factor * med_dt:
                keep[t] = False
                print('Drop trailing t=%d (trigger=%.2f, gap=%.2f)' % (
                    t, triggers[t], gap))
            else:
                break

    n_drop = int((~keep).sum())
    if n_drop:
        recon = recon[..., keep]
        print('Dropped %d trailing frame(s); shape %s' % (n_drop, recon.shape))
    else:
        print('No trailing high-trigger frames dropped')
    return recon


def export_labels_dicom(final_labels, dicom_list, output_dir, series_description='KMeans_Morph_Segmentation'):
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    nz = final_labels.shape[2]
    templates = dicom_list[:nz]
    if len(templates) < nz:
        raise ValueError('dicom_list has fewer than nz templates')

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
        ds.PixelData = pixel_array.astype(np.uint16).tostring()
        ds.SmallestImagePixelValue = 0
        ds.LargestImagePixelValue = int(pixel_array.max())
        ds[0x0028, 0x0106].VR = 'US'
        ds[0x0028, 0x0107].VR = 'US'
        ds.save_as(os.path.join(output_dir, '%04d.dcm' % zz))
    print('Wrote %d label DICOMs to %s' % (nz, output_dir))


def _read_dicom_force(path):
    try:
        return dicom.read_file(path)
    except Exception:
        return dicom.read_file(path, force=True)


def build_affine_from_dicom(dicom_list, nz, spacing):
    """
    Affine for arrays shaped (rows, cols, slices) as produced by dicom2hdf5.

    DICOM:
      pixel_array axes: 0 = row index, 1 = column index
      ImageOrientationPatient:
        [0:3] = direction of increasing columns (along a row)
        [3:6] = direction of increasing rows (along a column)
      PixelSpacing: [row_spacing, col_spacing]

    In-plane columns of the affine are negated to match axial DICOM
    display in ITK-SNAP (180° correction vs raw IOP mapping).
    """
    templates = dicom_list[:nz]
    ds0 = _read_dicom_force(templates[0])

    dy = float(ds0.PixelSpacing[0])  # spacing between rows  (axis 0)
    dx = float(ds0.PixelSpacing[1])  # spacing between cols  (axis 1)
    dz = float(spacing[2]) if spacing[2] else float(getattr(ds0, 'SliceThickness', 1.0) or 1.0)

    F = np.diag([dy, dx, abs(float(dz))]).astype(np.float64)
    origin = np.zeros(3, dtype=np.float64)

    try:
        iop = [float(x) for x in ds0.ImageOrientationPatient]
        d_row = np.array(iop[0:3], dtype=np.float64)  # +column direction
        d_col = np.array(iop[3:6], dtype=np.float64)  # +row direction
        d_sl = np.cross(d_row, d_col)
        n = np.linalg.norm(d_sl)
        if n > 1e-8:
            d_sl = d_sl / n

        origin = np.array(
            [float(x) for x in ds0.ImagePositionPatient], dtype=np.float64)

        if nz >= 2:
            dsN = _read_dicom_force(templates[nz - 1])
            ippN = np.array(
                [float(x) for x in dsN.ImagePositionPatient], dtype=np.float64)
            delta = ippN - origin
            dz_signed = float(np.dot(delta, d_sl) / float(nz - 1))
            if abs(dz_signed) > 1e-6:
                dz = dz_signed

        # index (i, j, k) = (row, col, slice)
        F[:, 0] = d_col * dy
        F[:, 1] = d_row * dx
        F[:, 2] = d_sl * float(dz)

        # 180° in-plane vs original DICOM axial view
        F[:, 0] *= -1.0
        F[:, 1] *= -1.0

    except Exception as e:
        print('Warning: incomplete DICOM orientation (%s); diagonal spacing' % e)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = F
    affine[:3, 3] = origin
    return affine


def write_nifti(array, affine, out_path):
    """
    Minimal NIfTI-1 (.nii) writer. array axes = (rows, cols, slices[, time]).
    No rot90 — orientation is entirely in the affine from DICOM.
    """
    data = np.ascontiguousarray(array)
    shape = data.shape
    ndim = data.ndim

    header = bytearray(348)
    header[0:4] = np.int32(348).tobytes()

    dim = np.zeros(8, dtype=np.int16)
    dim[0] = ndim
    for i in range(ndim):
        dim[i + 1] = shape[i]
    header[40:56] = dim.tobytes()

    if data.dtype == np.uint8:
        datatype, bitpix = 2, 8
    elif data.dtype == np.int16:
        datatype, bitpix = 4, 16
    elif data.dtype == np.uint16:
        datatype, bitpix = 512, 16  # DT_UINT16
    elif data.dtype == np.float32:
        datatype, bitpix = 16, 32
    else:
        data = data.astype(np.float32)
        datatype, bitpix = 16, 32

    header[70:72] = np.int16(datatype).tobytes()
    header[72:74] = np.int16(bitpix).tobytes()

    # pixdim from affine column norms
    pixdim = np.zeros(8, dtype=np.float32)
    pixdim[0] = 1.0  # qfac
    for i in range(min(3, ndim)):
        pixdim[i + 1] = float(np.linalg.norm(affine[:3, i]))
    if ndim >= 4:
        pixdim[4] = 1.0
    header[76:108] = pixdim.tobytes()

    header[108:112] = np.float32(352.0).tobytes()  # vox_offset

    # sform: affine method 2
    header[254:256] = np.int16(2).tobytes()  # sform_code = NIFTI_XFORM_SCANNER_ANAT
    header[280:284] = np.float32(affine[0, 0]).tobytes()
    header[284:288] = np.float32(affine[0, 1]).tobytes()
    header[288:292] = np.float32(affine[0, 2]).tobytes()
    header[292:296] = np.float32(affine[0, 3]).tobytes()
    header[296:300] = np.float32(affine[1, 0]).tobytes()
    header[300:304] = np.float32(affine[1, 1]).tobytes()
    header[304:308] = np.float32(affine[1, 2]).tobytes()
    header[308:312] = np.float32(affine[1, 3]).tobytes()
    header[312:316] = np.float32(affine[2, 0]).tobytes()
    header[316:320] = np.float32(affine[2, 1]).tobytes()
    header[320:324] = np.float32(affine[2, 2]).tobytes()
    header[324:328] = np.float32(affine[2, 3]).tobytes()

    header[344:348] = b'n+1\0'

    # NIfTI expects voxel data in Fortran order relative to dim[1],dim[2],...
    with open(out_path, 'wb') as f:
        f.write(header)
        f.write(b'\0' * 4)
        f.write(np.asfortranarray(data).tobytes(order='F'))
    print('Wrote', out_path, 'shape', shape)


def export_nifti_image_and_labels(recon, final_labels, dicom_list, spacing,
                                  output_dir, time_index=None, temp_res=None):
    """
    Export:
      image.nii  – one 3D time frame (default ~ mid / ~90s)
      labels.nii – segmentation, same grid + same affine
    No rot90; orientation from DICOM IOP/IPP.
    """
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    nx, ny, nz, nt = recon.shape
    assert final_labels.shape == (nx, ny, nz)

    if time_index is None:
        if temp_res is not None and temp_res > 0:
            time_index = int(min(max(round(90.0 / float(temp_res)), 0), nt - 1))
        else:
            time_index = min(nt // 2, nt - 1)

    affine = build_affine_from_dicom(dicom_list, nz, spacing)

    # Image: scale to uint16 for viewing
    vol = recon[:, :, :, time_index].astype(np.float64)
    vol = vol - vol.min()
    if vol.max() > 0:
        vol = vol / vol.max() * 4095.0
    img_u16 = np.round(vol).astype(np.uint16)

    labels_u8 = final_labels.astype(np.uint8)

    write_nifti(img_u16, affine, os.path.join(output_dir, 'image.nii'))
    write_nifti(labels_u8, affine, os.path.join(output_dir, 'labels.nii'))
    print('NIfTI export: time_index=%d / %d, affine origin=%s' % (
        time_index, nt, np.round(affine[:3, 3], 2)))


def run_kmeans_morph_pipeline(
        input_dir,
        output_dir=None,
        threshold_multiplier=1,
        root_path=None,
        gap_factor=2.5,
        run_moco=False,
        moco_ref_time_s=10.0,
        moco_maxiter=40,
        run_second_grabcut=False,
):
    """
    Returns
    -------
    recon : ndarray (nx, ny, nz, nt)
    final_labels : ndarray (nx, ny, nz) uint8
        0 bg, 2/3 L/R cortex, 4/5 L/R medulla, 6/7 L/R pelvis
    """
    if root_path is None:
        source_path = os.path.dirname(os.path.realpath(__file__))
        root_path = os.path.abspath(os.path.join(source_path, '..'))

    # ---- load ----
    dataset = load_dicom_dataset(input_dir, quarantine=True, always_scan=True)
    recon = dataset['recon']
    temp_res = dataset['temp_res'] / 1000.0
    spacing = np.array(dataset['spacing'], dtype=float)
    dicom_list = dataset['dicom_list']
    print('Loaded shape:', recon.shape, 'temp_res(s):', temp_res)

    # ---- trailing high triggers ----
    recon = drop_trailing_high_trigger_frames(recon, dicom_list, gap_factor=gap_factor)

    # ---- GrabCut ----
    print('Running GrabCut...')
    gc_labels = renal_segment_v4(
        recon, temp_res=temp_res, spacing=spacing,
        root_path=root_path, threshold_multiplier=threshold_multiplier,
    )
    print('GrabCut labels:', np.unique(gc_labels))

    # ---- optional MoCo ----
    if run_moco:
        from Source.motion_correction import motion_correct_affine_per_kidney
        print('Running MoCo ref_time_s=%.1f' % moco_ref_time_s)
        recon_mc, t_ref, _ = motion_correct_affine_per_kidney(
            recon, gc_labels, temp_res,
            ref_time_s=moco_ref_time_s, dilate_vox=3, maxiter=moco_maxiter,
        )
        recon = recon_mc
        print('MoCo done t_ref=%d (%.1fs)' % (t_ref, t_ref * temp_res))
        if run_second_grabcut:
            print('Second GrabCut...')
            gc_labels = renal_segment_v4(
                recon, temp_res=temp_res, spacing=spacing,
                root_path=root_path, threshold_multiplier=threshold_multiplier,
            )

    kid_ids = [k for k in np.unique(gc_labels) if k > 0]

    # ---- hole-fill ----
    kidney_raw = gc_labels > 0
    kidney_filled = np.zeros_like(kidney_raw)
    for z in range(kidney_raw.shape[2]):
        kidney_filled[:, :, z] = binary_fill_holes(kidney_raw[:, :, z])
    idx = distance_transform_edt(~kidney_raw, return_distances=False, return_indices=True)
    gc_filled = gc_labels[idx[0], idx[1], idx[2]]
    gc_filled[~kidney_filled] = 0
    kidney = gc_filled > 0
    gc_original = gc_filled > 0

    # ---- PCA + K-means ----
    print('PCA + K-means...')
    recon_diff = recon - np.amin(recon, axis=3, keepdims=True)
    sig = recon_diff.reshape(-1, recon.shape[-1]).astype(np.float32)
    pca = PCA(n_components=3)
    coeffs = pca.fit_transform(sig)
    pca_map = coeffs.reshape(recon.shape[:3] + (3,))
    for ch in range(3):
        c = pca_map[..., ch]
        c -= c.min()
        c /= (c.max() + 1e-8)
        pca_map[..., ch] = c

    pts = pca_map[kidney]
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    cluster_id = kmeans.fit_predict(pts)
    cluster_vol = np.zeros(recon.shape[:3], dtype=np.int8)
    cluster_vol[kidney] = cluster_id + 1

    # ---- cluster → tissue ----
    time = np.arange(recon.shape[3]) * temp_res
    nt = recon.shape[3]
    sizes, curves, peaks = {}, {}, {}
    for cid in (1, 2, 3):
        m = cluster_vol == cid
        sizes[cid] = int(m.sum())
        if sizes[cid] == 0:
            curves[cid] = None
            peaks[cid] = np.inf
            continue
        curve = recon[m].mean(axis=0)
        curves[cid] = curve
        peaks[cid] = time[int(np.argmax(curve))]
        print('Cluster %d: n=%d peak=%.1fs' % (cid, sizes[cid], peaks[cid]))

    cortex_id = max(sizes, key=sizes.get)
    t_ctx = peaks[cortex_id]
    i_ctx = int(np.argmin(np.abs(time - t_ctx)))
    win = max(1, int(round(10.0 / temp_res)))
    print('Cortex (largest)=%d' % cortex_id)

    remaining = [c for c in (1, 2, 3) if c != cortex_id]
    rise = {}
    for cid in remaining:
        c = curves[cid]
        late = c[int(0.67 * len(c)):].mean()
        at_ctx = c[i_ctx:i_ctx + win].mean()
        rise[cid] = float(late - at_ctx)
        print('Cluster %d post-cortical rise=%.1f' % (cid, rise[cid]))
    medulla_id = max(rise, key=rise.get)
    pelvis_id = [c for c in remaining if c != medulla_id][0]
    print('Assignment cortex=%d medulla=%d pelvis=%d' % (
        cortex_id, medulla_id, pelvis_id))

    i_early_end = max(i_ctx + win, int(round(40.0 / temp_res)))
    late_vol = recon[..., int(0.67 * nt):].mean(axis=3)
    at_ctx_vol = recon[..., i_ctx:i_ctx + win].mean(axis=3)
    rise_map = late_vol - at_ctx_vol
    early_peak = recon[..., :i_early_end].max(axis=3)
    baseline = recon[..., :max(1, int(round(5.0 / temp_res)))].mean(axis=3)
    early_enh = early_peak - baseline

    med_cluster = cluster_vol == medulla_id
    if med_cluster.any():
        rise_thr = np.percentile(rise_map[med_cluster], 40)
        ctx_cluster = cluster_vol == cortex_id
        early_thr = (np.median(early_enh[ctx_cluster]) if ctx_cluster.any()
                     else np.median(early_enh[kidney]))
        med_tight = med_cluster & (rise_map >= rise_thr) & (early_enh < early_thr)
        med_dropped = med_cluster & ~med_tight
    else:
        med_tight = med_cluster
        med_dropped = np.zeros_like(med_cluster)

    cluster_vol_refined = cluster_vol.copy()
    cluster_vol_refined[med_dropped] = 0

    # ---- morphology labels ----
    final_labels = np.zeros(recon.shape[:3], dtype=np.uint8)
    voxel_vol_mm3 = float(np.prod(spacing))

    for kid_id in kid_ids:
        mask = gc_filled == kid_id
        if kid_id == 1:
            final_labels[mask & (cluster_vol_refined == cortex_id)] = 2
            final_labels[mask & med_tight] = 4
            final_labels[mask & (cluster_vol_refined == pelvis_id)] = 6
        else:
            final_labels[mask & (cluster_vol_refined == cortex_id)] = 3
            final_labels[mask & med_tight] = 5
            final_labels[mask & (cluster_vol_refined == pelvis_id)] = 7

    final_labels[med_dropped & gc_original & (final_labels == 0)] = 1

    gc_temp_id = np.zeros_like(gc_filled)
    for kid_id in kid_ids:
        m = gc_filled == kid_id
        hull = np.zeros_like(m)
        for z in range(m.shape[2]):
            sl = m[:, :, z]
            if sl.sum() >= 20:
                hull[:, :, z] = convex_hull_image(sl)
        all_indents = hull & ~m
        labeled, n = cc_label(all_indents)
        if n == 0:
            temp_k = m.copy()
        else:
            sz = np.bincount(labeled.ravel())
            sz[0] = 0
            temp_k = m | (labeled == sz.argmax())
        gc_temp_id[temp_k & (gc_temp_id == 0)] = kid_id

    gc_temp = gc_temp_id > 0
    indent = gc_temp & ~gc_original

    mean_spacing = float(np.mean(spacing))
    gc_temp_cropped = np.zeros_like(gc_temp)
    for kid_id in kid_ids:
        m_temp = gc_temp_id == kid_id
        n_vox = int(m_temp.sum())
        if n_vox == 0:
            continue
        vol_mm3 = n_vox * voxel_vol_mm3
        radius_mm = (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        crop_mm = float(np.clip(0.15 * radius_mm, 3.0, 8.0))
        crop_vox = max(1, int(round(crop_mm / mean_spacing)))
        m = m_temp.copy()
        for _ in range(crop_vox):
            m = binary_erosion(m)
        gc_temp_cropped |= m

    brim = gc_original & ~gc_temp_cropped
    for kid_id in kid_ids:
        final_labels[brim & (gc_filled == kid_id)] = 2 if kid_id == 1 else 3
    final_labels[brim & ((final_labels == 6) | (final_labels == 7))] = 0
    for kid_id in kid_ids:
        m = brim & (gc_filled == kid_id) & ((final_labels == 0) | (final_labels == 1))
        final_labels[m] = 2 if kid_id == 1 else 3

    final_labels[gc_original & (final_labels == 0)] = 1

    final_labels[indent] = 0
    for kid_id in kid_ids:
        final_labels[indent & (gc_temp_id == kid_id)] = 6 if kid_id == 1 else 7
    still = indent & ((final_labels == 0) | (final_labels == 1))
    if still.any():
        idx = distance_transform_edt(~gc_original, return_distances=False, return_indices=True)
        nearest = gc_filled[idx[0], idx[1], idx[2]]
        for kid_id in kid_ids:
            final_labels[still & (nearest == kid_id)] = 6 if kid_id == 1 else 7

    for kid_id in kid_ids:
        ctx_val = 2 if kid_id == 1 else 3
        med_val = 4 if kid_id == 1 else 5
        pel_val = 6 if kid_id == 1 else 7
        m = final_labels == ctx_val
        if m.any():
            labeled, n = cc_label(m)
            if n > 1:
                sz = np.bincount(labeled.ravel()); sz[0] = 0
                keep = labeled == sz.argmax()
                final_labels[m & ~keep] = med_val
        m = final_labels == pel_val
        if m.any():
            labeled, n = cc_label(m)
            if n > 1:
                sz = np.bincount(labeled.ravel()); sz[0] = 0
                keep = labeled == sz.argmax()
                final_labels[m & ~keep] = ctx_val

    pelvis_mask = (final_labels == 6) | (final_labels == 7)
    cortex_mask = (final_labels == 2) | (final_labels == 3)
    pelvis_frac = uniform_filter(pelvis_mask.astype(np.float32), size=3)
    final_labels[cortex_mask & (pelvis_frac >= 0.45)] = 1
    final_labels[final_labels == 1] = 0

    print('Final counts cortex/medulla/pelvis:',
          int(((final_labels == 2) | (final_labels == 3)).sum()),
          int(((final_labels == 4) | (final_labels == 5)).sum()),
          int(((final_labels == 6) | (final_labels == 7)).sum()))

    if output_dir is not None:
        export_labels_dicom(final_labels, dicom_list, output_dir)
        np.save(os.path.join(output_dir, 'final_labels.npy'), final_labels)
        np.save(os.path.join(output_dir, 'recon_shape.npy'), np.array(recon.shape))

        export_nifti_image_and_labels(
            recon, final_labels, dicom_list, spacing, output_dir,
            time_index=None, temp_res=temp_res,
        )
        print('Saved DICOM labels + NIfTI image/labels to', output_dir)

    return recon, final_labels