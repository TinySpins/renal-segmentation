# Source/pipeline_kmeans_morph.py
"""
K-means + morphology renal tissue segmentation.

Hilum cleanup OFF unless hilum_open_fraction > 0.
Opening only identifies removal; subtracted from hole-filled GrabCut.

Outputs:
  python/, nifti/, dicom/{images,whole_kidney,labels}/, qa_report.pdf
"""
from __future__ import print_function

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.morphology import convex_hull_image, disk, binary_opening
from scipy.ndimage import (
    binary_fill_holes,
    binary_erosion,
    distance_transform_edt,
    label as cc_label,
    uniform_filter,
)

import dicom
from Source.segmentation import renal_segment_v4
from Source.dicom_load_tolerant import load_dicom_dataset
from Source.report_pdf import write_qa_pdf


# =============================================================================
# Logging
# =============================================================================

class RunLog(object):
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        print(msg)
        self.lines.append(str(msg))


# =============================================================================
# I/O helpers
# =============================================================================

def _ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def _read_dicom_force(path):
    try:
        return dicom.read_file(path)
    except Exception:
        return dicom.read_file(path, force=True)


def drop_trailing_high_trigger_frames(recon, dicom_list, gap_factor=2.5, log=None):
    log = log or (lambda m: print(m))
    nx, ny, nz, nt = recon.shape
    n_files = len(dicom_list)
    triggers = []
    for t in range(nt):
        idx = t * nz
        if idx >= n_files:
            break
        ds = _read_dicom_force(dicom_list[idx])
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

    log('Trigger times (per phase): %s' % (np.round(triggers, 2),))
    if nt < 3:
        return recon

    dt = np.diff(triggers)
    dt_pos = dt[dt > 1e-6]
    med_dt = float(np.median(dt_pos)) if dt_pos.size else 0.0
    log('Median Δtrigger = %.3f' % med_dt)

    keep = np.ones(nt, dtype=bool)
    if med_dt > 0:
        for t in range(nt - 1, 0, -1):
            gap = triggers[t] - triggers[t - 1]
            if gap > gap_factor * med_dt:
                keep[t] = False
                log('Drop trailing t=%d (trigger=%.2f, gap=%.2f)' % (
                    t, triggers[t], gap))
            else:
                break

    n_drop = int((~keep).sum())
    if n_drop:
        recon = recon[..., keep]
        log('Dropped %d trailing frame(s); shape %s' % (n_drop, recon.shape))
    else:
        log('No trailing high-trigger frames dropped')
    return recon


def build_affine_from_dicom(dicom_list, nz, spacing):
    templates = dicom_list[:nz]
    ds0 = _read_dicom_force(templates[0])
    dy = float(ds0.PixelSpacing[0])
    dx = float(ds0.PixelSpacing[1])
    dz = float(spacing[2]) if spacing[2] else float(getattr(ds0, 'SliceThickness', 1.0) or 1.0)
    F = np.diag([dy, dx, abs(float(dz))]).astype(np.float64)
    origin = np.zeros(3, dtype=np.float64)
    try:
        iop = [float(x) for x in ds0.ImageOrientationPatient]
        d_row = np.array(iop[0:3], dtype=np.float64)
        d_col = np.array(iop[3:6], dtype=np.float64)
        d_sl = np.cross(d_row, d_col)
        n = np.linalg.norm(d_sl)
        if n > 1e-8:
            d_sl = d_sl / n
        origin = np.array([float(x) for x in ds0.ImagePositionPatient], dtype=np.float64)
        if nz >= 2:
            dsN = _read_dicom_force(templates[nz - 1])
            ippN = np.array([float(x) for x in dsN.ImagePositionPatient], dtype=np.float64)
            delta = ippN - origin
            dz_signed = float(np.dot(delta, d_sl) / float(nz - 1))
            if abs(dz_signed) > 1e-6:
                dz = dz_signed
        F[:, 0] = d_col * dy
        F[:, 1] = d_row * dx
        F[:, 2] = d_sl * float(dz)
        F[:, 0] *= -1.0
        F[:, 1] *= -1.0
    except Exception as e:
        print('Warning: incomplete DICOM orientation (%s); diagonal spacing' % e)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = F
    affine[:3, 3] = origin
    return affine


def write_nifti(array, affine, out_path):
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
        datatype, bitpix = 512, 16
    elif data.dtype == np.float32:
        datatype, bitpix = 16, 32
    else:
        data = data.astype(np.float32)
        datatype, bitpix = 16, 32
    header[70:72] = np.int16(datatype).tobytes()
    header[72:74] = np.int16(bitpix).tobytes()
    pixdim = np.zeros(8, dtype=np.float32)
    pixdim[0] = 1.0
    for i in range(min(3, ndim)):
        pixdim[i + 1] = float(np.linalg.norm(affine[:3, i]))
    if ndim >= 4:
        pixdim[4] = 1.0
    header[76:108] = pixdim.tobytes()
    header[108:112] = np.float32(352.0).tobytes()
    header[254:256] = np.int16(2).tobytes()
    for r in range(3):
        for c in range(4):
            off = 280 + (r * 4 + c) * 4
            header[off:off + 4] = np.float32(affine[r, c]).tobytes()
    header[344:348] = b'n+1\0'
    with open(out_path, 'wb') as f:
        f.write(header)
        f.write(b'\0' * 4)
        f.write(np.asfortranarray(data).tobytes(order='F'))
    print('Wrote', out_path, 'shape', shape)


def export_volume_dicom(vol3d, dicom_list, output_dir, series_description,
                        scale_mode='labels'):
    _ensure_dir(output_dir)
    nz = vol3d.shape[2]
    templates = dicom_list[:nz]
    if len(templates) < nz:
        raise ValueError('dicom_list shorter than nz')
    series_uid = dicom.UID.generate_uid()
    for zz in range(nz):
        sl = vol3d[:, :, zz]
        if scale_mode == 'labels':
            pix = sl.astype(np.uint16) * 50 + 250
            pix[sl == 0] = 0
        elif scale_mode == 'mask':
            pix = (sl > 0).astype(np.uint16) * 1000
        else:
            v = sl.astype(np.float64)
            v = v - v.min()
            if v.max() > 0:
                v = v / v.max() * 4095.0
            pix = np.round(v).astype(np.uint16)
        ds = _read_dicom_force(templates[zz])
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDescription = series_description
        ds.InstanceNumber = zz + 1
        ds.SOPInstanceUID = dicom.UID.generate_uid()
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.PixelData = pix.astype(np.uint16).tostring()
        ds.SmallestImagePixelValue = 0
        ds.LargestImagePixelValue = int(pix.max()) if pix.size else 0
        ds[0x0028, 0x0106].VR = 'US'
        ds[0x0028, 0x0107].VR = 'US'
        ds.save_as(os.path.join(output_dir, '%04d.dcm' % zz))
    print('DICOM series ->', output_dir, '(%d slices)' % nz)


def export_all_outputs(recon, final_labels, whole_kidney, dicom_list, spacing,
                       output_dir, temp_res=None, time_index=None):
    nx, ny, nz, nt = recon.shape
    whole_kidney = (whole_kidney > 0).astype(np.uint8)
    final_labels = final_labels.astype(np.uint8)
    py_dir = os.path.join(output_dir, 'python')
    nii_dir = os.path.join(output_dir, 'nifti')
    dcm_img = os.path.join(output_dir, 'dicom', 'images')
    dcm_kid = os.path.join(output_dir, 'dicom', 'whole_kidney')
    dcm_lab = os.path.join(output_dir, 'dicom', 'labels')
    for d in (py_dir, nii_dir, dcm_img, dcm_kid, dcm_lab):
        _ensure_dir(d)
    np.save(os.path.join(py_dir, 'final_labels.npy'), final_labels)
    np.save(os.path.join(py_dir, 'whole_kidney.npy'), whole_kidney)
    np.save(os.path.join(py_dir, 'recon_shape.npy'), np.array(recon.shape))
    print('NumPy ->', py_dir)
    if time_index is None:
        if temp_res is not None and temp_res > 0:
            time_index = int(min(max(round(90.0 / float(temp_res)), 0), nt - 1))
        else:
            time_index = min(nt // 2, nt - 1)
    affine = build_affine_from_dicom(dicom_list, nz, spacing)
    vol = recon[:, :, :, time_index].astype(np.float64)
    vol = vol - vol.min()
    if vol.max() > 0:
        vol = vol / vol.max() * 4095.0
    img_u16 = np.round(vol).astype(np.uint16)
    write_nifti(img_u16, affine, os.path.join(nii_dir, 'image.nii'))
    write_nifti(final_labels, affine, os.path.join(nii_dir, 'labels.nii'))
    write_nifti(whole_kidney, affine, os.path.join(nii_dir, 'whole_kidney.nii'))
    print('NIfTI ->', nii_dir, '(t=%d)' % time_index)
    export_volume_dicom(img_u16, dicom_list, dcm_img, 'MR_image', 'image')
    export_volume_dicom(whole_kidney, dicom_list, dcm_kid, 'Whole_Kidney', 'mask')
    export_volume_dicom(final_labels, dicom_list, dcm_lab, 'Tissue_Labels', 'labels')
    print('DICOM ->', os.path.join(output_dir, 'dicom'))


# =============================================================================
# Hole-fill / hilum
# =============================================================================

def hole_fill_grabcut(gc_labels):
    kidney_raw = gc_labels > 0
    kidney_filled = np.zeros_like(kidney_raw)
    for z in range(kidney_raw.shape[2]):
        kidney_filled[:, :, z] = binary_fill_holes(kidney_raw[:, :, z])
    idx = distance_transform_edt(
        ~kidney_raw, return_distances=False, return_indices=True)
    gc_filled = gc_labels[idx[0], idx[1], idx[2]]
    gc_filled[~kidney_filled] = 0
    return gc_filled


def kidney_minor_axis_mm(mask3d, spacing):
    sx, sy = float(spacing[0]), float(spacing[1])
    widths = []
    for z in range(mask3d.shape[2]):
        sl = mask3d[:, :, z]
        if sl.sum() < 50:
            continue
        ys, xs = np.where(sl)
        pts = np.column_stack([xs * sx, ys * sy]).astype(np.float64)
        pts -= pts.mean(axis=0)
        if pts.shape[0] < 10:
            continue
        cov = np.cov(pts, rowvar=False)
        eig = np.linalg.eigvalsh(cov)
        eig = np.maximum(eig, 0.0)
        widths.append(2.0 * np.sqrt(eig[0]))
    if not widths:
        return None
    return float(np.median(widths))


def opening_radius_vox(minor_mm, spacing, fraction):
    sx, sy = float(spacing[0]), float(spacing[1])
    pix = 0.5 * (sx + sy)
    r_mm = float(fraction * minor_mm)
    r_vox = max(1, int(round(r_mm / max(pix, 1e-6))))
    return r_vox, r_mm


def subtract_hilum_protrusions(gc_filled, spacing, fraction=None, si_band=0.40, log=None):
    """
    Reference = hole-filled GrabCut.
    Opening only identifies candidates; removal = CCs touching mid SI band.
    Result = G0 & ~removal.
    """
    log = log or (lambda m: print(m))
    if fraction is None or float(fraction) <= 0:
        log('Hilum cleanup: disabled (pass --hilum-open-fraction to enable)')
        return gc_filled, np.zeros(gc_filled.shape, dtype=bool)

    fraction = float(fraction)
    si_band = min(max(float(si_band), 0.05), 1.0)
    log('Hilum cleanup: ON  fraction=%.3f  si_band=%.2f' % (fraction, si_band))
    log('  (opening identifies removal; subtracted from hole-filled GrabCut)')

    out = np.zeros_like(gc_filled)
    removal_all = np.zeros(gc_filled.shape, dtype=bool)

    for kid_id in [k for k in np.unique(gc_filled) if k > 0]:
        G0 = gc_filled == kid_id
        if not G0.any():
            continue

        minor = kidney_minor_axis_mm(G0, spacing)
        if minor is None:
            sx, sy = float(spacing[0]), float(spacing[1])
            pix = 0.5 * (sx + sy)
            r_mm, r_vox = 3.0, max(1, int(round(3.0 / max(pix, 1e-6))))
            log('Kidney %d: minor axis unknown → fallback r=%.2f mm (%d vox)' % (
                kid_id, r_mm, r_vox))
        else:
            r_vox, r_mm = opening_radius_vox(minor, spacing, fraction)
            log('Kidney %d: minor≈%.2f mm, fraction=%.3f → open r=%.2f mm (%d vox)' % (
                kid_id, minor, fraction, r_mm, r_vox))

        selem = disk(r_vox)
        opened = np.zeros_like(G0)
        for z in range(G0.shape[2]):
            sl = G0[:, :, z]
            if sl.any():
                opened[:, :, z] = binary_opening(sl, selem)

        candidates = G0 & ~opened
        zs = np.where(np.any(G0, axis=(0, 1)))[0]
        if zs.size == 0:
            out[G0] = kid_id
            continue
        z0, z1 = int(zs.min()), int(zs.max())
        zlen = z1 - z0 + 1
        margin = 0.5 * (1.0 - si_band) * zlen
        z_lo = int(np.floor(z0 + margin))
        z_hi = int(np.ceil(z1 - margin))
        if z_hi < z_lo:
            z_lo, z_hi = z0, z1
        band = np.zeros_like(G0)
        band[:, :, z_lo:z_hi + 1] = True
        log('Kidney %d: SI z=[%d,%d] band z=[%d,%d] (si_band=%.2f)' % (
            kid_id, z0, z1, z_lo, z_hi, si_band))

        removal = np.zeros_like(G0)
        if candidates.any():
            labeled, nlab = cc_label(candidates)
            n_drop, n_skip = 0, 0
            for i in range(1, nlab + 1):
                comp = labeled == i
                if np.any(comp & band):
                    removal[comp] = True
                    n_drop += 1
                else:
                    n_skip += 1
            log('Kidney %d: candidate CCs remove=%d keep=%d (vox removed=%d)' % (
                kid_id, n_drop, n_skip, int(removal.sum())))

        removal_all |= removal
        out[G0 & ~removal] = kid_id

    return out, removal_all


# =============================================================================
# Main
# =============================================================================

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
        hilum_open_fraction=None,
        hilum_si_band=0.40,
):
    log = RunLog()
    if root_path is None:
        source_path = os.path.dirname(os.path.realpath(__file__))
        root_path = os.path.abspath(os.path.join(source_path, '..'))

    dataset = load_dicom_dataset(input_dir, quarantine=True, always_scan=True)
    recon = dataset['recon']
    temp_res = dataset['temp_res'] / 1000.0
    spacing = np.array(dataset['spacing'], dtype=float)
    dicom_list = dataset['dicom_list']
    log('Input: %s' % input_dir)
    log('Loaded shape: %s  temp_res(s): %s  spacing: %s' % (
        recon.shape, temp_res, spacing))

    recon = drop_trailing_high_trigger_frames(
        recon, dicom_list, gap_factor=gap_factor, log=log)

    log('Running GrabCut...')
    gc_labels = renal_segment_v4(
        recon, temp_res=temp_res, spacing=spacing,
        root_path=root_path, threshold_multiplier=threshold_multiplier,
    )
    log('GrabCut labels: %s' % (np.unique(gc_labels),))

    if run_moco:
        from Source.motion_correction import motion_correct_affine_per_kidney
        log('Running MoCo ref_time_s=%.1f' % moco_ref_time_s)
        recon_mc, t_ref, _ = motion_correct_affine_per_kidney(
            recon, gc_labels, temp_res,
            ref_time_s=moco_ref_time_s, dilate_vox=3, maxiter=moco_maxiter,
        )
        recon = recon_mc
        log('MoCo done t_ref=%d (%.1fs)' % (t_ref, t_ref * temp_res))
        if run_second_grabcut:
            log('Second GrabCut...')
            gc_labels = renal_segment_v4(
                recon, temp_res=temp_res, spacing=spacing,
                root_path=root_path, threshold_multiplier=threshold_multiplier,
            )

    grabcut_raw = gc_labels > 0
    gc_filled = hole_fill_grabcut(gc_labels)
    hole_fill_added = (gc_filled > 0) & ~grabcut_raw
    log('After hole-fill: %s  (added vox=%d)' % (
        np.unique(gc_filled), int(hole_fill_added.sum())))

    gc_pre_hilum = gc_filled.copy()
    gc_filled, hilum_removed = subtract_hilum_protrusions(
        gc_filled, spacing,
        fraction=hilum_open_fraction,
        si_band=hilum_si_band,
        log=log,
    )
    log('After hilum step: %s  (removed vox=%d)' % (
        np.unique(gc_filled), int(hilum_removed.sum())))

    kid_ids = [k for k in np.unique(gc_filled) if k > 0]
    kidney = gc_filled > 0
    gc_original = kidney.copy()

    log('PCA + K-means...')
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
        log('Cluster %d: n=%d peak=%.1fs' % (cid, sizes[cid], peaks[cid]))

    cortex_id = max(sizes, key=sizes.get)
    t_ctx = peaks[cortex_id]
    i_ctx = int(np.argmin(np.abs(time - t_ctx)))
    win = max(1, int(round(10.0 / temp_res)))
    log('Cortex (largest)=%d' % cortex_id)

    remaining = [c for c in (1, 2, 3) if c != cortex_id]
    rise = {}
    for cid in remaining:
        c = curves[cid]
        late = c[int(0.67 * len(c)):].mean()
        at_ctx = c[i_ctx:i_ctx + win].mean()
        rise[cid] = float(late - at_ctx)
        log('Cluster %d post-cortical rise=%.1f' % (cid, rise[cid]))
    medulla_id = max(rise, key=rise.get)
    pelvis_id = [c for c in remaining if c != medulla_id][0]
    log('Assignment cortex=%d medulla=%d pelvis=%d' % (
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
        idx = distance_transform_edt(
            ~gc_original, return_distances=False, return_indices=True)
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
                sz = np.bincount(labeled.ravel())
                sz[0] = 0
                keep = labeled == sz.argmax()
                final_labels[m & ~keep] = med_val
        m = final_labels == pel_val
        if m.any():
            labeled, n = cc_label(m)
            if n > 1:
                sz = np.bincount(labeled.ravel())
                sz[0] = 0
                keep = labeled == sz.argmax()
                final_labels[m & ~keep] = ctx_val

    pelvis_mask = (final_labels == 6) | (final_labels == 7)
    cortex_mask = (final_labels == 2) | (final_labels == 3)
    pelvis_frac = uniform_filter(pelvis_mask.astype(np.float32), size=3)
    final_labels[cortex_mask & (pelvis_frac >= 0.45)] = 1

    uncertain_mask = final_labels == 1
    log('Uncertain (before drop): %d' % int(uncertain_mask.sum()))
    final_labels[uncertain_mask] = 0

    log('Final counts cortex/medulla/pelvis: %d %d %d' % (
        int(((final_labels == 2) | (final_labels == 3)).sum()),
        int(((final_labels == 4) | (final_labels == 5)).sum()),
        int(((final_labels == 6) | (final_labels == 7)).sum())))

    whole_kidney = (gc_filled > 0).astype(np.uint8)

    tissue_curves = {}
    for name, vals in [('cortex', (2, 3)), ('medulla', (4, 5)), ('pelvis', (6, 7))]:
        m = np.isin(final_labels, vals)
        tissue_curves[name] = recon[m].mean(axis=0) if m.any() else None

    if output_dir is not None:
        export_all_outputs(
            recon, final_labels, whole_kidney,
            dicom_list, spacing, output_dir,
            temp_res=temp_res, time_index=None,
        )
        pdf_path = os.path.join(output_dir, 'qa_report.pdf')
        write_qa_pdf(
            pdf_path,
            log_lines=log.lines,
            recon=recon,
            temp_res=temp_res,
            final_labels=final_labels,
            whole_kidney=whole_kidney,
            pca_map=pca_map,
            curves_by_tissue=tissue_curves,
            diag={
                'hole_fill_added': hole_fill_added,
                'hilum_removed': hilum_removed,
                'brim': brim,
                'uncertain': uncertain_mask,
            },
            time_index=None,
            pad_ap=3,
        )

    return recon, final_labels