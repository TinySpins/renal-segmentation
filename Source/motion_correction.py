# Source/motion_correction.py
"""
Per-kidney rigid + isotropic-scale motion correction for 4D DCE-MRI.

Call from the pipeline with the Flags values, e.g.:

    recon_mc, t_ref, summary = motion_correct_affine_per_kidney(
        recon, gc_labels, temp_res,
        ref_time_s=MOCO_REF_TIME_S,
        dilate_vox=3,
        maxiter=MOCO_MAXITER,
    )
"""

from __future__ import print_function

import numpy as np
from scipy.ndimage import binary_dilation, affine_transform, center_of_mass
from scipy.optimize import minimize


def _euler_scale_matrix(rx, ry, rz, scale):
    """Rotation (radians) then isotropic scale -> 3x3 matrix."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]], dtype=np.float64)
    return scale * (Rz.dot(Ry).dot(Rx))


def _apply_rigid_scale(vol, params, center):
    """
    params: tx, ty, tz, rx, ry, rz, log_scale
    affine_transform maps output index -> input index.
    """
    tx, ty, tz, rx, ry, rz, log_s = params
    scale = np.exp(log_s)
    A = _euler_scale_matrix(rx, ry, rz, scale)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.eye(3)
    t = np.array([tx, ty, tz], dtype=np.float64)
    c = np.array(center, dtype=np.float64)
    matrix = A_inv
    offset = c - A_inv.dot(c + t)
    return affine_transform(
        vol.astype(np.float32),
        matrix=matrix,
        offset=offset,
        order=1,
        mode='nearest',
    )


def _masked_ncc(a, b, mask):
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    if a.size < 50:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8
    return float((a * b).sum() / denom)


def register_kidney_frame(ref_roi, mov_roi, mask_roi, maxiter=40):
    """
    Estimate tx,ty,tz,rx,ry,rz,log_scale aligning mov -> ref on mask.
    Returns (params, warped_mov, ncc).
    """
    com = center_of_mass(mask_roi)
    if np.any(np.isnan(com)):
        com = 0.5 * np.array(mask_roi.shape, dtype=np.float64)

    def loss(p):
        warped = _apply_rigid_scale(mov_roi, p, com)
        return -_masked_ncc(ref_roi, warped, mask_roi)

    p0 = np.zeros(7, dtype=np.float64)
    res = minimize(
        loss, p0, method='Powell',
        options={'maxiter': int(maxiter), 'xtol': 1e-3, 'ftol': 1e-3},
    )
    warped = _apply_rigid_scale(mov_roi, res.x, com)
    return res.x, warped, -res.fun


def motion_correct_affine_per_kidney(
        recon,
        gc_labels,
        temp_res,
        ref_time_s=10.0,
        dilate_vox=3,
        maxiter=40,
):
    """
    Per-kidney rigid + isotropic scale motion correction.

    Parameters
    ----------
    recon : ndarray, shape (nx, ny, nz, nt)
    gc_labels : ndarray, shape (nx, ny, nz)
        GrabCut labels; values > 0 are kidneys.
    temp_res : float
        Temporal resolution in **seconds**.
    ref_time_s : float
        Reference time in **seconds** (from Flags: MOCO_REF_TIME_S).
    dilate_vox : int
        Dilate kidney mask before registration ROI.
    maxiter : int
        Powell max iterations per frame (from Flags: MOCO_MAXITER).

    Returns
    -------
    recon_mc : float32 ndarray, same shape as recon
    t_ref : int
        Reference frame index
    summary : list of (kid_id, t, params, ncc)
    """
    recon = np.asarray(recon)
    gc_labels = np.asarray(gc_labels)
    nx, ny, nz, nt = recon.shape
    temp_res = float(temp_res)
    ref_time_s = float(ref_time_s)

    t_ref = int(np.clip(round(ref_time_s / temp_res), 0, nt - 1))
    print('MoCo reference: requested %.1f s -> frame %d -> %.1f s  '
          '(temp_res=%.3f s, nt=%d)' % (
              ref_time_s, t_ref, t_ref * temp_res, temp_res, nt))

    ref_vol = recon[..., t_ref].astype(np.float32)
    recon_mc = recon.astype(np.float32).copy()
    summary = []

    kid_ids = sorted(int(k) for k in np.unique(gc_labels) if k > 0)
    if not kid_ids:
        print('MoCo: no kidney labels found; returning copy of recon')
        return recon_mc, t_ref, summary

    for kid_id in kid_ids:
        mask = gc_labels == kid_id
        mask_d = binary_dilation(mask, iterations=int(dilate_vox))
        coords = np.argwhere(mask_d)
        if coords.size == 0:
            continue

        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1
        ref_roi = ref_vol[x0:x1, y0:y1, z0:z1]
        m_roi = mask_d[x0:x1, y0:y1, z0:z1]

        print('MoCo kidney %d  bbox=(%d:%d, %d:%d, %d:%d)  ref_frame=%d' % (
            kid_id, x0, x1, y0, y1, z0, z1, t_ref))

        for t in range(nt):
            if t == t_ref:
                continue
            mov_roi = recon[x0:x1, y0:y1, z0:z1, t].astype(np.float32)
            p_opt, warped, ncc = register_kidney_frame(
                ref_roi, mov_roi, m_roi, maxiter=maxiter)
            recon_mc[x0:x1, y0:y1, z0:z1, t][m_roi] = warped[m_roi]
            summary.append((kid_id, t, p_opt.copy(), float(ncc)))

        print('MoCo kidney %d done' % kid_id)

    return recon_mc, t_ref, summary