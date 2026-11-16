# Source/dicom_load_tolerant.py
"""
Robust DICOM volume loading for the renal-segmentation pipeline.

- find_bad_dicoms: files that fail dicom.read_file (even with force=True)
- quarantine_bad_dicoms: move them to _bad_dicom/
- dicom2hdf5_tolerant: build volume from complete trigger x slice grid only
- load_dicom_dataset: auto quarantine + standard loader, fallback to tolerant
"""

from __future__ import print_function

import os
import sys
import shutil
from glob import glob
from collections import Counter

import numpy as np
import dicom

from Source.util import dicom2hdf5


def _list_dicom_files(dicom_path):
    files = glob(os.path.join(dicom_path, '*.dcm'))
    files += glob(os.path.join(dicom_path, '*.mag'))
    # do not scan quarantine subfolder
    files = [f for f in files if os.path.basename(os.path.dirname(f)) != '_bad_dicom'
             and '_bad_dicom' not in f.split(os.sep)]
    return sorted(files)


def _get_trigger(ds):
    if hasattr(ds, 'TriggerTime') and ds.TriggerTime is not None:
        return float(ds.TriggerTime)
    if hasattr(ds, 'AcquisitionTime') and ds.AcquisitionTime is not None:
        at = str(ds.AcquisitionTime)
        hh, mm = float(at[0:2]), float(at[2:4])
        ss = float(at[4:]) if len(at) > 4 else 0.0
        return hh * 3600 + mm * 60 + ss
    if hasattr(ds, 'ContentTime') and ds.ContentTime is not None:
        at = str(ds.ContentTime)
        hh, mm = float(at[0:2]), float(at[2:4])
        ss = float(at[4:]) if len(at) > 4 else 0.0
        return hh * 3600 + mm * 60 + ss
    return 0.0


def find_bad_dicoms(dicom_path, force_try=True):
    """
    Return list of file paths that cannot be read as DICOM images.
    Tries normal read first; optionally force=True; still requires pixel data.
    """
    files = _list_dicom_files(dicom_path)
    bad = []
    print('Scanning %d files for unreadable DICOM...' % len(files))
    for i, fp in enumerate(files):
        if i % 500 == 0:
            print('\t%d/%d' % (i, len(files)))
            sys.stdout.flush()
        ok = False
        try:
            ds = dicom.read_file(fp)
            _ = ds.pixel_array
            ok = True
        except Exception:
            if force_try:
                try:
                    ds = dicom.read_file(fp, force=True)
                    _ = ds.pixel_array
                    ok = True
                except Exception:
                    ok = False
            else:
                ok = False
        if not ok:
            bad.append(fp)
    print('Unreadable files: %d' % len(bad))
    return bad


def quarantine_bad_dicoms(dicom_path, bad=None):
    """Move bad files to dicom_path/_bad_dicom/. Returns list of moved paths."""
    if bad is None:
        bad = find_bad_dicoms(dicom_path)
    if not bad:
        print('No bad files to quarantine')
        return []

    qdir = os.path.join(dicom_path, '_bad_dicom')
    try:
        os.makedirs(qdir)
    except OSError:
        pass

    moved = []
    for fp in bad:
        dest = os.path.join(qdir, os.path.basename(fp))
        if os.path.abspath(fp) == os.path.abspath(dest):
            continue
        if not os.path.isfile(fp):
            continue
        shutil.move(fp, dest)
        moved.append(dest)
    print('Quarantined %d files under %s' % (len(moved), qdir))
    return moved


def dicom2hdf5_tolerant(dicom_path):
    """
    Build 4D recon from a (possibly incomplete) DICOM folder.
    Keeps only trigger times that have a full set of slices.
    """
    dicom_list = _list_dicom_files(dicom_path)
    print(len(dicom_list), 'files found (tolerant loader)')
    if not dicom_list:
        raise ValueError('No DICOM files in ' + dicom_path)

    ds0 = None
    for fp in dicom_list:
        try:
            ds0 = dicom.read_file(fp)
            break
        except Exception:
            try:
                ds0 = dicom.read_file(fp, force=True)
                break
            except Exception:
                continue
    if ds0 is None:
        raise ValueError('Could not read any DICOM in ' + dicom_path)

    rows = int(ds0.Rows)
    cols = int(ds0.Columns)
    fa = float(getattr(ds0, 'FlipAngle', 0) or 0)
    tr = float(getattr(ds0, 'RepetitionTime', 0) or 0)
    te = float(getattr(ds0, 'EchoTime', 0) or 0)
    x_spacing = float(ds0.PixelSpacing[0])
    y_spacing = float(ds0.PixelSpacing[1])
    if hasattr(ds0, 'SpacingBetweenSlices'):
        z_spacing = float(ds0.SpacingBetweenSlices)
    else:
        z_spacing = 0.0
    spacing = [x_spacing, y_spacing, z_spacing]

    entries = []
    for i, fp in enumerate(dicom_list):
        if i % max(1, len(dicom_list) // 10) == 0:
            print('\t%d/%d' % (i, len(dicom_list)))
            sys.stdout.flush()
        try:
            ds = dicom.read_file(fp)
        except Exception:
            try:
                ds = dicom.read_file(fp, force=True)
            except Exception:
                print('Skip unreadable:', os.path.basename(fp))
                continue
        try:
            pix = ds.pixel_array
            sl = float(ds.SliceLocation)
            tg = _get_trigger(ds)
        except Exception as e:
            print('Skip (no pixel/slice):', os.path.basename(fp), e)
            continue
        entries.append((pix, sl, tg, fp))
    print('\tDone')

    if not entries:
        raise ValueError('No readable image instances in ' + dicom_path)

    slice_locations = sorted(set(e[1] for e in entries))
    triggers_all = sorted(set(e[2] for e in entries))
    num_slices = len(slice_locations)
    print('Unique slices:', num_slices)
    print('Unique triggers:', len(triggers_all))

    grid = {}
    for e in entries:
        grid[(e[2], e[1])] = e

    good_triggers = []
    for tg in triggers_all:
        n = sum(1 for sl in slice_locations if (tg, sl) in grid)
        if n == num_slices:
            good_triggers.append(tg)
        else:
            print('Incomplete phase trigger=%.3f: %d/%d slices — drop' % (
                tg, n, num_slices))

    if not good_triggers:
        counts = []
        for tg in triggers_all:
            n = sum(1 for sl in slice_locations if (tg, sl) in grid)
            counts.append((n, tg))
        best_n = Counter(c[0] for c in counts).most_common(1)[0][0]
        good_triggers = [tg for n, tg in counts if n == best_n]
        print('Fallback: keep triggers with %d slices each (%d phases)' % (
            best_n, len(good_triggers)))
        common_sl = set(slice_locations)
        for tg in good_triggers:
            common_sl &= set(sl for sl in slice_locations if (tg, sl) in grid)
        slice_locations = sorted(common_sl)
        num_slices = len(slice_locations)
        print('Using %d common slices' % num_slices)

    good_triggers = sorted(good_triggers)
    num_timepoints = len(good_triggers)
    if num_timepoints < 2:
        raise ValueError('Fewer than 2 complete time points after filtering')

    temporal_resolution = float(good_triggers[1] - good_triggers[0])

    print('Kept phases:', num_timepoints)
    print('Data Shape = (%d, %d, %d, %d)' % (rows, cols, num_slices, num_timepoints))
    print('Spacing (mm) =', spacing)
    print('Temporal Res =', temporal_resolution)

    dtype = entries[0][0].dtype
    data4d = np.zeros((rows, cols, num_slices, num_timepoints), dtype=dtype)
    sorted_dicom_list = []
    for it, tg in enumerate(good_triggers):
        for iz, sl in enumerate(slice_locations):
            pix, _, _, fp = grid[(tg, sl)]
            data4d[:, :, iz, it] = pix
            sorted_dicom_list.append(fp)

    return {
        'recon': data4d,
        'shape': data4d.shape,
        'spacing': spacing,
        'temp_res': temporal_resolution,
        'fa': fa,
        'tr': tr,
        'te': te,
        'dicom_list': sorted_dicom_list,
    }


def load_dicom_dataset(dicom_path, quarantine=True, always_scan=True):
    """
    Auto path:
      1) optionally scan + quarantine unreadable files
      2) try standard dicom2hdf5
      3) on failure, use dicom2hdf5_tolerant
    """
    if always_scan or quarantine:
        bad = find_bad_dicoms(dicom_path, force_try=True)
        if bad and quarantine:
            quarantine_bad_dicoms(dicom_path, bad=bad)
        elif bad:
            print('Warning: %d bad files present (quarantine=False)' % len(bad))

    try:
        print('Trying standard dicom2hdf5...')
        dataset = dicom2hdf5(dicom_path)
        print('Standard loader OK')
        return dataset
    except Exception as e:
        print('Standard loader failed: %s' % e)
        print('Falling back to dicom2hdf5_tolerant...')
        dataset = dicom2hdf5_tolerant(dicom_path)
        print('Tolerant loader OK')
        return dataset