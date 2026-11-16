#!/usr/bin/env python
"""
K-means + morphology renal segmentation CLI.

Usage:
  python renalSegment_kmeans_morph.py /data/DicomIn /data/DicomOut
  python renalSegment_kmeans_morph.py /data/DicomIn /data/DicomOut --moco
"""
from __future__ import print_function

import os
import sys
import argparse

# ensure repo root on path when run as script
ROOT = os.path.dirname(os.path.realpath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Source.pipeline_kmeans_morph import run_kmeans_morph_pipeline


def main():
    p = argparse.ArgumentParser(description='Renal segmentation (K-means + morphology)')
    p.add_argument('input_dir', help='Folder with DICOM series')
    p.add_argument('output_dir', help='Folder for label DICOMs / npy')
    p.add_argument('--threshold-multiplier', type=float, default=1.0)
    p.add_argument('--gap-factor', type=float, default=2.5,
                    help='Trailing trigger gap vs median to drop last phases')
    p.add_argument('--moco', action='store_true', help='Enable motion correction')
    p.add_argument('--moco-ref-time', type=float, default=10.0,
                    help='MoCo reference time in seconds')
    p.add_argument('--moco-maxiter', type=int, default=40)
    p.add_argument('--second-grabcut', action='store_true',
                    help='Re-run GrabCut after MoCo')
    args = p.parse_args()

    run_kmeans_morph_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold_multiplier=args.threshold_multiplier,
        root_path=ROOT,
        gap_factor=args.gap_factor,
        run_moco=args.moco,
        moco_ref_time_s=args.moco_ref_time,
        moco_maxiter=args.moco_maxiter,
        run_second_grabcut=args.second_grabcut,
    )


if __name__ == '__main__':
    main()