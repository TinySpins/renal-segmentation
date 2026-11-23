#!/usr/bin/env python
"""K-means + morphology renal segmentation CLI."""
from __future__ import print_function

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.realpath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Source.pipeline_kmeans_morph import run_kmeans_morph_pipeline


def main():
    p = argparse.ArgumentParser(
        description='Renal segmentation (K-means + morphology)')
    p.add_argument('input_dir', help='Folder with DICOM series')
    p.add_argument('output_dir', help='Folder for outputs')
    p.add_argument('--threshold-multiplier', type=float, default=1.0)
    p.add_argument('--gap-factor', type=float, default=2.5)
    p.add_argument('--moco', action='store_true')
    p.add_argument('--moco-ref-time', type=float, default=10.0)
    p.add_argument('--moco-maxiter', type=int, default=40)
    p.add_argument('--second-grabcut', action='store_true')
    p.add_argument(
        '--hilum-open-fraction', type=float, default=None,
        help='Enable hilum cleanup; radius = fraction * minor-axis (mm). Omit=off.')
    p.add_argument(
        '--hilum-si-band', type=float, default=0.40,
        help='Central SI fraction gating hilum removals (default 0.40).')
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
        hilum_open_fraction=args.hilum_open_fraction,
        hilum_si_band=args.hilum_si_band,
    )


if __name__ == '__main__':
    main()