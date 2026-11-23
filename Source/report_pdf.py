# Source/report_pdf.py
"""PDF QA report for kmeans-morph pipeline."""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        try:
            os.makedirs(d)
        except OSError:
            pass


def _coronal_slab(vol3d, ap_index):
    """vol3d: (rows, cols, z). One coronal sheet at first-axis index."""
    sl = vol3d[ap_index, :, :]
    return np.rot90(sl, k=1)


def _kidney_ap_range(mask3d, pad=3):
    """AP indices with kidney present, plus pad; ordered back → front."""
    if mask3d is None or not np.any(mask3d):
        n = mask3d.shape[0] if mask3d is not None else 1
        mid = n // 2
        return list(range(max(0, mid - 2), min(n, mid + 3)))
    present = np.any(mask3d, axis=(1, 2))
    idxs = np.where(present)[0]
    if idxs.size == 0:
        return [mask3d.shape[0] // 2]
    a0, a1 = int(idxs.min()), int(idxs.max())
    a0 = max(0, a0 - pad)
    a1 = min(mask3d.shape[0] - 1, a1 + pad)
    # high → low AP index as back → front (swap if needed on your data)
    return list(range(a1, a0 - 1, -1))


def _grid_shape(n, max_cols=8):
    cols = min(max_cols, max(1, n))
    rows = int(np.ceil(float(n) / cols))
    return rows, cols


def _jet_label_color(val):
    """Color for tissue label val with vmin=2, vmax=7 on jet."""
    return plt.cm.jet((float(val) - 2.0) / 5.0)


def _page_coronal_grid(fig, recon, ap_list, time_index, overlay_dict,
                       title, legend_items=None):
    """
    Fill page with coronal thumbnails (left→right, then down).
    overlay_dict: name -> (mask3d, style)
      style ('bool', cmap, alpha) or ('labels', vmin, vmax, cmap, alpha)
    """
    n = len(ap_list)
    nrows, ncols = _grid_shape(n, max_cols=8)
    fig.set_size_inches(11.69, 8.27)

    bottom = 0.10 if legend_items else 0.03
    fig.subplots_adjust(
        left=0.02, right=0.98,
        top=0.93, bottom=bottom,
        wspace=0.04, hspace=0.12,
    )

    axes = []
    for i in range(nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        axes.append(ax)
    axes = np.array(axes).reshape(nrows, ncols)

    bg = recon[:, :, :, time_index]
    for i, ap in enumerate(ap_list):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        img = _coronal_slab(bg, ap)
        ax.imshow(img, cmap='gray', interpolation='none', aspect='equal')
        if overlay_dict:
            for name, (mask, style) in overlay_dict.items():
                if mask is None:
                    continue
                if style[0] == 'bool':
                    m = _coronal_slab(mask.astype(np.float32), ap)
                    m[m == 0] = np.nan
                    ax.imshow(
                        m, cmap=style[1], alpha=style[2],
                        vmin=0.5, vmax=1.5, interpolation='none',
                        aspect='equal')
                elif style[0] == 'labels':
                    lab = _coronal_slab(mask.astype(np.float32), ap)
                    lab[lab == 0] = np.nan
                    ax.imshow(
                        lab, cmap=style[3], alpha=style[4],
                        vmin=style[1], vmax=style[2],
                        interpolation='none', aspect='equal')
        ax.set_title('AP=%d' % ap, fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')

    fig.suptitle(title, fontsize=12, y=0.98)

    if legend_items:
        fig.legend(
            handles=legend_items,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(6, len(legend_items)),
            fontsize=8,
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            edgecolor='0.5',
        )


def write_qa_pdf(
        path,
        log_lines,
        recon,
        temp_res,
        final_labels,
        whole_kidney,
        pca_map,
        curves_by_tissue,
        diag,
        time_index=None,
        pad_ap=3,
):
    """
    6-page QA PDF:
      1 metrics/log
      2 tissue time curves
      3 whole kidney coronal grid
      4 tissue labels + legend
      5 diagnostics + legend
      6 PCA coronal grid
    """
    _ensure_dir(path)
    nx, ny, nz, nt = recon.shape
    if time_index is None:
        time_index = min(max(nt // 2, 0), nt - 1)

    ap_list = _kidney_ap_range(whole_kidney, pad=pad_ap)

    with PdfPages(path) as pdf:
        # ----- 1) metrics -----
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle('Processing metrics / log', fontsize=14, y=0.97)
        text = '\n'.join(log_lines) if log_lines else '(no log lines)'
        lines = text.split('\n')
        if len(lines) > 95:
            lines = (
                ['... (%d earlier lines omitted) ...' % (len(lines) - 95)]
                + lines[-95:]
            )
            text = '\n'.join(lines)
        fig.text(
            0.04, 0.93, text,
            family='monospace', fontsize=7,
            va='top', ha='left', wrap=True,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ----- 2) time curves -----
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(1, 1, 1)
        t = np.arange(nt) * float(temp_res)
        colors = {'cortex': 'C0', 'medulla': 'C1', 'pelvis': 'C2'}
        if curves_by_tissue:
            for name, y in curves_by_tissue.items():
                if y is None:
                    continue
                ax.plot(t, y, label=name, color=colors.get(name), lw=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mean signal')
        ax.set_title('Mean signal intensity per tissue')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ----- 3) whole kidney -----
        fig = plt.figure()
        _page_coronal_grid(
            fig, recon, ap_list, time_index,
            {'wk': (whole_kidney > 0, ('bool', 'Reds', 0.35))},
            'Whole kidney (coronal, back → front)',
            legend_items=None,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ----- 4) tissue labels -----
        handles_labels = [
            mpatches.Patch(color=_jet_label_color(2), label='2 L Cortex'),
            mpatches.Patch(color=_jet_label_color(3), label='3 R Cortex'),
            mpatches.Patch(color=_jet_label_color(4), label='4 L Medulla'),
            mpatches.Patch(color=_jet_label_color(5), label='5 R Medulla'),
            mpatches.Patch(color=_jet_label_color(6), label='6 L Pelvis'),
            mpatches.Patch(color=_jet_label_color(7), label='7 R Pelvis'),
        ]
        fig = plt.figure()
        _page_coronal_grid(
            fig, recon, ap_list, time_index,
            {'lab': (final_labels, ('labels', 2, 7, 'jet', 0.45))},
            'Tissue labels (coronal, back → front)',
            legend_items=handles_labels,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ----- 5) diagnostics -----
        d = diag or {}
        handles5 = []
        overlays = {}
        if d.get('hole_fill_added') is not None and np.any(d['hole_fill_added']):
            overlays['fill'] = (d['hole_fill_added'], ('bool', 'Blues', 0.45))
            handles5.append(
                mpatches.Patch(color=plt.cm.Blues(0.75), label='Hole-fill added'))
        if d.get('hilum_removed') is not None and np.any(d['hilum_removed']):
            overlays['hilum'] = (d['hilum_removed'], ('bool', 'Greens', 0.45))
            handles5.append(
                mpatches.Patch(color=plt.cm.Greens(0.75), label='Hilum removed'))
        if d.get('brim') is not None and np.any(d['brim']):
            overlays['brim'] = (d['brim'], ('bool', 'Reds', 0.45))
            handles5.append(
                mpatches.Patch(color=plt.cm.Reds(0.75), label='Brim'))
        if d.get('uncertain') is not None and np.any(d['uncertain']):
            overlays['unc'] = (d['uncertain'], ('bool', 'YlOrBr', 0.5))
            handles5.append(
                mpatches.Patch(color=plt.cm.YlOrBr(0.75),
                               label='Uncertain / excluded'))

        fig = plt.figure()
        _page_coronal_grid(
            fig, recon, ap_list, time_index, overlays,
            'Diagnostics: fill / hilum / brim / uncertain (coronal)',
            legend_items=handles5 if handles5 else None,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ----- 6) PCA -----
        fig = plt.figure(figsize=(11.69, 8.27))
        n = len(ap_list)
        nrows, ncols = _grid_shape(n, max_cols=8)
        fig.subplots_adjust(
            left=0.02, right=0.98, top=0.93, bottom=0.03,
            wspace=0.04, hspace=0.12,
        )
        for i, ap in enumerate(ap_list):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            rgb = np.stack([
                _coronal_slab(pca_map[..., ch], ap) for ch in range(3)
            ], axis=-1)
            rgb = np.clip(rgb, 0, 1)
            ax.imshow(rgb, interpolation='none', aspect='equal')
            ax.set_title('AP=%d' % ap, fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        for j in range(n, nrows * ncols):
            ax = fig.add_subplot(nrows, ncols, j + 1)
            ax.axis('off')
        fig.suptitle('PCA map (coronal, back → front)', fontsize=12, y=0.98)
        pdf.savefig(fig)
        plt.close(fig)

    print('QA PDF written:', path)