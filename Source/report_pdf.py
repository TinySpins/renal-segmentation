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
    sl = vol3d[ap_index, :, :]
    return np.rot90(sl, k=1)


def _kidney_ap_range(mask3d, pad=3):
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
    # back → front (high AP index → low); swap if orientation is opposite
    return list(range(a1, a0 - 1, -1))


def _grid_shape(n, max_cols=6):
    cols = min(max_cols, max(1, n))
    rows = int(np.ceil(float(n) / cols))
    return rows, cols


def _page_coronal_grid(fig, recon, ap_list, time_index, overlay_dict,
                       title, legend_items=None):
    n = len(ap_list)
    nrows, ncols = _grid_shape(n, max_cols=6)
    fig.set_size_inches(11.69, 8.27)
    axes = fig.subplots(nrows, ncols)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    bg = recon[:, :, :, time_index]
    for i, ap in enumerate(ap_list):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        img = _coronal_slab(bg, ap)
        ax.imshow(img, cmap='gray', interpolation='none')
        if overlay_dict:
            for name, (mask, style) in overlay_dict.items():
                if mask is None:
                    continue
                if style[0] == 'bool':
                    m = _coronal_slab(mask.astype(np.float32), ap)
                    m[m == 0] = np.nan
                    ax.imshow(m, cmap=style[1], alpha=style[2],
                              vmin=0.5, vmax=1.5, interpolation='none')
                elif style[0] == 'labels':
                    lab = _coronal_slab(mask.astype(np.float32), ap)
                    lab[lab == 0] = np.nan
                    ax.imshow(lab, cmap=style[3], alpha=style[4],
                              vmin=style[1], vmax=style[2], interpolation='none')
        ax.set_title('AP=%d' % ap, fontsize=7)
        ax.axis('off')

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')

    fig.suptitle(title, fontsize=12)
    if legend_items:
        fig.legend(handles=legend_items, loc='lower center',
                   ncol=min(6, len(legend_items)), fontsize=8,
                   framealpha=0.9)
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    else:
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])


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
    _ensure_dir(path)
    nx, ny, nz, nt = recon.shape
    if time_index is None:
        time_index = min(max(nt // 2, 0), nt - 1)

    ap_list = _kidney_ap_range(whole_kidney, pad=pad_ap)

    with PdfPages(path) as pdf:
        # 1) metrics
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle('Processing metrics / log', fontsize=14)
        text = '\n'.join(log_lines) if log_lines else '(no log lines)'
        lines = text.split('\n')
        if len(lines) > 90:
            lines = ['... (%d earlier lines omitted) ...' % (len(lines) - 90)] + lines[-90:]
            text = '\n'.join(lines)
        fig.text(0.05, 0.92, text, family='monospace', fontsize=7,
                 va='top', ha='left', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # 2) curves
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
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3) whole kidney
        fig = plt.figure()
        _page_coronal_grid(
            fig, recon, ap_list, time_index,
            {'wk': (whole_kidney > 0, ('bool', 'Reds', 0.35))},
            'Whole kidney (coronal, back → front)',
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 4) labels
        handles = [
            mpatches.Patch(color=plt.cm.jet((v - 2) / 5.0), label=lab)
            for v, lab in [
                (2, '2 L Cortex'), (3, '3 R Cortex'),
                (4, '4 L Medulla'), (5, '5 R Medulla'),
                (6, '6 L Pelvis'), (7, '7 R Pelvis'),
            ]
        ]
        fig = plt.figure()
        _page_coronal_grid(
            fig, recon, ap_list, time_index,
            {'lab': (final_labels, ('labels', 2, 7, 'jet', 0.45))},
            'Tissue labels (coronal, back → front)',
            legend_items=handles,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 5) diagnostics
        d = diag or {}
        handles5 = []
        overlays = {}
        if d.get('hole_fill_added') is not None and np.any(d['hole_fill_added']):
            overlays['fill'] = (d['hole_fill_added'], ('bool', 'Blues', 0.4))
            handles5.append(mpatches.Patch(color='blue', label='Hole-fill added'))
        if d.get('hilum_removed') is not None and np.any(d['hilum_removed']):
            overlays['hilum'] = (d['hilum_removed'], ('bool', 'Greens', 0.45))
            handles5.append(mpatches.Patch(color='green', label='Hilum removed'))
        if d.get('brim') is not None and np.any(d['brim']):
            overlays['brim'] = (d['brim'], ('bool', 'Reds', 0.4))
            handles5.append(mpatches.Patch(color='red', label='Brim'))
        if d.get('uncertain') is not None and np.any(d['uncertain']):
            overlays['unc'] = (d['uncertain'], ('bool', 'YlOrBr', 0.45))
            handles5.append(mpatches.Patch(color='orange', label='Uncertain / excluded'))
        fig = plt.figure()
        _page_coronal_grid(
            fig, recon, ap_list, time_index, overlays,
            'Diagnostics: fill / hilum / brim / uncertain (coronal)',
            legend_items=handles5 if handles5 else None,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 6) PCA
        fig = plt.figure()
        nrows, ncols = _grid_shape(len(ap_list), max_cols=6)
        fig.set_size_inches(11.69, 8.27)
        axes = fig.subplots(nrows, ncols)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        for i, ap in enumerate(ap_list):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            rgb = np.stack([
                _coronal_slab(pca_map[..., ch], ap) for ch in range(3)
            ], axis=-1)
            rgb = np.clip(rgb, 0, 1)
            ax.imshow(rgb, interpolation='none')
            ax.set_title('AP=%d' % ap, fontsize=7)
            ax.axis('off')
        for j in range(len(ap_list), nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].axis('off')
        fig.suptitle('PCA map (coronal, back → front)', fontsize=12)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    print('QA PDF written:', path)