# ============================================================
# UV Pixel Border Display (Blender 4.2+) v1.0.1
# Features:
#   - Pixelated UV island border overlay (fixed screen pixel size)
#   - Selected border edges highlighted (theme select color or custom)
#   - Texel-space expansion preview for bake dilation (scales with UV zoom)
#   - Mask export: expansion area as PNG (white=expanded, black=background)
#   - Auto seam marking: Maya-style, marks UV-disconnected edges as seams
#   - Feather edge: softens mask edges on export
# Performance:
#   - depsgraph filtering: only Mesh geometry/transform changes trigger rebuild
#   - Bitmap deduplication, expansion computed by UV bbox (viewport zoom zero-cost)
# Panel location: UV Editor sidebar (N) -> "Pixel Border"
# ============================================================
bl_info = {
    "name": "UV Pixel Border Display v1.0.1",
    "author": "TraeWork",
    "version": (1, 0, 1),
    "blender": (4, 2, 0),
    "location": "UV Editor sidebar (N) -> Pixel Border",
    "description": "Pixelated UV island border overlay with selection highlight, texel-space bake expansion preview, mask export and Maya-style auto seam marking",
    "category": "UV",
    "tracker_url": "https://github.com/LENS6/UV-Pixel-Border-Display/issues",
}

import bpy
import bmesh
import gpu
import os
import numpy as np
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, StringProperty


ITEMS_TEXTURE_SIZE = [
    ("512", "512", ""),
    ("1024", "1024", ""),
    ("2048", "2048", ""),
    ("4096", "4096", ""),
    ("8192", "8192", ""),
]

# Maximum expansion grid pixels (prevents memory explosion with large texture sizes / UV spans)
_EXPAND_GRID_LIMIT = 128 * 1024 * 1024

# Auto seam timer interval (seconds)
_SEAM_TIMER_INTERVAL = 0.3


# ------------------------------------------------------------
# Translation helper (for optional Chinese localization)
# ------------------------------------------------------------
def _t(en_text, zh_text):
    """Return text based on Blender UI language (for optional translation)"""
    try:
        locale = bpy.app.translations.locale
        if locale and locale.startswith('zh'):
            return zh_text
    except:
        pass
    return en_text


# ------------------------------------------------------------
# State
# ------------------------------------------------------------
class _State:
    __slots__ = ('handler', 'dirty', 'segs', 'segs_sel',
                 'batch', 'batch_sel', 'batch_expand',
                 'view_key', 'expand_key', 'shader', 'seam_pending')

    def __init__(self):
        self.handler = None
        self.dirty = True
        self.segs = None            # (N,4) float32: unselected border u0,v0,u1,v1
        self.segs_sel = None        # (M,4) float32: selected border
        self.batch = None           # unselected border blocks (screen pixels)
        self.batch_sel = None       # selected border blocks (screen pixels)
        self.batch_expand = None    # expansion area blocks (texel space, viewport-independent)
        self.view_key = None        # view key for screen pixel borders
        self.expand_key = None      # expansion parameter key (viewport-independent)
        self.shader = None
        self.seam_pending = False   # pending auto seam scan


_ST = _State()


def _prefs():
    """Access add-on preferences using __package__ (Extension-compliant)"""
    try:
        # Use __package__ instead of __name__ as required by Blender Extensions[reference:1]
        return bpy.context.preferences.addons[__package__].preferences
    except Exception:
        return None


def _tag_all_uv_areas():
    try:
        for win in bpy.context.window_manager.windows:
            for ar in win.screen.areas:
                if ar.type == 'IMAGE_EDITOR':
                    ar.tag_redraw()
    except Exception:
        pass


def _modal_running():
    try:
        for win in bpy.context.window_manager.windows:
            if win.modal_operators:
                return True
    except Exception:
        pass
    return False


def _theme_select_color():
    """Read theme select color (view_3d.edge_select), fallback to orange"""
    try:
        th = bpy.context.preferences.themes[0]
        c = th.view_3d.edge_select
        return (c[0], c[1], c[2], 1.0)
    except Exception:
        return (1.0, 0.5, 0.1, 1.0)


# Blender 5.0+ moved UV selection from BMLoopUV to BMLoop (uv_select_vert/uv_select_edge)
_UVSEL_NEW_API = bpy.app.version >= (5, 0, 0)

# Texture size options (shared by prefs and export operator)
_EXPAND_SIZE_ITEMS = [(s, s, "") for s in ("512", "1024", "2048", "4096", "8192")]


# ------------------------------------------------------------
# Data: UV-disconnected boundary edges (radial UV unwelded = boundary, independent of seam marks)
# Returns (unselected segs, selected segs)
# Selection logic:
#   - UV Sync mode: both mesh vertices selected
#   - Non-Sync mode: UV edge selected (5.x: loop.uv_select_edge / 4.x: both UV vertices selected)
# ------------------------------------------------------------
def _collect_boundary_segments():
    normal = []
    selected = []
    try:
        use_sync = bpy.context.tool_settings.use_uv_select_sync
    except Exception:
        use_sync = False
    new_api = _UVSEL_NEW_API
    for obj in bpy.context.objects_in_mode_unique_data:
        if obj.type != 'MESH':
            continue
        bm = bmesh.from_edit_mesh(obj.data)
        uv = bm.loops.layers.uv.active
        if uv is None:
            continue
        for e in bm.edges:
            boundary = False
            for l in e.link_loops:
                r = l.link_loop_radial_next
                if r is l:
                    boundary = True          # mesh open edge
                    break
                # radial loop ring: v0 matches l / r.link_loop_next, v1 matches l.next / r
                if (l[uv].uv != r.link_loop_next[uv].uv
                        or l.link_loop_next[uv].uv != r[uv].uv):
                    boundary = True          # UV disconnected (including overlapping disconnects)
                    break
            if boundary:
                for l in e.link_loops:       # draw both sides of disconnected edge
                    if l.face.hide:
                        continue
                    p0 = l[uv].uv
                    p1 = l.link_loop_next[uv].uv
                    seg = (p0.x, p0.y, p1.x, p1.y)
                    if use_sync:
                        sel = l.vert.select and l.link_loop_next.vert.select
                    elif new_api:
                        # Blender 5.x: UV selection stored on loop; uv_select_edge doesn't update
                        # for SHARED_LOCATION sticky mode when UVs are moved, so check both verts
                        sel = l.uv_select_vert and l.link_loop_next.uv_select_vert
                    else:
                        sel = l[uv].select and l.link_loop_next[uv].select
                    if sel:
                        selected.append(seg)
                    else:
                        normal.append(seg)
    a = np.array(normal, dtype=np.float32) if normal else None
    b = np.array(selected, dtype=np.float32) if selected else None
    return a, b


# ------------------------------------------------------------
# Data: Collect face triangles for fill export
# Returns (N,3,2) array, each triangle with 3 UV coordinates
# selected_only=True: only faces where all vertices are selected
#   (Sync: vert.select, Non-Sync: uv_select_vert)
# Also compatible with edge selection mode (all edges selected = face selected)
# ------------------------------------------------------------
def _collect_fill_triangles(selected_only=False):
    tris = []
    try:
        use_sync = bpy.context.tool_settings.use_uv_select_sync
    except Exception:
        use_sync = False
    new_api = _UVSEL_NEW_API
    for obj in bpy.context.objects_in_mode_unique_data:
        if obj.type != 'MESH':
            continue
        bm = bmesh.from_edit_mesh(obj.data)
        uv = bm.loops.layers.uv.active
        if uv is None:
            continue
        bm.faces.ensure_lookup_table()
        loop_tris = bm.calc_loop_triangles()
        for tri in loop_tris:
            face = tri[0].face
            if face.hide:
                continue

            # Face selection check
            if selected_only:
                if use_sync:
                    all_sel = all(l.vert.select for l in face.loops)
                else:
                    if new_api:
                        all_verts_sel = all(l.uv_select_vert for l in face.loops)
                        # Edge selection: check if both loops of each edge are marked as edge-selected
                        all_edges_sel = True
                        for i in range(len(face.loops)):
                            l1 = face.loops[i]
                            l2 = face.loops[(i+1) % len(face.loops)]
                            if not (l1.uv_select_edge and l2.uv_select_edge):
                                all_edges_sel = False
                                break
                        all_sel = all_verts_sel or all_edges_sel
                    else:
                        all_sel = all(l[uv].select for l in face.loops)
                if not all_sel:
                    continue

            uv0 = tri[0][uv].uv
            uv1 = tri[1][uv].uv
            uv2 = tri[2][uv].uv
            tris.append((uv0.x, uv0.y, uv1.x, uv1.y, uv2.x, uv2.y))
    if not tris:
        return None
    return np.array(tris, dtype=np.float32).reshape(-1, 3, 2)


# ------------------------------------------------------------
# Screen pixel rules: border block rasterization (viewport-zoom-independent)
# ------------------------------------------------------------
def _rasterize_cells(segs, a, b, c, d, w, h, ps):
    """Vectorized rasterization: sample segments at <=1 pixel steps, return pixel grid coords (M,2) int"""
    x0 = (segs[:, 0].astype(np.float64) * a + b) / ps
    y0 = (segs[:, 1].astype(np.float64) * c + d) / ps
    x1 = (segs[:, 2].astype(np.float64) * a + b) / ps
    y1 = (segs[:, 3].astype(np.float64) * c + d) / ps

    n = np.maximum(np.ceil(np.maximum(np.abs(x1 - x0), np.abs(y1 - y0))), 1.0).astype(np.int64)
    counts = n + 1
    total = int(counts.sum())
    if total == 0:
        return None

    seg_idx = np.repeat(np.arange(len(n)), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    t = (np.arange(total, dtype=np.float64) - starts) / n[seg_idx]

    xs = x0[seg_idx] + t * (x1[seg_idx] - x0[seg_idx])
    ys = y0[seg_idx] + t * (y1[seg_idx] - y0[seg_idx])
    cx = np.floor(xs).astype(np.int64)
    cy = np.floor(ys).astype(np.int64)
    max_x = int(w / ps) + 1
    max_y = int(h / ps) + 1
    mask = (cx >= 0) & (cx <= max_x) & (cy >= 0) & (cy <= max_y)
    if not mask.any():
        return None

    cx = cx[mask]
    cy = cy[mask]
    # Bitmap deduplication (faster than np.unique sort)
    grid = np.zeros((max_y + 2, max_x + 2), dtype=bool)
    grid[cy, cx] = True
    gy, gx = np.nonzero(grid)
    return np.stack([gx, gy], axis=1)


def _cells_to_batch(cells, a, b, c, d, ps):
    """Pixel grid -> UV space solid blocks (inverse transform ensures exact pixel grid alignment)"""
    px0 = cells[:, 0].astype(np.float64) * ps
    py0 = cells[:, 1].astype(np.float64) * ps
    ux0 = ((px0 - b) / a).astype(np.float32)
    uy0 = ((py0 - d) / c).astype(np.float32)
    ux1 = ((px0 + ps - b) / a).astype(np.float32)
    uy1 = ((py0 + ps - d) / c).astype(np.float32)

    m = cells.shape[0]
    v = np.empty((m * 6, 2), dtype=np.float32)
    v[0::6, 0] = ux0; v[0::6, 1] = uy0
    v[1::6, 0] = ux1; v[1::6, 1] = uy0
    v[2::6, 0] = ux1; v[2::6, 1] = uy1
    v[3::6, 0] = ux0; v[3::6, 1] = uy0
    v[4::6, 0] = ux1; v[4::6, 1] = uy1
    v[5::6, 0] = ux0; v[5::6, 1] = uy1
    return batch_for_shader(_ST.shader, 'TRIS', {"pos": v})


# ------------------------------------------------------------
# Texel rules: expansion area (scales with viewport zoom, but computation is viewport-independent)
# ------------------------------------------------------------
def _rasterize_cells_uv(segs, t, ix0, iy0, W, H):
    """Rasterize UV segments to texel grid (local origin ix0,iy0, return local grid coords)"""
    x0 = segs[:, 0].astype(np.float64) / t
    y0 = segs[:, 1].astype(np.float64) / t
    x1 = segs[:, 2].astype(np.float64) / t
    y1 = segs[:, 3].astype(np.float64) / t

    n = np.maximum(np.ceil(np.maximum(np.abs(x1 - x0), np.abs(y1 - y0))), 1.0).astype(np.int64)
    counts = n + 1
    total = int(counts.sum())
    if total == 0:
        return None

    seg_idx = np.repeat(np.arange(len(n)), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    tt = (np.arange(total, dtype=np.float64) - starts) / n[seg_idx]

    xs = x0[seg_idx] + tt * (x1[seg_idx] - x0[seg_idx])
    ys = y0[seg_idx] + tt * (y1[seg_idx] - y0[seg_idx])
    cx = np.floor(xs).astype(np.int64) - ix0
    cy = np.floor(ys).astype(np.int64) - iy0
    mask = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    if not mask.any():
        return None
    # Bitmap deduplication
    grid = np.zeros((H, W), dtype=bool)
    grid[cy[mask], cx[mask]] = True
    gy, gx = np.nonzero(grid)
    return np.stack([gx, gy], axis=1)


def _dilate_cells_grid(cells, n, W, H):
    """Dilate texel grid outward by n rings (8-neighbor iteration, pure numpy morphology)
    Returns full dilation area local coords — includes original boundary texels
    (bake dilation covers boundaries; removing them would expose seams)"""
    g = np.zeros((H, W), dtype=bool)
    g[cells[:, 1], cells[:, 0]] = True

    for _ in range(n):
        g2 = g.copy()
        g2[:-1, :] |= g[1:, :]      # up
        g2[1:, :] |= g[:-1, :]      # down
        g2[:, :-1] |= g[:, 1:]      # left
        g2[:, 1:] |= g[:, :-1]      # right
        g2[:-1, :-1] |= g[1:, 1:]   # diagonals ×4
        g2[1:, 1:] |= g[:-1, :-1]
        g2[:-1, 1:] |= g[1:, :-1]
        g2[1:, :-1] |= g[:-1, 1:]
        g = g2

    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        return None
    return np.stack([xs, ys], axis=1)


def _texel_cells_to_batch(cells, ix0, iy0, t):
    """Texel grid -> UV space solid blocks (size = 1 texel, scales with viewport zoom)"""
    ux0 = ((cells[:, 0] + ix0) * t).astype(np.float32)
    uy0 = ((cells[:, 1] + iy0) * t).astype(np.float32)
    ux1 = ux0 + np.float32(t)
    uy1 = uy0 + np.float32(t)

    m = cells.shape[0]
    v = np.empty((m * 6, 2), dtype=np.float32)
    v[0::6, 0] = ux0; v[0::6, 1] = uy0
    v[1::6, 0] = ux1; v[1::6, 1] = uy0
    v[2::6, 0] = ux1; v[2::6, 1] = uy1
    v[3::6, 0] = ux0; v[3::6, 1] = uy0
    v[4::6, 0] = ux1; v[4::6, 1] = uy1
    v[5::6, 0] = ux0; v[5::6, 1] = uy1
    return batch_for_shader(_ST.shader, 'TRIS', {"pos": v})


def _expand_bbox(segs, t, n):
    """Texel bbox for expansion area (with n+1 ring margin), returns (ix0, iy0, ix1, iy1)"""
    umin = float(min(segs[:, 0].min(), segs[:, 2].min()))
    umax = float(max(segs[:, 0].max(), segs[:, 2].max()))
    vmin = float(min(segs[:, 1].min(), segs[:, 3].min()))
    vmax = float(max(segs[:, 1].max(), segs[:, 3].max()))
    ix0 = int(np.floor(umin / t)) - n - 1
    iy0 = int(np.floor(vmin / t)) - n - 1
    ix1 = int(np.ceil(umax / t)) + n + 1
    iy1 = int(np.ceil(vmax / t)) + n + 1
    return ix0, iy0, ix1, iy1


def _rebuild_expand(p):
    """Compute texel expansion area from boundary UV bbox (only called on boundary/param change, not on viewport zoom)"""
    _ST.batch_expand = None
    _ST.expand_key = None
    expand_px = getattr(p, 'expand_px', 0) if p is not None else 0
    if expand_px <= 0:
        return
    img_size = int(getattr(p, 'expand_image_size', '2048')) if p is not None else 2048

    parts = [s for s in (_ST.segs, _ST.segs_sel) if s is not None and len(s) > 0]
    if not parts:
        _ST.expand_key = (img_size, expand_px, 0)
        return
    allsegs = np.concatenate(parts, axis=0)

    t = 1.0 / img_size
    n = expand_px
    ix0, iy0, ix1, iy1 = _expand_bbox(allsegs, t, n)
    W = ix1 - ix0 + 1
    H = iy1 - iy0 + 1
    if W * H > _EXPAND_GRID_LIMIT:
        # UV span too large (e.g., multiple tiles at 8192), skip to prevent freezing
        _ST.expand_key = (img_size, expand_px, -1)
        return

    cells = _rasterize_cells_uv(allsegs, t, ix0, iy0, W, H)
    if cells is not None and len(cells) > 0:
        dil = _dilate_cells_grid(cells, n, W, H)
        if dil is not None and len(dil) > 0:
            _ST.batch_expand = _texel_cells_to_batch(dil, ix0, iy0, t)
    _ST.expand_key = (img_size, expand_px, 1)


# ------------------------------------------------------------
# Feather function
# ------------------------------------------------------------
def _feather_alpha(alpha, radius):
    """Feather alpha channel (Gaussian blur), radius in pixels.
    Returns feathered grayscale values (0-1), background 0, foreground 1, edge transition.
    """
    if radius <= 0:
        return alpha
    try:
        from scipy.ndimage import gaussian_filter
        sigma = radius / 2.0
        blurred = gaussian_filter(alpha.astype(np.float32), sigma=sigma, mode='constant', cval=0.0)
        blurred = np.clip(blurred, 0.0, 1.0)
        return blurred
    except ImportError:
        # No scipy, use separable convolution approximation
        kernel = np.ones((radius,), dtype=np.float32) / radius
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=alpha.astype(np.float32))
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=tmp)
        out = np.clip(out, 0.0, 1.0)
        return out


# ------------------------------------------------------------
# Fill mask generation (supports feather) — outputs RGBA, alpha=1, RGB=grayscale
# ------------------------------------------------------------
def _rasterize_triangles(tris_uv, t, img_size):
    """Rasterize triangle list to texel grid, return covered cell coordinates"""
    cells_list = []
    tris_tex = tris_uv / t
    for tri in tris_tex:
        v0, v1, v2 = tri[0], tri[1], tri[2]
        x_min = int(np.floor(min(v0[0], v1[0], v2[0])))
        x_max = int(np.ceil(max(v0[0], v1[0], v2[0]))) - 1
        y_min = int(np.floor(min(v0[1], v1[1], v2[1])))
        y_max = int(np.ceil(max(v0[1], v1[1], v2[1]))) - 1
        x_min = max(x_min, 0)
        x_max = min(x_max, img_size - 1)
        y_min = max(y_min, 0)
        y_max = min(y_max, img_size - 1)
        if x_min > x_max or y_min > y_max:
            continue

        xs = np.arange(x_min, x_max + 1)
        ys = np.arange(y_min, y_max + 1)
        xx, yy = np.meshgrid(xs, ys)
        points = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1).astype(np.float64)

        v0 = np.asarray(v0, dtype=np.float64)
        v1 = np.asarray(v1, dtype=np.float64)
        v2 = np.asarray(v2, dtype=np.float64)
        detT = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
        if abs(detT) < 1e-12:
            continue

        lambda1 = ((v1[1] - v2[1]) * (points[:, 0] - v2[0]) + (v2[0] - v1[0]) * (points[:, 1] - v2[1])) / detT
        lambda2 = ((v2[1] - v0[1]) * (points[:, 0] - v2[0]) + (v0[0] - v2[0]) * (points[:, 1] - v2[1])) / detT
        lambda0 = 1.0 - lambda1 - lambda2

        eps = 1e-9
        inside = (lambda0 >= -eps) & (lambda1 >= -eps) & (lambda2 >= -eps)
        if inside.any():
            cells_list.append(points[inside].astype(np.int64))

    if not cells_list:
        return None
    all_cells = np.concatenate(cells_list, axis=0)
    grid = np.zeros((img_size, img_size), dtype=bool)
    grid[all_cells[:, 1], all_cells[:, 0]] = True
    gy, gx = np.nonzero(grid)
    return np.stack([gx, gy], axis=1)


def _build_fill_mask(img_size, n, feather_px=0):
    """Generate fill face mask, prioritize selected faces, fallback to all faces, dilate n pixels, optional feather.
    Returns RGBA pixel buffer (black background opaque, white foreground, grayscale edge when feathered).
    """
    tris_sel = _collect_fill_triangles(selected_only=True)
    if tris_sel is not None and len(tris_sel) > 0:
        tris = tris_sel
        scope = "selected_faces"
    else:
        tris = _collect_fill_triangles(selected_only=False)
        if tris is None or len(tris) == 0:
            return None, "no_faces"
        scope = "all_faces"

    t = 1.0 / img_size
    cells = _rasterize_triangles(tris, t, img_size)
    if cells is None or len(cells) == 0:
        return None, "no_coverage"

    if n > 0:
        cells = _dilate_cells_grid(cells, n, img_size, img_size)
        if cells is None:
            return None, "no_coverage"

    # Build grayscale: foreground=1, background=0
    gray = np.zeros((img_size, img_size), dtype=np.float32)
    gray[cells[:, 1], cells[:, 0]] = 1.0

    if feather_px > 0:
        gray = _feather_alpha(gray, feather_px)

    # RGBA buffer: RGB=grayscale, Alpha=1
    buf = np.zeros((img_size * img_size, 4), dtype=np.float32)
    buf[:, 0] = gray.ravel()
    buf[:, 1] = gray.ravel()
    buf[:, 2] = gray.ravel()
    buf[:, 3] = 1.0
    return buf.ravel(), scope


def _build_expansion_mask(segs, img_size, n, feather_px=0):
    """Generate edge expansion mask, expansion area=white, rest=black, optional feather.
    Returns RGBA pixel buffer (black background opaque, white foreground, grayscale edge when feathered).
    """
    t = 1.0 / img_size
    ix0, iy0, ix1, iy1 = _expand_bbox(segs, t, n)
    cx0 = max(ix0, 0)
    cy0 = max(iy0, 0)
    cx1 = min(ix1, img_size - 1)
    cy1 = min(iy1, img_size - 1)
    if cx1 < cx0 or cy1 < cy0:
        return None
    W = cx1 - cx0 + 1
    H = cy1 - cy0 + 1

    cells = _rasterize_cells_uv(segs, t, cx0, cy0, W, H)
    gray = np.zeros((img_size, img_size), dtype=np.float32)
    if cells is not None and len(cells) > 0:
        dil = _dilate_cells_grid(cells, n, W, H)
        if dil is not None and len(dil) > 0:
            g_sub = np.zeros((H, W), dtype=bool)
            g_sub[dil[:, 1], dil[:, 0]] = True
            gray[cy0:cy0 + H, cx0:cx0 + W] = g_sub

    if feather_px > 0:
        gray = _feather_alpha(gray, feather_px)

    buf = np.zeros((img_size * img_size, 4), dtype=np.float32)
    buf[:, 0] = gray.ravel()
    buf[:, 1] = gray.ravel()
    buf[:, 2] = gray.ravel()
    buf[:, 3] = 1.0
    return buf.ravel()


# ------------------------------------------------------------
# Draw (UV Editor only)
# ------------------------------------------------------------
def _draw():
    p = _prefs()
    if p is not None and not p.enabled:
        return
    try:
        ctx = bpy.context
        area = ctx.area
        if area is None or area.ui_type != 'UV':
            return

        if _modal_running():
            return

        if ctx.mode != 'EDIT_MESH':
            return

        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if region is None:
            return
        v2d = region.view2d
        w, h = region.width, region.height
        if w <= 0 or h <= 0:
            return

        if _ST.shader is None:
            _ST.shader = gpu.shader.from_builtin('UNIFORM_COLOR')

        ps = getattr(p, 'pixel_size', 4.0) if p is not None else 4.0
        color = tuple(getattr(p, 'color', (1.0, 0.15, 0.15, 1.0))) if p is not None \
            else (1.0, 0.15, 0.15, 1.0)
        if p is not None and not getattr(p, 'use_theme_select_color', True):
            color_sel = tuple(getattr(p, 'select_color', (1.0, 0.5, 0.1, 1.0)))
        else:
            color_sel = _theme_select_color()
        color_expand = tuple(getattr(p, 'expand_color', (0.2, 0.6, 1.0, 0.35))) if p is not None \
            else (0.2, 0.6, 1.0, 0.35)

        if _ST.dirty:
            _ST.segs, _ST.segs_sel = _collect_boundary_segments()
            _ST.dirty = False
            _ST.view_key = None
            _rebuild_expand(p)

        if _ST.segs is None and _ST.segs_sel is None:
            return

        uv00 = v2d.region_to_view(0, 0)
        uv11 = v2d.region_to_view(w, h)
        du = uv11[0] - uv00[0]
        dv = uv11[1] - uv00[1]
        if du == 0 or dv == 0:
            return
        a = w / du
        b = -a * uv00[0]
        c = h / dv
        d = -c * uv00[1]

        key = (round(a, 6), round(b, 6), round(c, 6), round(d, 6), w, h, ps)
        if _ST.view_key != key:
            _ST.batch = None
            _ST.batch_sel = None
            if _ST.segs is not None:
                cells = _rasterize_cells(_ST.segs, a, b, c, d, w, h, ps)
                if cells is not None and len(cells) > 0:
                    _ST.batch = _cells_to_batch(cells, a, b, c, d, ps)
            if _ST.segs_sel is not None:
                cells_sel = _rasterize_cells(_ST.segs_sel, a, b, c, d, w, h, ps)
                if cells_sel is not None and len(cells_sel) > 0:
                    _ST.batch_sel = _cells_to_batch(cells_sel, a, b, c, d, ps)
            _ST.view_key = key

        if _ST.batch is None and _ST.batch_sel is None and _ST.batch_expand is None:
            return

        gpu.state.blend_set('ALPHA')
        _ST.shader.bind()
        if _ST.batch_expand is not None:
            _ST.shader.uniform_float("color", color_expand)
            _ST.batch_expand.draw(_ST.shader)
        if _ST.batch is not None:
            _ST.shader.uniform_float("color", color)
            _ST.batch.draw(_ST.shader)
        if _ST.batch_sel is not None:
            _ST.shader.uniform_float("color", color_sel)
            _ST.batch_sel.draw(_ST.shader)
        gpu.state.blend_set('NONE')
    except Exception:
        pass


# ------------------------------------------------------------
# Auto seam
# ------------------------------------------------------------
def _auto_seam_timer():
    p = _prefs()
    if p is None or not getattr(p, 'auto_seam', False):
        _ST.seam_pending = False
        return _SEAM_TIMER_INTERVAL
    if not _ST.seam_pending:
        return _SEAM_TIMER_INTERVAL

    ctx = bpy.context
    if ctx.mode != 'EDIT_MESH' or _modal_running():
        return _SEAM_TIMER_INTERVAL
    _ST.seam_pending = False

    try:
        for obj in ctx.objects_in_mode_unique_data:
            if obj.type != 'MESH':
                continue
            bm = bmesh.from_edit_mesh(obj.data)
            uv = bm.loops.layers.uv.active
            if uv is None:
                continue
            changed = False
            for e in bm.edges:
                if e.seam:
                    continue
                boundary = False
                for l in e.link_loops:
                    r = l.link_loop_radial_next
                    if r is l:
                        boundary = True
                        break
                    if (l[uv].uv != r.link_loop_next[uv].uv
                            or l.link_loop_next[uv].uv != r[uv].uv):
                        boundary = True
                        break
                if boundary:
                    e.seam = True
                    changed = True
            if changed:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    except Exception:
        pass
    return _SEAM_TIMER_INTERVAL


# ------------------------------------------------------------
# Copy image to clipboard (Windows priority, convert to RGB)
# ------------------------------------------------------------
def _copy_image_to_clipboard(img):
    """Copy Blender image to system clipboard, pasteable into Photoshop etc.
    Returns (success, message)
    """
    try:
        import platform
        if platform.system() != 'Windows':
            return False, "Windows clipboard only"

        try:
            import win32clipboard
            from PIL import Image
        except ImportError:
            return False, "Requires pywin32 and Pillow"

        w, h = img.size
        pixels = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        rgb = (pixels[:, :, :3] * 255).astype(np.uint8)
        rgb = np.flipud(rgb)

        pil_img = Image.fromarray(rgb, 'RGB')

        import io
        output = io.BytesIO()
        pil_img.save(output, format='BMP')
        data = output.getvalue()[14:]  # Remove BMP header
        output.close()

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()
        return True, "Image copied to clipboard"
    except Exception as e:
        return False, str(e)


# ------------------------------------------------------------
# Operator: Export Mask (supports fill/edge modes, copy path to clipboard)
# ------------------------------------------------------------
class PIXELUVBORDER_OT_export_mask(bpy.types.Operator):
    bl_idname = "pixel_uv_border.export_mask"
    bl_label = "Mask Export"
    bl_description = "Export the expansion area as a black/white PNG mask"

    filepath: StringProperty(
        name="File Path",
        subtype='FILE_PATH',
        default=""
    )
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.tga",
                                options={'HIDDEN'})

    img_size: EnumProperty(
        name="Image Size",
        items=_EXPAND_SIZE_ITEMS,
        default='2048'
    )
    expand_px: IntProperty(
        name="Dilation",
        default=8, min=0, max=64
    )
    fill_faces: BoolProperty(
        name="Fill Faces",
        description="Fill face interiors (selected faces priority)",
        default=False
    )
    copy_to_clipboard: BoolProperty(
        name="Copy to Clipboard",
        description="Copy the exported image to system clipboard",
        default=False
    )
    feather: BoolProperty(
        name="Feather Edge",
        description="Feather the mask edge",
        default=False
    )
    feather_px: IntProperty(
        name="Feather Pixels",
        description="Feather pixels",
        default=4, min=1, max=64
    )

    file_format: EnumProperty(
        name="Format",
        items=[
            ('PNG', "PNG (.png)", ""),
            ('JPEG', "JPEG (.jpg)", ""),
            ('TIFF', "TIFF (.tif)", ""),
            ('BMP', "BMP (.bmp)", ""),
            ('TARGA', "Targa (.tga)", ""),
        ],
        default='PNG'
    )
    color_mode: EnumProperty(
        name="Color",
        items=[('BW', "BW", ""), ('RGB', "RGB", ""), ('RGBA', "RGBA", "")],
        default='RGBA'
    )
    color_depth: EnumProperty(
        name="Depth",
        items=[('8', "8", ""), ('16', "16", "")],
        default='8'
    )
    compression: IntProperty(
        name="Compression",
        default=15, min=0, max=100, subtype='PERCENTAGE'
    )
    color_space: EnumProperty(
        name="Color Space",
        items=[('sRGB', "sRGB", ""), ('Non-Color', "Non-Color", "")],
        default='sRGB'
    )

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def invoke(self, context, event):
        p = _prefs()
        if p is not None:
            self.img_size = getattr(p, 'expand_image_size', '2048')
            self.expand_px = getattr(p, 'expand_px', 8)
            self.feather = getattr(p, 'feather', False)
            self.feather_px = getattr(p, 'feather_px', 4)
        if self.copy_to_clipboard:
            return self.execute(context)
        self.filepath = "uv_mask.png"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not self.filepath and not self.copy_to_clipboard:
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        p = _prefs()
        if p is None:
            self.report({'ERROR'}, "Addon preferences not found")
            return {'CANCELLED'}

        img_size = int(self.img_size)
        n = self.expand_px
        feather_px = self.feather_px if self.feather else 0

        # Sync preferences
        p.expand_image_size = self.img_size
        p.expand_px = n
        p.feather = self.feather
        p.feather_px = self.feather_px

        if self.fill_faces:
            pixels, scope = _build_fill_mask(img_size, n, feather_px)
            if pixels is None:
                self.report({'WARNING'}, f"No faces to fill ({scope})")
                return {'CANCELLED'}
        else:
            segs_normal, segs_sel = _collect_boundary_segments()
            if segs_sel is not None and len(segs_sel) > 0:
                segs = segs_sel
                scope = "selected_edges"
            elif segs_normal is not None and len(segs_normal) > 0:
                segs = segs_normal
                scope = "all_edges"
            else:
                self.report({'WARNING'}, "No UV borders found")
                return {'CANCELLED'}

            pixels = _build_expansion_mask(segs, img_size, n, feather_px)
            if pixels is None:
                self.report({'WARNING'}, "UV outside texture bounds")
                return {'CANCELLED'}

        img = bpy.data.images.new("UV_Mask", width=img_size, height=img_size, alpha=True)
        try:
            img.colorspace_settings.name = self.color_space
        except:
            pass
        img.pixels.foreach_set(pixels)
        img.update()

        if self.copy_to_clipboard:
            ok, msg = _copy_image_to_clipboard(img)
            if ok:
                self.report({'INFO'}, msg)
            else:
                self.report({'WARNING'}, f"Clipboard failed: {msg}")
            bpy.data.images.remove(img)
            return {'FINISHED'}
        else:
            ext = {'PNG': '.png', 'JPEG': '.jpg', 'TIFF': '.tif',
                   'BMP': '.bmp', 'TARGA': '.tga'}[self.file_format]
            fp = bpy.path.abspath(self.filepath)
            root, old_ext = os.path.splitext(fp)
            if old_ext.lower() != ext:
                fp = root + ext

            img.filepath_raw = fp
            img.file_format = self.file_format

            ims = context.scene.render.image_settings
            backup = (ims.file_format, ims.color_mode, ims.color_depth, ims.compression)
            ims.file_format = self.file_format
            ims.color_mode = 'RGB' if self.file_format == 'JPEG' and self.color_mode == 'BW' \
                else self.color_mode
            ims.color_depth = self.color_depth if self.file_format in {'PNG', 'TIFF'} else '8'
            ims.compression = self.compression
            img.save()
            ims.file_format, ims.color_mode, ims.color_depth, ims.compression = backup

            self.report({'INFO'}, f"Mask saved: {fp}")
            return {'FINISHED'}


# ------------------------------------------------------------
# UV Context Menu
# ------------------------------------------------------------
def _uv_context_menu_draw(self, context):
    layout = self.layout
    layout.separator()
    layout.operator_context = 'INVOKE_DEFAULT'
    op = layout.operator("pixel_uv_border.export_mask", text="Mask Export")
    op.fill_faces = False
    op.copy_to_clipboard = False


def _uv_context_menu():
    for name in ('IMAGE_MT_uvs_context_menu', 'IMAGEEDITOR_MT_uvs_context_menu'):
        cls = getattr(bpy.types, name, None)
        if cls is not None:
            return cls
    return None


# ------------------------------------------------------------
# Start / Stop
# ------------------------------------------------------------
def _start():
    if _ST.handler is None:
        _ST.handler = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_VIEW')
    _ST.dirty = True
    _ST.view_key = None
    _tag_all_uv_areas()


def _stop():
    if _ST.handler is not None:
        try:
            bpy.types.SpaceImageEditor.draw_handler_remove(_ST.handler, 'WINDOW')
        except Exception:
            pass
        _ST.handler = None
    _tag_all_uv_areas()


@persistent
def _on_depsgraph(scene, depsgraph):
    for u in depsgraph.updates:
        id_ = u.id
        if isinstance(id_, bpy.types.Mesh):
            _ST.dirty = True
            _ST.seam_pending = True
            return
        if (isinstance(id_, bpy.types.Object) and id_.type == 'MESH'
                and (u.is_updated_geometry or u.is_updated_transform)):
            _ST.dirty = True
            _ST.seam_pending = True
            return


@persistent
def _on_load_post(filepath):
    _ST.dirty = True
    _ST.view_key = None
    _ST.expand_key = None
    _ST.seam_pending = True


# ------------------------------------------------------------
# Preferences (stored in AddonPreferences, UI shown in N panel)
# ------------------------------------------------------------
def _on_visual_change(self, context):
    _ST.view_key = None
    _tag_all_uv_areas()


def _on_expand_param_change(self, context):
    _ST.dirty = True
    _tag_all_uv_areas()


def _on_auto_seam_change(self, context):
    if self.auto_seam:
        _ST.seam_pending = True


def _on_enabled_change(self, context):
    if self.enabled:
        _start()
    else:
        _stop()


class PixelUVBorderPrefs(bpy.types.AddonPreferences):
    # Use __package__ as required by Blender Extensions[reference:2]
    bl_idname = __package__

    enabled: BoolProperty(
        name="Enable Overlay",
        default=True,
        update=_on_enabled_change
    )
    auto_seam: BoolProperty(
        name="Auto Seam",
        default=False,
        update=_on_auto_seam_change
    )
    pixel_size: FloatProperty(
        name="Pixel Size",
        default=4.0, min=0.25, max=64.0, step=25, precision=2,
        update=_on_visual_change
    )
    color: FloatVectorProperty(
        name="Border Color",
        subtype='COLOR', size=4, default=(1.0, 0.15, 0.15, 1.0),
        min=0.0, max=1.0,
        update=_on_visual_change
    )
    use_theme_select_color: BoolProperty(
        name="Use Theme Select Color",
        default=True,
        update=_on_visual_change
    )
    select_color: FloatVectorProperty(
        name="Select Color",
        subtype='COLOR', size=4, default=(1.0, 0.5, 0.1, 1.0),
        min=0.0, max=1.0,
        update=_on_visual_change
    )
    expand_image_size: EnumProperty(
        name="Image Size",
        items=ITEMS_TEXTURE_SIZE,
        default="2048",
        update=_on_expand_param_change
    )
    expand_px: IntProperty(
        name="Expand Pixels",
        default=8, min=0, max=64,
        update=_on_expand_param_change
    )
    expand_color: FloatVectorProperty(
        name="Expand Color",
        subtype='COLOR', size=4, default=(0.2, 0.6, 1.0, 0.35),
        min=0.0, max=1.0,
        update=_on_visual_change
    )
    feather: BoolProperty(
        name="Feather Edge",
        description="Feather the mask edge",
        default=False,
        update=_on_visual_change
    )
    feather_px: IntProperty(
        name="Feather Pixels",
        description="Feather pixels",
        default=4, min=1, max=64,
        update=_on_visual_change
    )


# ------------------------------------------------------------
# N Panel (bl_category in English "Pixel Border", poll relaxed)
# ------------------------------------------------------------
class PIXELUVBORDER_PT_panel(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Pixel Border"
    bl_label = "Pixel Border"

    @classmethod
    def poll(cls, context):
        return context.space_data is not None

    def draw(self, context):
        p = _prefs()
        layout = self.layout
        if p is None:
            layout.label(text="Settings unavailable")
            return

        # ---- Face Mode (Fill) ----
        box = layout.box()
        row = box.row()
        row.label(text="Face Mode (Fill)", icon='FACESEL')
        col = box.column(align=True)
        row = col.row(align=True)
        op = row.operator("pixel_uv_border.export_mask", text="Export Mask")
        op.fill_faces = True
        op.copy_to_clipboard = False
        op = row.operator("pixel_uv_border.export_mask", text="Copy to Clipboard")
        op.fill_faces = True
        op.copy_to_clipboard = True

        layout.separator()

        # ---- Edge Mode (Border) ----
        box = layout.box()
        row = box.row()
        row.label(text="Edge Mode (Border)", icon='EDGESEL')
        col = box.column(align=True)
        row = col.row(align=True)
        op = row.operator("pixel_uv_border.export_mask", text="Export Mask")
        op.fill_faces = False
        op.copy_to_clipboard = False
        op = row.operator("pixel_uv_border.export_mask", text="Copy to Clipboard")
        op.fill_faces = False
        op.copy_to_clipboard = True

        layout.separator()

        # ---- Settings ----
        col = layout.column(align=True)
        col.prop(p, 'enabled', text="Enable Overlay", toggle=True)
        col.prop(p, 'auto_seam', text="Auto Seam", toggle=True)
        col.prop(p, 'pixel_size', text="Pixel Size", slider=True)
        col.prop(p, 'color', text="Border Color")
        row_sel = col.row(align=True)
        row_sel.prop(p, 'use_theme_select_color', text="Theme Select Color", toggle=True)
        sub = row_sel.row(align=True)
        sub.enabled = not p.use_theme_select_color
        sub.prop(p, 'select_color', text="Select Color")
        col.separator()
        row_sz = col.row(align=True)
        row_sz.prop(p, 'expand_image_size', text="Image Size")
        col.prop(p, 'expand_px', text="Expand Pixels", slider=True)
        row_ex = col.row(align=True)
        row_ex.prop(p, 'expand_color', text="Expand Color")

        # ---- Feather Settings ----
        col.separator()
        col.prop(p, 'feather', text="Feather Edge", toggle=True)
        sub = col.column()
        sub.enabled = p.feather
        sub.prop(p, 'feather_px', text="Feather Pixels", slider=True)


classes = (
    PixelUVBorderPrefs,
    PIXELUVBORDER_OT_export_mask,
    PIXELUVBORDER_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    if not bpy.app.timers.is_registered(_auto_seam_timer):
        bpy.app.timers.register(_auto_seam_timer, first_interval=_SEAM_TIMER_INTERVAL, persistent=True)
    menu = _uv_context_menu()
    if menu is not None:
        menu.append(_uv_context_menu_draw)
    p = _prefs()
    if p is None or p.enabled:
        _start()


def unregister():
    _stop()
    try:
        menu = _uv_context_menu()
        if menu is not None:
            menu.remove(_uv_context_menu_draw)
        if bpy.app.timers.is_registered(_auto_seam_timer):
            bpy.app.timers.unregister(_auto_seam_timer)
        if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
        if _on_load_post in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(_on_load_post)
    except Exception:
        pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()