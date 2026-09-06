# Pixel UV Border

Pixelated UV border overlay for Blender's UV Editor, with selection highlight, texel-accurate bake expansion preview, mask export, and Maya-style auto seam marking.

**Pixel UV Border** overlays every genuinely split UV border in the UV Editor as a blocky pixel border — completely independent of seam marks, so overlapping split UVs are detected too. The border keeps a constant screen-space pixel size and stays razor sharp at any zoom level.

## Features

- **Pixel Border** — Constant screen-size pixel blocks drawn along all split UV borders. Adjustable pixel size and color.
- **Selection Highlight** — Selected border edges are tinted with the theme's select color (or a custom color).
- **Bake Expansion Preview** — Live preview of outward dilation in *true texel size* at your chosen texture resolution (512–8192, 8 px by default). The preview scales with the UV view: **what you see is what you bake**.
- **Auto Seam (Maya-style)** — Edges with split UVs are automatically marked as seams. Additive only: it never removes existing seams. Toggleable.
- **Mask Export** — Render the dilation / fill area to a black-and-white mask (PNG, JPEG, TIFF, BMP, TGA) with optional feathering, plus one-click *Copy to Clipboard* (Windows) for pasting straight into Photoshop or Substance.
- **Zero-Overhead Navigation** — Zooming and panning cost nothing. Border data is rebuilt only when mesh geometry or UVs change, and the expansion area is decoupled from the viewport.

## Requirements

- Blender **4.2 or newer** (works on 5.x)
- NumPy — bundled with Blender
- Optional: SciPy for fast feathering on export (a pure-NumPy fallback is included)

## Installation

1. Download **`pixel_uv_border_v1.0.1.zip`** from the [Releases](../../releases) page, or use the `pixel_uv_border` folder in this repository.
2. In Blender: **Edit → Preferences → Get Extensions → Install from Disk** (Blender 4.2+) or **Add-ons → Install** and select the zip / the `pixel_uv_border` folder.
3. Enable **Pixel UV Border** in the add-ons list.

## Usage

1. Enter **Edit Mode** on a mesh with UVs.
2. Open the **UV Editor** and press **N** → **Pixel Border** tab.

### Panel Reference

| Section | Controls |
| --- | --- |
| **Face Mode (Fill)** | Export Mask / Copy to Clipboard for filled face interiors (selected faces take priority) |
| **Edge Mode (Border)** | Export Mask / Copy to Clipboard for boundary edge expansion |
| **Settings** | Enable Overlay, Auto Seam, Pixel Size, Border Color, Theme Select Color / Select Color, Image Size, Expand Pixels, Expand Color, Feather Edge / Feather Pixels |

### Notes

- The **pixel border** shows *UV-disconnected* edges — not Blender's seam marks. Two island borders overlapping at the same location are shown as two borders.
- **Auto Seam** only *adds* seams where UVs are disconnected; existing seams are never touched.
- The **expansion preview** works in real texels: at 2048 resolution with 8 px dilation, the highlighted band is exactly 8 texels wide, and it zooms together with the UV view.
- **Mask export** prioritizes selected edges (Edge Mode) or selected faces (Face Mode), and falls back to all borders / all faces when nothing is selected.

## Performance Notes

- Depsgraph-filtered rebuilds: only mesh geometry / transform edits mark the data dirty.
- Rasterized samples are deduplicated via bitmaps — no per-frame sorting.
- The expansion area is computed once per parameter change from the UV bounding box and is completely viewport-independent.

## Location

UV Editor → Sidebar (**N**) → **Pixel Border** tab. Works on meshes in Edit Mode.

## Support

- Report bugs or request features: <https://github.com/LENS6/pixel_uv_border/issues>
- Source code and releases: <https://github.com/LENS6/pixel_uv_border>

## License

GPL-3.0-or-later
