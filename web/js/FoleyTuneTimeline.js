import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PALETTE = {
    red: "#e74c3c", blue: "#3498db", green: "#2ecc71",
    yellow: "#f1c40f", purple: "#9b59b6", orange: "#e67e22",
};

function resolveColor(entry) {
    if (!entry) return PALETTE.blue;
    return PALETTE[entry.color] || PALETTE.blue;
}

const RULER_H = 22;
const THUMB_H = 60;
const TRACK_H = 48;
const PREVIEW_H = 220;    // frame-preview area below the timeline (scrub player)
const HANDLE_W = 6;       // visual width of the resize-handle indicator
const EDGE_GRAB = 10;     // hit tolerance around a segment edge (px, straddles it)
const CANVAS_H = RULER_H + TRACK_H;
const TOTAL_H = THUMB_H + CANVAS_H + PREVIEW_H;
const MIN_SEG_FRAMES = 1;   // shortest segment, in frames
const DEFAULT_FPS = 30;     // fallback when the video fps is unknown
const SNAP_SEC = 8;         // segment boundaries snap to this grid (hold Shift for fine)
const PLAYHEAD_COLOR = "#ffcc00";

class TimelineEditor {
    constructor(node, container) {
        this.node = node;
        this.container = container;

        this.duration = 0;
        this.entries = [];
        this.segments = [];
        this.selectedIdx = -1;
        this.hoverIdx = -1;
        this.hoverEdge = null; // "left" | "right" | null
        this.drag = null;
        this._commitTimer = null;
        this.playhead = 0;       // scrub position in seconds (frame-snapped)
        this.videoPath = "";
        this.fps = DEFAULT_FPS;
        this._frameTimer = null;

        // Find and hide the segments_json widget. It's a multiline STRING, so it
        // owns a DOM <textarea>; flipping .type does NOT remove that element —
        // without hiding it explicitly it stays visible and inflates node height.
        this.segWidget = node.widgets?.find(w => w.name === "segments_json");
        if (this.segWidget) {
            this.segWidget.computeSize = () => [0, -4];
            this.segWidget.type = "converted-widget";
            // Hide the DOM <textarea> too (keeps the value serialized, unlike
            // .hidden which can drop it from widgets_values).
            const el = this.segWidget.inputEl
                || this.segWidget.element
                || (this.segWidget.options && this.segWidget.options.element);
            if (el && el.style) el.style.display = "none";
        }

        // Re-render the crossfade band live when the crossfade_frames widget changes.
        const xfWidget = node.widgets?.find(w => w.name === "crossfade_frames");
        if (xfWidget) {
            const prev = xfWidget.callback;
            xfWidget.callback = (...args) => {
                if (prev) prev.apply(xfWidget, args);
                this.render();
            };
        }

        this._buildDOM();
        this._bindEvents();

        this._resizeObserver = new ResizeObserver(() => this._onResize());
        this._resizeObserver.observe(container);
    }

    _buildDOM() {
        // Thumbnail strip
        this.thumbStrip = document.createElement("div");
        this.thumbStrip.style.cssText =
            `width:100%;height:${THUMB_H}px;background:#111;overflow:hidden;position:relative;`;
        this.thumbImg = document.createElement("img");
        // Stretch the full-duration sprite across the strip so thumbnail x maps
        // linearly to time (aligns with the ruler below).
        this.thumbImg.style.cssText =
            "width:100%;height:100%;position:absolute;left:0;top:0;object-fit:fill;";
        this.thumbStrip.appendChild(this.thumbImg);
        this.container.appendChild(this.thumbStrip);

        // Canvas for ruler + segments
        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = `width:100%;height:${CANVAS_H}px;display:block;`;
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        // Scrub player: a larger frame at the playhead, for precise alignment.
        // Click/drag the ruler to move the playhead.
        this.preview = document.createElement("div");
        this.preview.style.cssText =
            `width:100%;height:${PREVIEW_H}px;background:#0a0a12;position:relative;` +
            `display:flex;align-items:center;justify-content:center;overflow:hidden;`;
        this.previewImg = document.createElement("img");
        this.previewImg.style.cssText =
            "max-width:100%;max-height:100%;object-fit:contain;display:block;";
        this.preview.appendChild(this.previewImg);
        this.previewLabel = document.createElement("div");
        this.previewLabel.style.cssText =
            "position:absolute;left:6px;top:4px;font:11px monospace;color:#ffcc00;" +
            "background:rgba(0,0,0,0.55);padding:1px 5px;border-radius:3px;pointer-events:none;";
        this.previewLabel.textContent = "scrub the ruler ▸";
        this.preview.appendChild(this.previewLabel);
        this.container.appendChild(this.preview);
    }

    _bindEvents() {
        const c = this.canvas;
        c.addEventListener("pointerdown", (e) => this._onPointerDown(e));
        c.addEventListener("pointermove", (e) => this._onPointerMove(e));
        c.addEventListener("pointerup", (e) => this._onPointerUp(e));
        c.addEventListener("pointerleave", () => this._onPointerLeave());
        c.addEventListener("dblclick", (e) => this._onDblClick(e));
        c.addEventListener("contextmenu", (e) => this._onContextMenu(e));
    }

    // --- Coordinate helpers ---

    _dpr() { return window.devicePixelRatio || 1; }

    _canvasW() { return this.canvas.clientWidth; }

    _secToX(sec) {
        return this.duration > 0 ? (sec / this.duration) * this._canvasW() : 0;
    }

    _xToSec(x) {
        // Snap to the nearest frame boundary.
        return this._snapSec(this._xToSecRaw(x));
    }

    _xToSecRaw(x) {
        return this.duration > 0 ? Math.max(0, Math.min(this.duration, (x / this._canvasW()) * this.duration)) : 0;
    }

    // --- Frame helpers (the timeline works in frames at the video fps) ---

    _frame(sec) { return Math.round(sec * this.fps); }       // sec -> frame index
    _snapSec(sec) { return this._frame(sec) / this.fps; }    // snap sec to a frame
    _minSeg() { return MIN_SEG_FRAMES / this.fps; }          // shortest segment, sec

    // Snap a segment boundary to the SNAP_SEC grid so chunks land on round
    // boundaries (e.g. 8s). The clip's true ends (0 / duration) are always
    // candidates so the first/last chunk can reach the real edge. Hold Shift
    // (fine=true) to drop back to per-frame snapping for an off-grid cut.
    _snapSegSec(sec, fine) {
        if (fine) return this._snapSec(sec);
        const grid = Math.round(sec / SNAP_SEC) * SNAP_SEC;
        let best = grid;
        for (const c of [0, this.duration]) {
            if (Math.abs(sec - c) < Math.abs(sec - best)) best = c;
        }
        return this._snapSec(Math.max(0, Math.min(this.duration, best)));
    }

    // Nearest segment to the right/left of `seg` (by facing edge). Used so a
    // dragged edge pushes its neighbour instead of overlapping it.
    _neighborRight(seg) {
        let best = null;
        for (const o of this.segments) {
            if (o === seg || o.start_sec < seg.start_sec) continue;
            if (!best || o.start_sec < best.start_sec) best = o;
        }
        return best;
    }
    _neighborLeft(seg) {
        let best = null;
        for (const o of this.segments) {
            if (o === seg || o.end_sec > seg.end_sec) continue;
            if (!best || o.end_sec > best.end_sec) best = o;
        }
        return best;
    }

    _pointerPos(e) {
        // getBoundingClientRect reflects LiteGraph's zoom (CSS transform), but
        // _secToX/_canvasW use the canvas's logical width. Rescale screen px to
        // logical px so hit-testing (esp. the ~10px edge zone) is correct at any
        // zoom — otherwise edges become ungrabbable when zoomed in/out.
        const r = this.canvas.getBoundingClientRect();
        const sx = r.width ? this.canvas.clientWidth / r.width : 1;
        const sy = r.height ? this.canvas.clientHeight / r.height : 1;
        return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
    }

    _hitTest(x, y) {
        if (y < RULER_H || y > RULER_H + TRACK_H) return { idx: -1, edge: null };
        // First pass: edges win over bodies, and the grab zone straddles the
        // boundary (a few px outside the segment counts too) so handles are
        // actually grabbable. Reverse order = topmost segment first.
        for (let i = this.segments.length - 1; i >= 0; i--) {
            const seg = this.segments[i];
            const left = this._secToX(seg.start_sec);
            const right = this._secToX(seg.end_sec);
            const nearLeft = Math.abs(x - left) <= EDGE_GRAB;
            const nearRight = Math.abs(x - right) <= EDGE_GRAB;
            if (nearLeft && nearRight) {
                // Tiny segment — pick the closer edge.
                return { idx: i, edge: (x - left <= right - x) ? "left" : "right" };
            }
            if (nearLeft) return { idx: i, edge: "left" };
            if (nearRight) return { idx: i, edge: "right" };
        }
        // Second pass: clicking the body = move.
        for (let i = this.segments.length - 1; i >= 0; i--) {
            const seg = this.segments[i];
            const left = this._secToX(seg.start_sec);
            const right = this._secToX(seg.end_sec);
            if (x >= left && x <= right) return { idx: i, edge: null };
        }
        return { idx: -1, edge: null };
    }

    // --- Events ---

    _onPointerDown(e) {
        // Only the left button starts a drag/selection. A right-button down
        // would set this.drag, then contextmenu deletes the segment, leaving
        // this.drag pointing at a removed index (crash on next move).
        if (e.button !== 0) return;
        e.stopPropagation();
        const { x, y } = this._pointerPos(e);

        // Ruler band = scrub the playhead (frame preview below).
        if (y < RULER_H) {
            this.drag = { type: "scrub" };
            this.canvas.setPointerCapture(e.pointerId);
            this._setPlayhead(this._xToSecRaw(x));
            return;
        }

        const hit = this._hitTest(x, y);

        if (hit.idx < 0) {
            this.selectedIdx = -1;
            this.render();
            return;
        }

        this.selectedIdx = hit.idx;
        const seg = this.segments[hit.idx];

        if (hit.edge) {
            this.drag = {
                type: `resize-${hit.edge}`,
                idx: hit.idx,
                startX: x,
                origStart: seg.start_sec,
                origEnd: seg.end_sec,
            };
        } else {
            this.drag = {
                type: "move",
                idx: hit.idx,
                startX: x,
                origStart: seg.start_sec,
                origEnd: seg.end_sec,
            };
        }
        this.canvas.setPointerCapture(e.pointerId);
        this.render();
    }

    _onPointerMove(e) {
        e.stopPropagation();
        const { x, y } = this._pointerPos(e);

        if (this.drag) {
            if (this.drag.type === "scrub") {
                this._setPlayhead(this._xToSecRaw(x));
                return;
            }
            const seg = this.segments[this.drag.idx];
            if (!seg) { this.drag = null; return; }  // segment removed mid-drag

            if (this.drag.type === "move") {
                // Raw (unsnapped) pixel delta keeps dragging smooth; snap the
                // final position once.
                const dSecRaw = this._canvasW() > 0
                    ? ((x - this.drag.startX) / this._canvasW()) * this.duration : 0;
                const len = this.drag.origEnd - this.drag.origStart;
                let ns = this._snapSegSec(Math.max(0, Math.min(this.duration - len, this.drag.origStart + dSecRaw)), e.shiftKey);
                // Block overlap: keep the segment inside the room between its
                // neighbours (pushes nothing — a move just can't ride over them).
                let lo = 0, hi = this.duration - len;
                for (const o of this.segments) {
                    if (o === seg) continue;
                    if (o.end_sec <= this.drag.origStart + 1e-6) lo = Math.max(lo, o.end_sec);
                    else if (o.start_sec >= this.drag.origEnd - 1e-6) hi = Math.min(hi, o.start_sec - len);
                }
                if (hi < lo) hi = lo;
                ns = Math.max(lo, Math.min(hi, ns));
                seg.start_sec = ns;
                seg.end_sec = ns + len;
            } else if (this.drag.type === "resize-left") {
                // Edge snaps to the SNAP_SEC grid (Shift = per-frame), clamped to [0,dur].
                let ns = Math.min(seg.end_sec - this._minSeg(), this._snapSegSec(this._xToSecRaw(x), e.shiftKey));
                // Push (or roll, when already touching) the left neighbour's
                // right edge instead of overlapping it.
                const nb = this._neighborLeft(seg);
                if (nb && (ns < nb.end_sec || Math.abs(seg.start_sec - nb.end_sec) < 1e-6)) {
                    ns = Math.max(ns, nb.start_sec + this._minSeg());  // don't crush it
                    nb.end_sec = ns;
                }
                seg.start_sec = ns;
            } else if (this.drag.type === "resize-right") {
                let ns = Math.max(seg.start_sec + this._minSeg(), this._snapSegSec(this._xToSecRaw(x), e.shiftKey));
                const nb = this._neighborRight(seg);
                if (nb && (ns > nb.start_sec || Math.abs(seg.end_sec - nb.start_sec) < 1e-6)) {
                    ns = Math.min(ns, nb.end_sec - this._minSeg());
                    nb.start_sec = ns;
                }
                seg.end_sec = ns;
            }
            this._commitDebounced();
            this.render();
            return;
        }

        // Hover feedback
        const hit = this._hitTest(x, y);
        const changed = hit.idx !== this.hoverIdx || hit.edge !== this.hoverEdge;
        this.hoverIdx = hit.idx;
        this.hoverEdge = hit.edge;

        if (y < RULER_H) {
            this.canvas.style.cursor = "ew-resize";  // ruler = scrub zone
        } else if (hit.edge) {
            this.canvas.style.cursor = "ew-resize";
        } else if (hit.idx >= 0) {
            this.canvas.style.cursor = "grab";
        } else {
            this.canvas.style.cursor = "default";
        }

        if (changed) this.render();
    }

    _onPointerUp(e) {
        if (this.drag) {
            const wasScrub = this.drag.type === "scrub";
            this.drag = null;
            this.canvas.releasePointerCapture(e.pointerId);
            if (!wasScrub) this._commitFlush();  // scrub doesn't change segments
            this.render();
        }
    }

    // --- Scrub player ---

    _setPlayhead(sec) {
        // Snap the playhead to a frame boundary too.
        this.playhead = this._snapSec(Math.max(0, Math.min(this.duration || 0, sec)));
        this.previewLabel.textContent = `frame ${this._frame(this.playhead)}  ·  ${this.playhead.toFixed(2)}s`;
        this._loadPreviewFrame();
        this.render();
    }

    _loadPreviewFrame() {
        if (!this.videoPath) return;
        // Debounce so dragging the playhead doesn't spawn an ffmpeg per pixel.
        if (this._frameTimer) clearTimeout(this._frameTimer);
        this._frameTimer = setTimeout(() => {
            this._frameTimer = null;
            this.previewImg.src = api.apiURL(
                "/foleytune/timeline_frame?video_path=" + encodeURIComponent(this.videoPath) +
                "&t=" + this.playhead.toFixed(2));
        }, 80);
    }

    _onPointerLeave() {
        if (!this.drag) {
            this.hoverIdx = -1;
            this.hoverEdge = null;
            this.canvas.style.cursor = "default";
            this.render();
        }
    }

    _onDblClick(e) {
        e.stopPropagation();
        e.preventDefault();
        if (!this.entries.length || this.duration <= 0) return;

        const { x } = this._pointerPos(e);
        const sec = this._xToSec(x);

        let entryIdx = 0;
        if (this.entries.length > 1) {
            const labels = this.entries.map((en, i) => `${i}: ${en.label}`).join("\n");
            const input = prompt(`Select LoRA entry:\n${labels}`, "0");
            if (input === null) return;
            entryIdx = parseInt(input, 10);
            if (entryIdx < 0 || entryIdx >= this.entries.length) return;
        }

        const startSec = this._snapSec(Math.max(0, sec - 2));
        const endSec = this._snapSec(Math.min(this.duration, sec + 2));
        this.segments.push({
            entry_id: this.entries[entryIdx].id,  // stable ref (survives reorder)
            entry_index: entryIdx,                // fallback for older graphs
            start_sec: startSec,
            end_sec: endSec,
            strength: this.entries[entryIdx].strength,
        });
        this.segments.sort((a, b) => a.start_sec - b.start_sec);
        this.selectedIdx = this.segments.length - 1;
        this._commitFlush();
        this.render();
    }

    _onContextMenu(e) {
        e.preventDefault();
        e.stopPropagation();
        const { x, y } = this._pointerPos(e);
        const hit = this._hitTest(x, y);
        if (hit.idx >= 0) {
            this.segments.splice(hit.idx, 1);
            this.selectedIdx = -1;
            this.drag = null;  // any in-flight drag now points at a stale index
            this._commitFlush();
            this.render();
        }
    }

    // --- State sync ---

    _commitDebounced() {
        if (this._commitTimer) clearTimeout(this._commitTimer);
        this._commitTimer = setTimeout(() => {
            this._commitTimer = null;
            this._commitFlush();
        }, 120);
    }

    _commitFlush() {
        if (this._commitTimer) {
            clearTimeout(this._commitTimer);
            this._commitTimer = null;
        }
        const json = JSON.stringify(this.segments);
        // Backend reads the widget value (positional in widgets_values).
        if (this.segWidget) this.segWidget.value = json;
        // Mirror into node.properties, which serializes BY NAME — so the data
        // survives widget reordering / node-definition changes that would
        // otherwise scramble the positional widget slot and scrap the segments.
        this.node.properties = this.node.properties || {};
        this.node.properties.foleytune_segments = json;
    }

    // --- Resize ---

    _onResize() {
        this._enforceWidth();
        this._updateCanvasSize();
        this.render();
    }

    // A ComfyUI-wide bug collapses preview/DOM-widget elements to ~half width
    // when the node is selected/re-laid-out (VHS Load Video shows it too). Guard:
    // if our element is narrower than a sibling widget that stayed full width,
    // force it back. The _enforcing flag breaks the ResizeObserver feedback loop.
    _referenceWidth() {
        let w = 0;
        for (const widget of (this.node.widgets || [])) {
            const el = widget.inputEl || widget.element;
            if (el && el !== this.container && el.offsetWidth > w) w = el.offsetWidth;
        }
        if (!w && this.container.parentElement) w = this.container.parentElement.clientWidth;
        return w;
    }

    _enforceWidth() {
        if (this._enforcing) return;
        const ref = this._referenceWidth();
        if (ref > 0 && this.container.clientWidth < ref - 2) {
            this._enforcing = true;
            this.container.style.width = ref + "px";
            requestAnimationFrame(() => { this._enforcing = false; });
        }
    }

    _updateCanvasSize() {
        const dpr = this._dpr();
        const w = this._canvasW();
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(CANVAS_H * dpr);
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // Scale thumbnail to match
        this.thumbImg.style.width = this.thumbStrip.clientWidth + "px";
    }

    // --- Rendering ---

    render() {
        const ctx = this.ctx;
        const w = this._canvasW();
        if (w <= 0 || this.duration <= 0) return;

        // Keep the backing store synced to the current layout width. A LiteGraph
        // zoom changes the displayed width without firing the ResizeObserver, so
        // otherwise drawing coords (_secToX) drift vs the mouse, growing to the
        // right — most visible on the playhead.
        if (this.canvas.width !== Math.round(w * this._dpr())) {
            this._updateCanvasSize();
        }

        ctx.clearRect(0, 0, w, CANVAS_H);

        this._drawRuler(ctx, w);
        this._drawTrack(ctx, w);
        this._drawSegments(ctx, w);
        this._drawCrossfades(ctx, w);
        this._drawChunkPlan(ctx, w);
        this._drawPlayhead(ctx, w);
    }

    // The REAL generation chunk plan — how align_chunks_to_schedule() (utils.py)
    // tiles the whole clip: boundaries are the clip ends + every segment edge,
    // and any region longer than the chunk window is sub-split evenly. Assumes
    // the sampler's default chunk_duration = SNAP_SEC (8s). Mirrors the backend
    // so the timeline shows exactly what will generate — gaps and long-segment
    // splits included. Returns [[cs, ce], ...].
    _chunkPlan() {
        const dur = this.duration;
        if (!(dur > 0) || !this.segments.length) return [];
        const W = SNAP_SEC;
        const bset = new Set([0, dur]);
        for (const s of this.segments) {
            bset.add(Math.max(0, Math.min(dur, s.start_sec)));
            bset.add(Math.max(0, Math.min(dur, s.end_sec)));
        }
        const bounds = [...bset].sort((a, b) => a - b);
        const keep = [];
        for (let i = 0; i < bounds.length - 1; i++) {
            const cs = bounds[i], ce = bounds[i + 1];
            const span = ce - cs;
            if (span < 1e-3) continue;
            if (span <= W + 1e-6) { keep.push([cs, ce]); continue; }
            const n = Math.ceil(span / W), step = span / n;
            for (let k = 0; k < n; k++) {
                keep.push([cs + k * step, k === n - 1 ? ce : cs + (k + 1) * step]);
            }
        }
        return keep;
    }

    _coveredBySegment(cs, ce) {
        const mid = (cs + ce) / 2;
        return this.segments.some(s => mid >= s.start_sec - 1e-6 && mid <= s.end_sec + 1e-6);
    }

    // Overlay the chunk plan: a dashed cut at each chunk boundary, a "cN" index
    // per chunk, and a diagonal hatch over GAP chunks (no segment — they still
    // generate as base-model audio, which is the surprise in the logs).
    _drawChunkPlan(ctx, w) {
        const plan = this._chunkPlan();
        if (!plan.length) return;
        const yTop = RULER_H;
        const yBot = RULER_H + TRACK_H;
        ctx.save();
        for (let i = 0; i < plan.length; i++) {
            const [cs, ce] = plan[i];
            const x1 = this._secToX(cs), x2 = this._secToX(ce);
            if (!this._coveredBySegment(cs, ce)) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(x1, yTop, x2 - x1, TRACK_H);
                ctx.clip();
                ctx.strokeStyle = "rgba(255,140,0,0.45)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                for (let xx = x1 - TRACK_H; xx < x2; xx += 7) {
                    ctx.moveTo(xx, yBot); ctx.lineTo(xx + TRACK_H, yTop);
                }
                ctx.stroke();
                ctx.restore();
            }
            ctx.fillStyle = "rgba(255,205,90,0.95)";
            ctx.font = "bold 9px monospace";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.fillText(`c${i}`, x1 + 3, yTop + 2);
        }
        // Dashed cuts at interior chunk boundaries.
        ctx.strokeStyle = "rgba(255,175,45,0.9)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        for (let i = 1; i < plan.length; i++) {
            const x = this._secToX(plan[i][0]);
            ctx.beginPath();
            ctx.moveTo(x, yTop); ctx.lineTo(x, yBot);
            ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.restore();
    }

    // crossfade_frames widget value (the backend blends adjacent segments over
    // this many frames at the video fps; 0 = hard cuts).
    _crossfadeFrames() {
        const wgt = this.node?.widgets?.find(w => w.name === "crossfade_frames");
        const v = wgt ? Number(wgt.value) : 0;
        return Number.isFinite(v) && v > 0 ? v : 0;
    }

    // Draw a fade band at each touching segment boundary: a left->right colour
    // gradient (prev segment -> next segment) with an ✕ hatch, width = crossfade,
    // centred on the cut. Mirrors the backend's equal-power blend.
    _drawCrossfades(ctx, w) {
        const xfFrames = this._crossfadeFrames();
        if (!xfFrames || this.segments.length < 2) return;
        const xfSec = xfFrames / this.fps;
        const y = RULER_H + 3;
        const h = TRACK_H - 6;
        const segs = [...this.segments].sort((a, b) => a.start_sec - b.start_sec);
        for (let i = 0; i < segs.length - 1; i++) {
            const B = segs[i].end_sec;
            // only at adjacent (touching) boundaries — skip gaps
            if (Math.abs(segs[i + 1].start_sec - B) > 1e-3) continue;
            const x1 = this._secToX(Math.max(segs[i].start_sec, B - xfSec / 2));
            const x2 = this._secToX(Math.min(segs[i + 1].end_sec, B + xfSec / 2));
            if (x2 - x1 < 1) continue;
            const cL = resolveColor(this._entryFor(segs[i]));
            const cR = resolveColor(this._entryFor(segs[i + 1]));
            const grad = ctx.createLinearGradient(x1, 0, x2, 0);
            grad.addColorStop(0, cL);
            grad.addColorStop(1, cR);
            ctx.save();
            ctx.globalAlpha = 0.55;
            ctx.fillStyle = grad;
            ctx.fillRect(x1, y, x2 - x1, h);
            // ✕ hatch reads as "crossfade"
            ctx.globalAlpha = 0.5;
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x1, y); ctx.lineTo(x2, y + h);
            ctx.moveTo(x1, y + h); ctx.lineTo(x2, y);
            ctx.stroke();
            ctx.restore();
        }
    }

    _drawPlayhead(ctx) {
        if (this.duration <= 0) return;
        const x = this._secToX(this.playhead);
        ctx.fillStyle = PLAYHEAD_COLOR;
        ctx.fillRect(x - 0.75, 0, 1.5, CANVAS_H);
        // Triangle handle at the top of the ruler.
        ctx.beginPath();
        ctx.moveTo(x - 5, 0);
        ctx.lineTo(x + 5, 0);
        ctx.lineTo(x, 7);
        ctx.closePath();
        ctx.fill();
    }

    _totalFrames() { return Math.max(1, Math.round(this.duration * this.fps)); }

    _tickStepFrames(w) {
        // Frame interval that keeps tick labels at least ~46px apart.
        const pxPerFrame = w / this._totalFrames();
        const rawStep = 46 / pxPerFrame;
        const nice = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 3000, 6000];
        return nice.find(s => s >= rawStep) || 6000;
    }

    _drawRuler(ctx, w) {
        ctx.fillStyle = "#1a1a2e";
        ctx.fillRect(0, 0, w, RULER_H);
        ctx.font = "10px monospace";

        const totalF = this._totalFrames();
        const stepF = this._tickStepFrames(w);

        for (let f = 0; f <= totalF; f += stepF) {
            const x = this._secToX(f / this.fps);
            const isMajor = (f % (stepF * 5)) === 0;

            ctx.fillStyle = "#555";
            ctx.fillRect(x, 0, 1, isMajor ? RULER_H - 4 : RULER_H - 10);

            if (isMajor || (w / totalF) * stepF > 50) {
                ctx.fillStyle = "#999";
                ctx.fillText(`${f}`, x + 3, RULER_H - 3);
            }
        }

        // Bottom border
        ctx.fillStyle = "#333";
        ctx.fillRect(0, RULER_H - 1, w, 1);
    }

    _drawTrack(ctx, w) {
        ctx.fillStyle = "#0e0e1a";
        ctx.fillRect(0, RULER_H, w, TRACK_H);

        // Grid lines at each ruler tick (frames) for alignment reference.
        ctx.fillStyle = "#1a1a2e";
        const totalF = this._totalFrames();
        const stepF = this._tickStepFrames(w);
        for (let f = 0; f <= totalF; f += stepF) {
            ctx.fillRect(this._secToX(f / this.fps), RULER_H, 1, TRACK_H);
        }

        // Brighter lines at each SNAP_SEC boundary — these are the snap targets.
        ctx.fillStyle = "#3a5a8c";
        for (let s = SNAP_SEC; s < this.duration; s += SNAP_SEC) {
            ctx.fillRect(this._secToX(s), RULER_H, 1, TRACK_H);
        }
    }

    _drawSegments(ctx, w) {
        const dragIdx = this.drag?.idx ?? -1;

        // Two-pass: non-dragged first, then dragged on top
        const order = this.segments.map((_, i) => i)
            .sort((a, b) => (a === dragIdx ? 1 : 0) - (b === dragIdx ? 1 : 0));

        for (const i of order) {
            this._drawSegment(ctx, i);
        }
    }

    _entryFor(seg) {
        // Resolve by stable id first (survives chain reorder), then by index.
        if (seg.entry_id != null) {
            const byId = this.entries.find(e => e && e.id === seg.entry_id);
            if (byId) return byId;
        }
        return this.entries[seg.entry_index] || {};
    }

    _drawSegment(ctx, i) {
        const seg = this.segments[i];
        const entry = this._entryFor(seg);
        const color = resolveColor(entry);
        // ✎ marks an entry with a per-segment prompt (see entry node).
        const hasPrompt = !!(entry.prompt && entry.prompt.trim());
        const label = (hasPrompt ? "✎ " : "") + (entry.label || "LoRA");

        const x = this._secToX(seg.start_sec);
        const x2 = this._secToX(seg.end_sec);
        const segW = x2 - x;
        const y = RULER_H + 3;
        const h = TRACK_H - 6;
        const r = 4;

        const isSelected = i === this.selectedIdx;
        const isHover = i === this.hoverIdx && !this.drag;
        const isDragging = this.drag?.idx === i;

        // Background fill
        ctx.save();
        ctx.globalAlpha = isDragging ? 0.9 : isHover ? 0.85 : 0.7;
        ctx.fillStyle = color;
        this._roundRect(ctx, x, y, segW, h, r);
        ctx.fill();
        ctx.restore();

        // Border
        ctx.save();
        ctx.strokeStyle = isDragging ? "#ffd700" : isSelected ? "#fff" : color;
        ctx.lineWidth = isSelected || isDragging ? 2 : 1;
        this._roundRect(ctx, x, y, segW, h, r);
        ctx.stroke();
        ctx.restore();

        // Label text
        if (segW > 20) {
            ctx.save();
            ctx.fillStyle = "#fff";
            ctx.font = "bold 11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.shadowColor = "#000";
            ctx.shadowBlur = 3;

            const centerX = x + segW / 2;
            const centerY = y + h / 2;

            // Clip to segment bounds
            ctx.beginPath();
            this._roundRect(ctx, x + 2, y, segW - 4, h, r);
            ctx.clip();

            ctx.fillText(label, centerX, centerY - 6);

            // Frame range + duration (frames / fps, so it matches the cut exactly).
            ctx.font = "9px monospace";
            ctx.fillStyle = "#ccc";
            ctx.shadowBlur = 0;
            const sf = this._frame(seg.start_sec);
            const ef = this._frame(seg.end_sec);
            const durSec = (ef - sf) / this.fps;
            ctx.fillText(`f${sf}–f${ef} · ${durSec.toFixed(2)}s`, centerX, centerY + 8);

            ctx.restore();
        }

        // Resize handle indicators — full height + bright so the grab zone is
        // discoverable (the hit tolerance is EDGE_GRAB px around each edge).
        if (isHover || isSelected || isDragging) {
            ctx.fillStyle = "rgba(255,255,255,0.85)";
            ctx.fillRect(x, y, 3, h);
            ctx.fillRect(x2 - 3, y, 3, h);
        }
    }

    _roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.arcTo(x + w, y, x + w, y + r, r);
        ctx.lineTo(x + w, y + h - r);
        ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
        ctx.lineTo(x + r, y + h);
        ctx.arcTo(x, y + h, x, y + h - r, r);
        ctx.lineTo(x, y + r);
        ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
    }

    // --- Public API ---

    update(dur, videoPath, fps, newEntries) {
        this.duration = dur;
        this.entries = newEntries || [];
        this.videoPath = videoPath || "";
        this.fps = (fps && fps > 0) ? fps : DEFAULT_FPS;
        this.playhead = this._snapSec(Math.max(0, Math.min(this.duration, this.playhead)));

        if (videoPath) {
            const url = api.apiURL("/foleytune/timeline_thumbnails?video_path=" +
                encodeURIComponent(videoPath));
            this.thumbImg.onerror = () =>
                console.warn("[FoleyTune timeline] thumbnail request failed (see Network tab):", url);
            if (this.thumbImg.src !== url) {
                this.thumbImg.src = url;
            }
            this._loadPreviewFrame();  // seed the scrub preview at the current playhead
        } else {
            console.warn("[FoleyTune timeline] no video_path in features — thumbnail strip will stay blank");
        }

        this._enforceWidth();
        this._updateCanvasSize();
        this.render();
        // Re-measure next frame in case the DOM widget hasn't settled its width.
        requestAnimationFrame(() => {
            this._enforceWidth();
            this._updateCanvasSize();
            this.render();
        });
    }

    restoreSegments(json) {
        try {
            const parsed = JSON.parse(json);
            if (Array.isArray(parsed)) this.segments = parsed;
        } catch (_) {}
    }

    // Recover segments from the most reliable source available, then re-sync
    // every store. Prefer node.properties (name-keyed, reorder-proof); fall back
    // to the widget value (for graphs saved before properties mirroring). If the
    // widget slot was scrambled, re-committing repairs it from the good source.
    _restore() {
        const props = this.node.properties && this.node.properties.foleytune_segments;
        for (const candidate of [props, this.segWidget && this.segWidget.value]) {
            if (!candidate || candidate === "[]") continue;
            try {
                const parsed = JSON.parse(candidate);
                if (Array.isArray(parsed) && parsed.length) {
                    this.segments = parsed;
                    this._commitFlush();  // heal widget + properties from the good copy
                    return;
                }
            } catch (_) { /* try next source */ }
        }
    }

    destroy() {
        this._resizeObserver?.disconnect();
        if (this._commitTimer) clearTimeout(this._commitTimer);
    }
}


app.registerExtension({
    name: "FoleyTune.LoRATimeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "FoleyTuneLoRATimeline") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;

            const container = document.createElement("div");
            // Always reserve the timeline height (shows an empty box before the
            // node runs — reverted from the collapse-when-empty behaviour).
            container.style.cssText =
                `width:100%;height:${TOTAL_H}px;display:flex;flex-direction:column;` +
                `overflow:hidden;user-select:none;`;

            const wHeight = () => TOTAL_H + 8;

            // Defer construction so widgets list is settled
            let editor = null;
            setTimeout(() => {
                editor = new TimelineEditor(node, container);
                // Robust restore: node.properties (reorder-proof) > widget value,
                // re-syncing both so a scrambled widget slot self-heals.
                editor._restore();
                node._timelineEditor = editor;
            }, 0);

            // DOM Widget. Use a neutral widget type (NOT "preview") — the
            // "preview" type makes ComfyUI apply image aspect-ratio sizing,
            // which computed the width from the height and collapsed the
            // timeline to ~half (esp. when the node was selected/re-laid-out).
            // A plain type is sized full-width like the textareas. We only
            // supply the height via getHeight.
            this.addDOMWidget("timeline", "foleytune_timeline", container, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: wHeight,
                getHeight: wHeight,
                getValue() { return ""; },
                setValue() {},
            });

            // Receive execution output
            const origOnExecuted = node.onExecuted;
            node.onExecuted = function (output) {
                origOnExecuted?.apply(this, arguments);
                if (!editor) return;

                const dur = output?.duration?.[0] ?? 0;
                const vpath = output?.video_path?.[0] ?? "";
                const fps = output?.fps?.[0] ?? 0;
                const ents = output?.entries?.[0] ?? [];

                editor.update(dur, vpath, fps, ents);
                node.setSize([node.size[0], node.computeSize([node.size[0], 0])[1]]);
                node?.graph?.setDirtyCanvas(true);
            };

            // Keep the timeline spanning the full node width when the node is
            // resized (the canvas re-measures clientWidth and redraws).
            const origOnResize = node.onResize;
            node.onResize = function (size) {
                origOnResize?.apply(this, arguments);
                editor?._onResize();
            };

            // Restore from saved workflow
            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                origOnConfigure?.apply(this, arguments);
                // segments_json will be restored by ComfyUI from widgets_values
                // editor.restoreSegments is called in the deferred init above
            };

            // Cleanup
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                editor?.destroy();
                origOnRemoved?.apply(this, arguments);
            };
        };
    },
});


// Timeline Entry: fold the "inherit the sampler's global seed" choice INTO the
// native control_after_generate dropdown (instead of a duplicate mode widget).
// Picking "sampler" parks the seed at -1, the sentinel the backend reads as
// inherit; fixed/randomize/increment keep working on the number as usual.
app.registerExtension({
    name: "FoleyTune.LoRATimelineEntry",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "FoleyTuneLoRATimelineEntry") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;

            const wire = () => {
                const seedW = node.widgets?.find(w => w.name === "seed");
                const ctrlW = node.widgets?.find(w => w.name === "control_after_generate");
                if (!seedW || !ctrlW) return false;

                const vals = ctrlW.options?.values;
                if (Array.isArray(vals) && !vals.includes("sampler")) vals.push("sampler");

                if (!ctrlW._ftSamplerHook) {
                    ctrlW._ftSamplerHook = true;
                    const orig = ctrlW.callback;
                    ctrlW.callback = function (...args) {
                        const r = orig?.apply(this, args);
                        if (ctrlW.value === "sampler") {
                            seedW.value = -1;
                            seedW.callback?.call(seedW, -1);
                            node.graph?.setDirtyCanvas(true, true);
                        }
                        return r;
                    };
                    if (ctrlW.value === "sampler") seedW.value = -1;
                }
                return true;
            };

            // control_after_generate is appended a tick after the widgets are
            // built, so retry a few frames until it exists.
            let tries = 0;
            const tick = () => { if (!wire() && tries++ < 20) setTimeout(tick, 50); };
            setTimeout(tick, 0);
        };
    },
});
