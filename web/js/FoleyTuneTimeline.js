import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const COLORS = {
    red: "#e74c3c", blue: "#3498db", green: "#2ecc71",
    yellow: "#f1c40f", purple: "#9b59b6", orange: "#e67e22",
};

app.registerExtension({
    name: "FoleyTune.LoRATimeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "FoleyTuneLoRATimeline") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;

            const container = document.createElement("div");
            container.style.cssText = "width:100%;position:relative;user-select:none;";

            // Thumbnail strip
            const thumbStrip = document.createElement("div");
            thumbStrip.style.cssText =
                "width:100%;height:60px;background:#111;overflow:hidden;position:relative;";
            const thumbImg = document.createElement("img");
            thumbImg.style.cssText = "height:100%;position:absolute;left:0;top:0;";
            thumbStrip.appendChild(thumbImg);
            container.appendChild(thumbStrip);

            // Ruler
            const ruler = document.createElement("canvas");
            ruler.style.cssText = "width:100%;height:20px;display:block;";
            ruler.height = 20;
            container.appendChild(ruler);

            // Segment track
            const track = document.createElement("div");
            track.style.cssText =
                "width:100%;height:40px;background:#1a1a2e;position:relative;overflow:hidden;border:1px solid #333;";
            container.appendChild(track);

            // State
            let duration = 0;
            let entries = [];
            let segments = [];
            let selectedIdx = -1;
            let dragState = null;

            const segmentsWidget = node.widgets.find(w => w.name === "segments_json");

            function secToX(sec) {
                return (sec / duration) * track.clientWidth;
            }
            function xToSec(x) {
                const sec = (x / track.clientWidth) * duration;
                return Math.round(sec * 2) / 2;
            }

            function renderSegments() {
                track.querySelectorAll(".lora-seg").forEach(el => el.remove());
                segments.forEach((seg, i) => {
                    const el = document.createElement("div");
                    el.className = "lora-seg";
                    const entry = entries[seg.entry_index] || {};
                    const color = COLORS[entry.color] || COLORS.blue;
                    const left = secToX(seg.start_sec);
                    const width = secToX(seg.end_sec) - left;
                    el.style.cssText =
                        `position:absolute;top:2px;bottom:2px;left:${left}px;width:${width}px;` +
                        `background:${color}88;border:2px solid ${color};border-radius:4px;` +
                        `cursor:grab;display:flex;align-items:center;justify-content:center;` +
                        `font-size:11px;color:#fff;text-shadow:0 1px 2px #000;overflow:hidden;white-space:nowrap;`;
                    el.textContent = entry.label || `LoRA ${seg.entry_index}`;
                    if (i === selectedIdx) el.style.outline = "2px solid #fff";

                    // Resize handles
                    const leftHandle = document.createElement("div");
                    leftHandle.style.cssText =
                        "position:absolute;left:0;top:0;bottom:0;width:6px;cursor:ew-resize;";
                    const rightHandle = document.createElement("div");
                    rightHandle.style.cssText =
                        "position:absolute;right:0;top:0;bottom:0;width:6px;cursor:ew-resize;";

                    leftHandle.addEventListener("mousedown", (e) => {
                        e.stopPropagation();
                        dragState = { type: "resize-left", idx: i, startX: e.clientX, origStart: seg.start_sec, origEnd: seg.end_sec };
                    });
                    rightHandle.addEventListener("mousedown", (e) => {
                        e.stopPropagation();
                        dragState = { type: "resize-right", idx: i, startX: e.clientX, origStart: seg.start_sec, origEnd: seg.end_sec };
                    });
                    el.addEventListener("mousedown", (e) => {
                        selectedIdx = i;
                        dragState = { type: "move", idx: i, startX: e.clientX, origStart: seg.start_sec, origEnd: seg.end_sec };
                        renderSegments();
                    });
                    el.addEventListener("contextmenu", (e) => {
                        e.preventDefault();
                        segments.splice(i, 1);
                        selectedIdx = -1;
                        syncWidget();
                        renderSegments();
                    });

                    el.appendChild(leftHandle);
                    el.appendChild(rightHandle);
                    track.appendChild(el);
                });
            }

            function renderRuler() {
                ruler.width = ruler.clientWidth;
                const ctx = ruler.getContext("2d");
                ctx.fillStyle = "#222";
                ctx.fillRect(0, 0, ruler.width, ruler.height);
                ctx.fillStyle = "#aaa";
                ctx.font = "10px monospace";
                const step = duration <= 30 ? 1 : duration <= 120 ? 5 : 10;
                for (let t = 0; t <= duration; t += step) {
                    const x = secToX(t);
                    ctx.fillRect(x, 0, 1, t % (step * 5) === 0 ? 14 : 8);
                    if (t % (step * 2) === 0) ctx.fillText(`${t}s`, x + 2, 18);
                }
            }

            function syncWidget() {
                if (segmentsWidget) {
                    segmentsWidget.value = JSON.stringify(segments);
                }
            }

            // Double-click on track to add segment
            track.addEventListener("dblclick", (e) => {
                if (!entries.length || duration <= 0) return;
                const rect = track.getBoundingClientRect();
                const sec = xToSec(e.clientX - rect.left);
                let entryIdx = 0;
                if (entries.length > 1) {
                    const input = prompt(`Entry index (0-${entries.length - 1}):`, "0");
                    if (input === null) return;
                    entryIdx = parseInt(input, 10);
                    if (entryIdx < 0 || entryIdx >= entries.length) return;
                }
                const startSec = Math.max(0, sec - 2);
                const endSec = Math.min(duration, sec + 2);
                segments.push({ entry_index: entryIdx, start_sec: startSec, end_sec: endSec, strength: entries[entryIdx].strength });
                segments.sort((a, b) => a.start_sec - b.start_sec);
                syncWidget();
                renderSegments();
            });

            // Global drag handlers
            const onMouseMove = (e) => {
                if (!dragState) return;
                const dx = e.clientX - dragState.startX;
                const dSec = (dx / track.clientWidth) * duration;
                const seg = segments[dragState.idx];

                if (dragState.type === "move") {
                    const len = dragState.origEnd - dragState.origStart;
                    let newStart = Math.round((dragState.origStart + dSec) * 2) / 2;
                    newStart = Math.max(0, Math.min(duration - len, newStart));
                    seg.start_sec = newStart;
                    seg.end_sec = newStart + len;
                } else if (dragState.type === "resize-left") {
                    seg.start_sec = Math.max(0, Math.min(seg.end_sec - 0.5,
                        Math.round((dragState.origStart + dSec) * 2) / 2));
                } else if (dragState.type === "resize-right") {
                    seg.end_sec = Math.max(seg.start_sec + 0.5, Math.min(duration,
                        Math.round((dragState.origEnd + dSec) * 2) / 2));
                }
                syncWidget();
                renderSegments();
            };
            const onMouseUp = () => { dragState = null; };
            document.addEventListener("mousemove", onMouseMove);
            document.addEventListener("mouseup", onMouseUp);

            // DOM Widget
            const widget = this.addDOMWidget("timeline", "preview", container, {
                serialize: false,
                hideOnZoom: false,
                getValue() { return ""; },
                setValue() {},
            });
            widget.computeSize = function (width) {
                return [width, duration > 0 ? 130 : -4];
            };

            // Update from execution output
            const origOnExecuted = node.onExecuted;
            node.onExecuted = function (output) {
                origOnExecuted?.apply(this, arguments);

                if (output?.duration?.[0]) duration = output.duration[0];
                if (output?.video_path?.[0]) {
                    const url = api.apiURL("/foleytune/timeline_thumbnails?video_path=" +
                        encodeURIComponent(output.video_path[0]));
                    thumbImg.src = url;
                    thumbImg.onload = () => {
                        thumbImg.style.width = thumbStrip.clientWidth + "px";
                    };
                }
                if (output?.entries?.[0]) entries = output.entries[0];

                renderRuler();
                renderSegments();
                node.setSize([node.size[0], node.computeSize([node.size[0], 0])[1]]);
                node?.graph?.setDirtyCanvas(true);
            };

            // Cleanup
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                document.removeEventListener("mousemove", onMouseMove);
                document.removeEventListener("mouseup", onMouseUp);
                origOnRemoved?.apply(this, arguments);
            };
        };
    },
});
