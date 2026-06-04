import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const VIDEO_EXTENSIONS = ["webm", "mp4", "mkv", "gif", "mov", "avi"];

// Unique datalist id per file-manager node instance (node.id is unstable at creation).
let _ftRenameSeq = 0;

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

// Keep a video combo's values in the same A->Z order the Python loader produces
// with sorted(os.listdir(...)), so the live list matches the post-refresh list.
// Plain Array.sort() compares by UTF-16 code unit, matching Python's sorted() for
// ASCII filenames (uppercase before lowercase) — do NOT use localeCompare here.
function sortFileWidget(widget) {
    widget?.options?.values?.sort();
}

function addVideoPreview(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);

        const node = this;
        const container = document.createElement("div");
        container.style.width = "100%";
        // Stay collapsed until a video's metadata actually loads. Otherwise an
        // empty <video controls> element renders its ~150px intrinsic size as a
        // black placeholder box (e.g. a combiner that hasn't been executed yet).
        container.hidden = true;

        const videoEl = document.createElement("video");
        videoEl.controls = true;
        videoEl.loop = true;
        videoEl.muted = true;
        videoEl.style.width = "100%";
        videoEl.onmouseenter = () => { videoEl.muted = false; };
        videoEl.onmouseleave = () => { videoEl.muted = true; };
        container.appendChild(videoEl);

        const previewWidget = this.addDOMWidget("videopreview", "preview", container, {
            serialize: false,
            hideOnZoom: false,
            getValue() { return container.value; },
            setValue(v) { container.value = v; },
        });

        previewWidget.videoEl = videoEl;
        previewWidget.aspectRatio = null;

        previewWidget.computeSize = function (width) {
            if (this.aspectRatio && !container.hidden) {
                let height = (node.size[0] - 20) / this.aspectRatio + 10;
                if (!(height > 0)) height = 0;
                // The DOM-widget layer sizes the <video> element to
                // (computedHeight - 2*margin). With the default 10px margin,
                // computedHeight = height + 10 makes the element exactly
                // (node.size[0]-20)/aspectRatio tall — the video's intrinsic
                // height — so it neither overflows the node nor leaves a gap.
                // It must be set here, not just returned: the frontend reads
                // widget.computedHeight directly (defaulting to 50 -> a ~30px
                // box) when no layout pass has refreshed it.
                this.computedHeight = height + 10;
                return [width, height];
            }
            // No video / hidden: collapse the DOM-widget box completely. The
            // element height is (computedHeight - 2*margin); setting it to
            // 2*margin yields a 0px box so no empty black player is shown.
            this.computedHeight = 2 * (this.margin ?? 10);
            return [width, -4];
        };

        videoEl.addEventListener("loadedmetadata", () => {
            const aspectRatio = videoEl.videoWidth / videoEl.videoHeight;
            previewWidget.aspectRatio = aspectRatio;
            // Only reveal when there's an actual video track to show; a file
            // with no video dimensions gives a NaN/0 ratio and would otherwise
            // un-hide an empty 0-height box.
            container.hidden = !(aspectRatio > 0);
            fitHeight(node);
        });

        videoEl.addEventListener("error", () => {
            container.hidden = true;
            fitHeight(node);
        });

        node._ftVideoPreview = previewWidget;

        const onExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            onExecuted?.apply(this, arguments);
            if (output?.gifs?.[0]) {
                const g = output.gifs[0];
                const params = new URLSearchParams({
                    filename: g.filename,
                    type: g.type || "temp",
                    subfolder: g.subfolder || "",
                });
                videoEl.src = api.apiURL("/view?" + params.toString());
            }
        };
    };
}

function addUploadWidget(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);

        const node = this;
        const pathWidget = this.widgets.find((w) => w.name === "video");
        if (!pathWidget) return;

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "video/*,image/gif";
        fileInput.style.display = "none";
        document.body.appendChild(fileInput);

        fileInput.onchange = async () => {
            if (!fileInput.files.length) return;
            const file = fileInput.files[0];
            const body = new FormData();
            body.append("image", file);
            body.append("overwrite", "true");
            const resp = await api.fetchApi("/upload/image", { method: "POST", body });
            if (resp.ok) {
                const data = await resp.json();
                if (!pathWidget.options.values.includes(data.name)) {
                    pathWidget.options.values.push(data.name);
                }
                sortFileWidget(pathWidget);
                pathWidget.value = data.name;
                pathWidget.callback?.(data.name);
            }
        };

        const uploadWidget = this.addWidget("button", "choose video to upload", null, () => {
            fileInput.click();
        });
        uploadWidget.serialize = false;

        this.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
        this.onDragDrop = async (e) => {
            const file = e?.dataTransfer?.files?.[0];
            if (!file) return false;
            const ext = file.name.split(".").pop()?.toLowerCase();
            if (!VIDEO_EXTENSIONS.includes(ext)) return false;
            const body = new FormData();
            body.append("image", file);
            body.append("overwrite", "true");
            const resp = await api.fetchApi("/upload/image", { method: "POST", body });
            if (resp.ok) {
                const data = await resp.json();
                if (!pathWidget.options.values.includes(data.name)) {
                    pathWidget.options.values.push(data.name);
                }
                sortFileWidget(pathWidget);
                pathWidget.value = data.name;
                pathWidget.callback?.(data.name);
            }
            return true;
        };

        function showPreview(filename) {
            if (!filename) return;
            const pw = node._ftVideoPreview;
            if (!pw) return;
            const params = new URLSearchParams({
                filename,
                type: "input",
                subfolder: "",
            });
            pw.videoEl.src = api.apiURL("/view?" + params.toString());
        }

        const origCallback = pathWidget.callback;
        pathWidget.callback = function (value) {
            origCallback?.apply(this, arguments);
            showPreview(value);
        };

        requestAnimationFrame(() => showPreview(pathWidget.value));
    };
}

function addFileManagerWidgets(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);

        const node = this;

        // Find the FoleyTuneVideoLoaderUpload this node feeds, plus its `video` widget.
        function getLoader() {
            const out = node.outputs?.[0];
            if (!out?.links?.length) return null;
            for (const linkId of out.links) {
                const link = node.graph?.links?.[linkId];
                if (!link) continue;
                const target = node.graph.getNodeById(link.target_id);
                const widget = target?.widgets?.find((w) => w.name === "video");
                if (widget) return { node: target, widget };
            }
            return null;
        }

        function selectAfterRemoval(widget, removed) {
            const values = widget.options?.values || [];
            const idx = values.indexOf(removed);
            if (idx >= 0) values.splice(idx, 1);
            widget.value = values[0] || "";
            widget.callback?.(widget.value);
        }

        // Rename field: an <input> with a <datalist> that suggests existing file base
        // names (extension stripped) from the connected loader, filtered by prefix.
        const listId = `ft-rename-list-${_ftRenameSeq++}`;
        const wrap = document.createElement("div");
        wrap.style.width = "100%";
        const input = document.createElement("input");
        input.type = "text";
        input.placeholder = "new name (extension kept automatically)";
        input.autocomplete = "off";
        input.setAttribute("list", listId);
        input.style.cssText =
            "width:100%;box-sizing:border-box;background:#222;color:#ddd;" +
            "border:1px solid #444;border-radius:4px;padding:2px 6px;font-size:12px;";
        const datalist = document.createElement("datalist");
        datalist.id = listId;
        wrap.appendChild(input);
        wrap.appendChild(datalist);
        node._ftRenameInput = input;

        function refreshSuggestions() {
            const loader = getLoader();
            const typed = input.value.trim().toLowerCase();
            datalist.innerHTML = "";
            if (!loader) return;
            const bases = new Set();
            for (const v of (loader.widget.options?.values || [])) {
                const base = v.replace(/\s*\[[^\]]*\]$/, "").replace(/\.[^.]+$/, "");
                if (base) bases.add(base);
            }
            for (const base of [...bases].sort()) {
                if (!typed || base.toLowerCase().startsWith(typed)) {
                    const opt = document.createElement("option");
                    opt.value = base;
                    datalist.appendChild(opt);
                }
            }
        }
        input.addEventListener("focus", refreshSuggestions);
        input.addEventListener("input", refreshSuggestions);

        const renameWidget = node.addDOMWidget("new_name", "text", wrap, {
            serialize: false,
            hideOnZoom: false,
            getValue() { return input.value; },
            setValue(v) { input.value = v ?? ""; },
        });
        renameWidget.computeSize = function (width) { return [width, 30]; };
        requestAnimationFrame(() => fitHeight(node));

        const renameBtn = node.addWidget("button", "rename file", null, async () => {
            const loader = getLoader();
            if (!loader) {
                alert("Connect this node's output to a FoleyTune Video Loader (Upload) input.");
                return;
            }
            const oldName = loader.widget.value;
            if (!oldName) { alert("No video is selected on the loader."); return; }
            const newName = (node._ftRenameInput?.value || "").trim();
            if (!newName) { alert("Type a new name first."); return; }

            let resp;
            try {
                resp = await api.fetchApi("/foleytune/rename_input", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name: oldName, new_name: newName }),
                });
            } catch (e) {
                alert("Rename request failed: " + e);
                return;
            }
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) { alert("Rename failed: " + (data.error || resp.status)); return; }

            const finalName = data.name;
            const values = loader.widget.options?.values || [];
            const idx = values.indexOf(oldName);
            if (idx >= 0) values[idx] = finalName;
            else if (!values.includes(finalName)) values.push(finalName);
            sortFileWidget(loader.widget);
            loader.widget.value = finalName;
            loader.widget.callback?.(finalName);
            loader.node.setDirtyCanvas(true, true);
            if (node._ftRenameInput) node._ftRenameInput.value = "";
        });
        renameBtn.serialize = false;

        const deleteBtn = node.addWidget("button", "delete file", null, async () => {
            const loader = getLoader();
            if (!loader) {
                alert("Connect this node's output to a FoleyTune Video Loader (Upload) input.");
                return;
            }
            const name = loader.widget.value;
            if (!name) { alert("No video is selected on the loader."); return; }
            if (!confirm(`Delete "${name}"?\nIt will be moved to the trash (recoverable).`)) return;

            let resp;
            try {
                resp = await api.fetchApi("/foleytune/delete_input", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name }),
                });
            } catch (e) {
                alert("Delete request failed: " + e);
                return;
            }
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) { alert("Delete failed: " + (data.error || resp.status)); return; }

            selectAfterRemoval(loader.widget, name);
            loader.node.setDirtyCanvas(true, true);
        });
        deleteBtn.serialize = false;
    };
}

app.registerExtension({
    name: "FoleyTune.VideoNodes",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === "FoleyTuneVideoLoader") {
            addVideoPreview(nodeType);
        }
        if (nodeData?.name === "FoleyTuneVideoLoaderUpload") {
            addVideoPreview(nodeType);
            addUploadWidget(nodeType);
        }
        if (nodeData?.name === "FoleyTuneVideoCombiner") {
            addVideoPreview(nodeType);
        }
        if (nodeData?.name === "FoleyTuneVideoFileManager") {
            addFileManagerWidgets(nodeType);
        }
    },
});
