export default {
    props: {
        interactive: {
            type: Boolean,
            default: false
        }
    },
    template: `
        <div class="video-tracker w-full h-full relative bg-slate-900 rounded-lg overflow-hidden flex flex-col justify-between">
            <!-- Hidden file inputs always mounted so Load Video and Load Session (.zip) work at any time -->
            <input type="file" ref="fileInput" class="hidden" accept="video/*" @change="handleFileUpload">
            <input type="file" ref="sessionInput" class="hidden" accept=".zip" @change="handleSessionZipUpload">

            <!-- Main viewport area -->
            <div class="relative w-full flex-grow flex items-center justify-center overflow-hidden">
                <div v-if="!activeMode" class="flex flex-col items-center gap-6 p-8">
                    <div class="text-white text-xl mb-4 font-semibold">Select Media Source</div>
                    <div class="flex gap-4">
                        <button @click="startWebcam" class="px-6 py-3 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700 transition flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                <path d="M2 6a2 2 0 012-2h12a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" />
                            </svg>
                            Start Camera
                        </button>
                        <div class="relative">
                            <button class="px-6 py-3 bg-teal-600 text-white rounded-lg shadow hover:bg-teal-700 transition flex items-center gap-2" @click="open_file_dialog">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clip-rule="evenodd" />
                                </svg>
                                Upload Video
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Video element used for live webcam capture (hidden in webcam mode) and visible file playback -->
                <video ref="videoElement" 
                       class="w-full h-full object-contain" 
                       :class="{ 'hidden': activeMode !== 'file' }"
                       playsinline 
                       muted 
                       controls 
                       v-show="activeMode === 'file'"></video>

                <!-- Buffered display canvas for 200ms delayed real-time webcam playback AND static session GT frame review -->
                <canvas ref="displayCanvas"
                        class="w-full h-full object-contain"
                        :class="{ 'hidden': activeMode !== 'webcam' && activeMode !== 'session' }"
                        v-show="activeMode === 'webcam' || activeMode === 'session'"></canvas>
                
                <!-- Hidden canvas for capturing and encoding webcam/video frames -->
                <canvas ref="hiddenCanvas" class="hidden"></canvas>
                
                <!-- SVG overlay for drawing skeleton edges and interactive keypoint markers -->
                <svg class="absolute top-0 left-0 w-full h-full"
                     :class="activeKeypointId !== null || draggingKeypointId !== null ? 'cursor-crosshair pointer-events-auto' : 'pointer-events-none'"
                     v-if="activeMode"
                     @mousedown="handleCanvasMouseDown($event)"
                     @mousemove="onMouseMove($event)"
                     @mouseup="stopDrag($event)"
                     @mouseleave="stopDrag($event)">
                    <!-- Skeleton anatomical lines -->
                    <line v-for="line in visibleSkeletonLines"
                          :key="line.key"
                          :x1="line.x1"
                          :y1="line.y1"
                          :x2="line.x2"
                          :y2="line.y2"
                          :stroke="line.color"
                          stroke-width="2.5"
                          :stroke-opacity="line.occluded ? 0.4 : 0.85"
                          :stroke-dasharray="line.occluded ? '4,4' : null"
                          class="pointer-events-none" />

                    <!-- Skeleton keypoint markers -->
                    <g v-for="(marker, i) in visibleMarkers" :key="marker.id !== undefined ? marker.id : i">
                        <circle class="pointer-events-auto cursor-grab"
                                :cx="marker.x"
                                :cy="marker.y"
                                :r="activeKeypointId === marker.id || draggingKeypointId === marker.id || hoveredKeypointId === marker.id ? 9 : 6"
                                :fill="getKeypointColor(marker.id)"
                                :fill-opacity="marker.visibility === 0 ? 0.35 : 0.95"
                                :stroke="activeKeypointId === marker.id ? '#fde047' : 'white'"
                                :stroke-width="activeKeypointId === marker.id ? 3 : 2"
                                :stroke-dasharray="marker.visibility === 0 ? '3,2' : null"
                                @mousedown.stop.prevent="startDrag(marker, $event)"
                                @mouseenter="hoveredKeypointId = marker.id"
                                @mouseleave="hoveredKeypointId = null">
                            <title>{{ marker.id !== undefined ? (marker.id + ': ' + getKeypointName(marker.id)) : 'Marker' }}</title>
                        </circle>
                        <text v-if="(activeKeypointId === marker.id || hoveredKeypointId === marker.id || draggingKeypointId === marker.id) && marker.id !== undefined"
                              :x="marker.x + 12"
                              :y="marker.y + 4"
                              fill="white"
                              stroke="#0f172a"
                              stroke-width="3"
                              paint-order="stroke"
                              font-size="11"
                              font-weight="bold"
                              class="pointer-events-none select-none">{{ marker.id }}: {{ getKeypointName(marker.id) }}</text>
                    </g>
                </svg>

                <!-- Session GT Review Mode Top-Left Banner -->
                <div v-if="activeMode === 'session'" class="absolute top-3 left-3 flex items-center gap-2 bg-amber-500/90 text-slate-950 px-3 py-1.5 rounded-lg text-xs font-bold shadow-lg backdrop-blur-sm z-20">
                    <span>Session GT Review ({{ activeSessionIndex + 1 }}/{{ sessionSamples.length }}): {{ sessionSamples[activeSessionIndex]?.sample_id || '' }}</span>
                    <button @click="exit_session_review" class="ml-2 bg-slate-900 text-white hover:bg-slate-800 px-2 py-0.5 rounded text-[11px] font-semibold transition">
                        Back to Video
                    </button>
                </div>

                <!-- Reset overlay button -->
                <button v-if="activeMode" @click="stop" class="absolute top-3 right-3 bg-red-600/80 hover:bg-red-700 text-white px-3 py-1 rounded text-sm shadow backdrop-blur-sm transition z-20">
                    Close Media
                </button>

                <!-- Progress Overlay -->
                <div v-if="progressStage" class="absolute inset-0 flex items-center justify-center bg-black/70 z-50 backdrop-blur-sm">
                    <div class="bg-slate-800 p-6 rounded-xl shadow-2xl w-80 text-center border border-slate-700">
                        <div class="text-white text-lg font-semibold mb-4">{{ progressStage }}</div>
                        <div class="w-full bg-slate-700 rounded-full h-3 mb-2 overflow-hidden">
                            <div class="bg-blue-500 h-3 rounded-full transition-all duration-300 ease-out" :style="{ width: progressPercent + '%' }"></div>
                        </div>
                        <div class="text-sm text-slate-400">{{ Math.round(progressPercent) }}%</div>
                    </div>
                </div>
            </div>

            <!-- Bottom Session Filmstrip (Saved GT Frames - 100% Client-Side) -->
            <div v-if="interactive"
                 class="bg-slate-950/95 border-t border-slate-800 px-3 py-2 flex items-center gap-2.5 overflow-x-auto w-full z-20 shrink-0">
                <div class="flex items-center gap-2 shrink-0 pr-1">
                    <span class="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
                        Session GT ({{ sessionSamples.length }})
                    </span>
                    <button v-if="activeMode === 'file' || activeMode === 'webcam'"
                            @click="save_current_frame_to_session"
                            class="px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-semibold shadow flex items-center gap-1 shrink-0 transition">
                        + Save Frame to Session
                    </button>
                </div>
                <div v-if="sessionSamples.length === 0" class="text-xs text-slate-400 italic py-2">
                    No frames saved in session yet. Load a video, pause on a frame, adjust keypoints, and click "+ Save Frame to Session (GT)".
                </div>
                <div v-for="(sample, idx) in sessionSamples"
                     :key="sample.sample_id || idx"
                     @click="select_session_sample({ index: idx })"
                     class="relative group cursor-pointer rounded-md overflow-hidden border transition shrink-0"
                     :class="activeMode === 'session' && activeSessionIndex === idx ? 'ring-2 ring-amber-400 border-amber-400' : 'border-slate-700 hover:border-blue-400'">
                    <img v-if="sample.image_b64" :src="sample.image_b64" class="h-12 w-20 object-cover block bg-slate-800" />
                    <div v-else class="h-12 w-20 bg-slate-800 flex items-center justify-center text-[10px] text-slate-400">Frame #{{ sample.frame_index }}</div>
                    <div class="absolute bottom-0 inset-x-0 bg-black/75 text-[10px] text-white px-1 py-0.5 truncate font-mono">
                        {{ sample.video_name }} #{{ sample.frame_index }}
                    </div>
                    <button @click.stop="delete_session_sample({ index: idx })"
                            title="Remove from Session"
                            class="absolute top-0.5 right-0.5 bg-red-600/90 hover:bg-red-700 text-white rounded-full w-4 h-4 flex items-center justify-center text-[10px] leading-none opacity-80 group-hover:opacity-100">
                        &times;
                    </button>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            activeMode: null, // 'webcam', 'file', or 'session'
            stream: null,
            markers: [],
            all_frames_markers: [],
            animationFrameId: null,
            isProcessing: false,
            progressStage: '',
            progressPercent: 0,
            estimatedSeconds: 0,
            // Real-time 200ms buffer, frame indexing, dilation, and interpolation state
            bufferDelayMs: 200,
            frameDilation: 4,
            interpolationMode: 'linear',
            frameIndex: 0,
            lastInferredFrameIndex: -4,
            frameBuffer: [],
            frameMarkersMap: {},
            inferredKeyframes: [],
            displayedFrameIndex: -1,
            // Client-side multi-video annotation session state
            currentVideoName: 'video',
            sessionSamples: [],
            activeSessionIndex: -1,
            previousMediaMode: null,
            sessionImageWidth: 640,
            sessionImageHeight: 480,
            // COCO 17-keypoint colors, names, keys, skeleton edges, visibility, and interactive state
            keypointColors: {
                0: '#EF4444',
                1: '#F97316',
                2: '#F59E0B',
                3: '#EAB308',
                4: '#84CC16',
                5: '#22C55E',
                6: '#10B981',
                7: '#14B8A6',
                8: '#06B6D4',
                9: '#0EA5E9',
                10: '#3B82F6',
                11: '#6366F1',
                12: '#8B5CF6',
                13: '#A855F7',
                14: '#D946EF',
                15: '#EC4899',
                16: '#F43F5E'
            },
            keypointNames: {
                0: 'Nose',
                1: 'L. Eye',
                2: 'R. Eye',
                3: 'L. Ear',
                4: 'R. Ear',
                5: 'L. Shoulder',
                6: 'R. Shoulder',
                7: 'L. Elbow',
                8: 'R. Elbow',
                9: 'L. Wrist',
                10: 'R. Wrist',
                11: 'L. Hip',
                12: 'R. Hip',
                13: 'L. Knee',
                14: 'R. Knee',
                15: 'L. Ankle',
                16: 'R. Ankle'
            },
            keypointKeys: {
                0: 'nose',
                1: 'left_eye',
                2: 'right_eye',
                3: 'left_ear',
                4: 'right_ear',
                5: 'left_shoulder',
                6: 'right_shoulder',
                7: 'left_elbow',
                8: 'right_elbow',
                9: 'left_wrist',
                10: 'right_wrist',
                11: 'left_hip',
                12: 'right_hip',
                13: 'left_knee',
                14: 'right_knee',
                15: 'left_ankle',
                16: 'right_ankle'
            },
            skeletonEdges: [
                [0, 1], [0, 2], [1, 3], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [13, 15], [12, 14], [14, 16]
            ],
            visibilityMap: {
                0: true, 1: true, 2: true, 3: true, 4: true,
                5: true, 6: true, 7: true, 8: true, 9: true,
                10: true, 11: true, 12: true, 13: true, 14: true,
                15: true, 16: true
            },
            activeKeypointId: null,
            draggingKeypointId: null,
            hoveredKeypointId: null,
            currentFrameIndex: 0
        }
    },
    computed: {
        visibleMarkers() {
            return (this.markers || []).filter(m => m.id === undefined || this.isKeypointVisible(m.id));
        },
        visibleSkeletonLines() {
            const byId = {};
            for (const m of (this.markers || [])) {
                if (m.id !== undefined && this.isKeypointVisible(m.id)) {
                    byId[m.id] = m;
                }
            }
            const lines = [];
            for (const [idA, idB] of this.skeletonEdges) {
                const mA = byId[idA];
                const mB = byId[idB];
                if (mA && mB) {
                    lines.push({
                        key: `${idA}-${idB}`,
                        x1: mA.x,
                        y1: mA.y,
                        x2: mB.x,
                        y2: mB.y,
                        color: this.getKeypointColor(idA),
                        occluded: (mA.visibility === 0 || mB.visibility === 0)
                    });
                }
            }
            return lines;
        }
    },
    methods: {
        getKeypointColor(id) {
            return this.keypointColors[id] || '#3B82F6';
        },
        getKeypointName(id) {
            return this.keypointNames[id] || ('KP ' + id);
        },
        isKeypointVisible(id) {
            return this.visibilityMap[id] !== false;
        },
        getCurrentFrameIndex() {
            if (this.activeMode === 'session' && this.sessionSamples[this.activeSessionIndex]) {
                return Math.max(0, Number(this.sessionSamples[this.activeSessionIndex].frame_index || 0));
            }
            if (this.activeMode === 'file') {
                const video = this.$refs.videoElement;
                return Math.max(0, Math.floor(((video && video.currentTime) || 0) * 30));
            }
            return Math.max(0, this.displayedFrameIndex);
        },
        ensureFrameMarkersArray(frameIdx) {
            if (this.activeMode === 'session' && this.sessionSamples[this.activeSessionIndex]) {
                const sample = this.sessionSamples[this.activeSessionIndex];
                if (!Array.isArray(sample.markers)) {
                    sample.markers = [];
                }
                return sample.markers;
            }
            const idx = Math.max(0, Math.floor(frameIdx));
            if (this.activeMode === 'file') {
                if (!Array.isArray(this.all_frames_markers)) {
                    this.all_frames_markers = [];
                }
                while (this.all_frames_markers.length <= idx) {
                    this.all_frames_markers.push([]);
                }
                if (!Array.isArray(this.all_frames_markers[idx])) {
                    this.all_frames_markers[idx] = [];
                }
                return this.all_frames_markers[idx];
            } else {
                if (!Array.isArray(this.frameMarkersMap[idx])) {
                    const existing = this.getMarkersForFrame(idx);
                    this.frameMarkersMap[idx] = existing.map(m => ({ ...m }));
                }
                if (!Array.isArray(this.all_frames_markers)) {
                    this.all_frames_markers = [];
                }
                while (this.all_frames_markers.length <= idx) {
                    this.all_frames_markers.push([]);
                }
                this.all_frames_markers[idx] = this.frameMarkersMap[idx];
                return this.frameMarkersMap[idx];
            }
        },
        captureCurrentFrameBase64() {
            if (this.activeMode === 'session' && this.sessionSamples[this.activeSessionIndex]) {
                return this.sessionSamples[this.activeSessionIndex].image_b64 || null;
            }
            const canvas = this.$refs.hiddenCanvas;
            if (!canvas) return null;
            if (this.activeMode === 'file') {
                const video = this.$refs.videoElement;
                if (!video || !video.videoWidth || !video.videoHeight) return null;
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                return canvas.toDataURL('image/jpeg', 0.92);
            } else if (this.activeMode === 'webcam') {
                const displayCanvas = this.$refs.displayCanvas;
                if (!displayCanvas || !displayCanvas.width || !displayCanvas.height) return null;
                canvas.width = displayCanvas.width;
                canvas.height = displayCanvas.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(displayCanvas, 0, 0, canvas.width, canvas.height);
                return canvas.toDataURL('image/jpeg', 0.92);
            }
            return null;
        },
        emitSessionStatus(action = 'updated') {
            const currentSample = (this.activeMode === 'session' && this.sessionSamples[this.activeSessionIndex])
                ? this.sessionSamples[this.activeSessionIndex]
                : null;
            const currentMarkers = currentSample
                ? (currentSample.markers || [])
                : (this.all_frames_markers[this.currentFrameIndex] || []);
            this.$emit('session_updated', {
                count: this.sessionSamples.length,
                active_session_index: this.activeSessionIndex,
                active_mode: this.activeMode,
                saved_sample_ids: this.sessionSamples.map(s => s.sample_id),
                current_sample_id: currentSample ? currentSample.sample_id : null,
                current_frame_index: this.currentFrameIndex,
                current_markers: currentMarkers,
                action: action
            });
        },
        pixelToNormalized(clientX, clientY) {
            const video = this.$refs.videoElement;
            const displayEl = (this.activeMode === 'webcam' || this.activeMode === 'session')
                ? this.$refs.displayCanvas
                : video;
            const rect = (displayEl || this.$el).getBoundingClientRect();
            const bounds = this.calculateVideoBounds();

            const localX = clientX - rect.left;
            const localY = clientY - rect.top;

            if (!bounds.actualWidth || !bounds.actualHeight) {
                const fallbackX = rect.width > 0 ? Math.max(0.0, Math.min(1.0, localX / rect.width)) : 0.5;
                const fallbackY = rect.height > 0 ? Math.max(0.0, Math.min(1.0, localY / rect.height)) : 0.5;
                return { x: fallbackX, y: fallbackY, inside: true };
            }

            const relX = (localX - bounds.offsetX) / bounds.actualWidth;
            const relY = (localY - bounds.offsetY) / bounds.actualHeight;
            const inside = relX >= 0.0 && relX <= 1.0 && relY >= 0.0 && relY <= 1.0;

            return {
                x: Math.max(0.0, Math.min(1.0, relX)),
                y: Math.max(0.0, Math.min(1.0, relY)),
                inside
            };
        },
        refreshCurrentFrameDisplayMarkers(frameIdx) {
            const bounds = this.calculateVideoBounds();
            let rawMarkers = [];
            if (this.activeMode === 'session' && this.sessionSamples[this.activeSessionIndex]) {
                rawMarkers = this.sessionSamples[this.activeSessionIndex].markers || [];
            } else if (this.activeMode === 'file') {
                rawMarkers = this.all_frames_markers[frameIdx] || [];
            } else {
                rawMarkers = this.getMarkersForFrame(frameIdx);
            }

            if (bounds.actualWidth > 0 && bounds.actualHeight > 0) {
                this.markers = rawMarkers.map(m => ({
                    id: m.id,
                    x: bounds.offsetX + (m.x * bounds.actualWidth),
                    y: bounds.offsetY + (m.y * bounds.actualHeight),
                    visibility: m.visibility !== undefined ? m.visibility : 1.0
                }));
            } else {
                const rect = this.$el ? this.$el.getBoundingClientRect() : { width: 640, height: 480 };
                const w = rect.width || 640;
                const h = rect.height || 480;
                this.markers = rawMarkers.map(m => ({
                    id: m.id,
                    x: m.x * w,
                    y: m.y * h,
                    visibility: m.visibility !== undefined ? m.visibility : 1.0
                }));
            }
        },
        upsertFrameMarker(frameIdx, kpId, normX, normY, visibility = 1.0) {
            const frameMarkers = this.ensureFrameMarkersArray(frameIdx);
            const targetId = Number(kpId);
            const existingIdx = frameMarkers.findIndex(m => Number(m.id) === targetId);
            const updatedMarker = {
                id: targetId,
                x: Number(normX),
                y: Number(normY),
                visibility: visibility !== undefined ? Number(visibility) : 1.0
            };

            if (existingIdx >= 0) {
                frameMarkers.splice(existingIdx, 1, updatedMarker);
            } else {
                frameMarkers.push(updatedMarker);
            }

            this.refreshCurrentFrameDisplayMarkers(frameIdx);
            return frameMarkers;
        },
        startDrag(marker, event) {
            if (!marker || marker.id === undefined) return;
            if (this.activeMode === 'file' && this.$refs.videoElement) {
                this.$refs.videoElement.pause();
            }
            this.draggingKeypointId = marker.id;
            this.activeKeypointId = marker.id;
        },
        handleCanvasMouseDown(event) {
            if (this.activeKeypointId === null || this.activeKeypointId === undefined) return;
            if (this.activeMode === 'file' && this.$refs.videoElement) {
                this.$refs.videoElement.pause();
            }
            const norm = this.pixelToNormalized(event.clientX, event.clientY);
            const frameIdx = this.getCurrentFrameIndex();
            const frameMarkers = this.upsertFrameMarker(
                frameIdx,
                this.activeKeypointId,
                norm.x,
                norm.y,
                1.0
            );
            this.draggingKeypointId = this.activeKeypointId;
            this.$emit('annotation_updated', {
                frame_index: frameIdx,
                keypoint_id: this.activeKeypointId,
                markers: frameMarkers
            });
            if (this.activeMode === 'session') {
                this.emitSessionStatus('edited');
            }
        },
        onMouseMove(event) {
            if (this.draggingKeypointId === null || this.draggingKeypointId === undefined) return;
            const norm = this.pixelToNormalized(event.clientX, event.clientY);
            const frameIdx = this.getCurrentFrameIndex();
            const frameMarkers = this.ensureFrameMarkersArray(frameIdx);
            const existing = frameMarkers.find(m => Number(m.id) === Number(this.draggingKeypointId));
            const vis = existing && existing.visibility !== undefined ? existing.visibility : 1.0;
            this.upsertFrameMarker(frameIdx, this.draggingKeypointId, norm.x, norm.y, vis);
        },
        stopDrag(event) {
            if (this.draggingKeypointId === null || this.draggingKeypointId === undefined) return;
            const frameIdx = this.getCurrentFrameIndex();
            const kpId = this.draggingKeypointId;
            const frameMarkers = this.ensureFrameMarkersArray(frameIdx);
            this.draggingKeypointId = null;
            this.$emit('annotation_updated', {
                frame_index: frameIdx,
                keypoint_id: kpId,
                markers: frameMarkers
            });
            if (this.activeMode === 'session') {
                this.emitSessionStatus('edited');
            }
        },
        set_keypoint_visibility(payload) {
            if (!payload || payload.id === undefined) return;
            this.visibilityMap = {
                ...this.visibilityMap,
                [payload.id]: Boolean(payload.visible)
            };
        },
        set_all_keypoints_visibility(payload) {
            const visible = Boolean(payload && payload.visible);
            const nextMap = {};
            for (let i = 0; i < 17; i++) {
                nextMap[i] = visible;
            }
            this.visibilityMap = nextMap;
        },
        set_active_keypoint(payload) {
            this.activeKeypointId = (payload && payload.id !== undefined && payload.id !== null)
                ? Number(payload.id)
                : null;
        },
        set_frame_keypoint_state(payload) {
            if (!payload || payload.id === undefined || !payload.state) return;
            const kpId = Number(payload.id);
            const state = payload.state;
            const frameIdx = this.getCurrentFrameIndex();
            const frameMarkers = this.ensureFrameMarkersArray(frameIdx);
            const idx = frameMarkers.findIndex(m => Number(m.id) === kpId);

            if (state === 'delete') {
                if (idx >= 0) {
                    frameMarkers.splice(idx, 1);
                }
            } else if (state === 'occluded') {
                if (idx >= 0) {
                    frameMarkers[idx] = { ...frameMarkers[idx], visibility: 0.0 };
                } else {
                    frameMarkers.push({ id: kpId, x: 0.5, y: 0.5, visibility: 0.0 });
                }
            } else if (state === 'visible') {
                if (idx >= 0) {
                    frameMarkers[idx] = { ...frameMarkers[idx], visibility: 1.0 };
                } else {
                    frameMarkers.push({ id: kpId, x: 0.5, y: 0.5, visibility: 1.0 });
                }
            }

            this.refreshCurrentFrameDisplayMarkers(frameIdx);
            this.$emit('annotation_updated', {
                frame_index: frameIdx,
                keypoint_id: kpId,
                state: state,
                markers: frameMarkers
            });
            if (this.activeMode === 'session') {
                this.emitSessionStatus('edited');
            }
        },
        step_frame(payload) {
            if (this.activeMode === 'session') {
                this.step_session_sample(payload);
                return;
            }
            if (this.activeMode !== 'file') return;
            const video = this.$refs.videoElement;
            if (!video) return;
            video.pause();
            const delta = (payload && payload.delta !== undefined) ? Number(payload.delta) : 0;
            const maxDuration = (video.duration && isFinite(video.duration)) ? video.duration : 3600;
            video.currentTime = Math.max(0, Math.min(maxDuration, (video.currentTime || 0) + (delta / 30.0)));
            const frameIdx = this.getCurrentFrameIndex();
            this.currentFrameIndex = frameIdx;
            this.refreshCurrentFrameDisplayMarkers(frameIdx);
            this.$emit('frame_changed', { frame_index: frameIdx });
        },
        toggle_playback() {
            if (this.activeMode !== 'file') return;
            const video = this.$refs.videoElement;
            if (!video) return;
            if (video.paused) {
                video.play();
            } else {
                video.pause();
            }
        },
        open_file_dialog() {
            if (this.$refs.fileInput) {
                this.$refs.fileInput.click();
            }
        },
        open_session_dialog() {
            if (this.$refs.sessionInput) {
                this.$refs.sessionInput.click();
            }
        },
        save_current_frame_to_session() {
            if (!this.activeMode) return;
            if (this.activeMode === 'file' && this.$refs.videoElement) {
                this.$refs.videoElement.pause();
            }
            const frameIdx = this.getCurrentFrameIndex();
            const videoName = (this.activeMode === 'session' && this.sessionSamples[this.activeSessionIndex])
                ? (this.sessionSamples[this.activeSessionIndex].video_name || this.currentVideoName || 'video')
                : (this.currentVideoName || 'video');
            const sampleId = `${videoName}_frame_${String(frameIdx).padStart(6, '0')}`;
            const imageB64 = this.captureCurrentFrameBase64();
            const rawMarkers = this.ensureFrameMarkersArray(frameIdx);
            const clonedMarkers = rawMarkers.map(m => ({
                id: Number(m.id),
                x: Number(m.x),
                y: Number(m.y),
                visibility: m.visibility !== undefined ? Number(m.visibility) : 1.0
            }));

            const sampleObj = {
                sample_id: sampleId,
                video_name: videoName,
                frame_index: frameIdx,
                image_b64: imageB64,
                markers: clonedMarkers
            };

            const existingIdx = this.sessionSamples.findIndex(s => s.sample_id === sampleId);
            if (existingIdx >= 0) {
                this.sessionSamples.splice(existingIdx, 1, sampleObj);
                this.activeSessionIndex = existingIdx;
            } else {
                this.sessionSamples.push(sampleObj);
                this.activeSessionIndex = this.sessionSamples.length - 1;
            }

            this.emitSessionStatus('saved');
        },
        renderSessionSampleOnCanvas(sample) {
            if (!sample) return;
            const displayCanvas = this.$refs.displayCanvas;
            if (!displayCanvas) return;

            if (sample.image_b64) {
                const img = new Image();
                img.onload = () => {
                    this.sessionImageWidth = img.width || 640;
                    this.sessionImageHeight = img.height || 480;
                    displayCanvas.width = this.sessionImageWidth;
                    displayCanvas.height = this.sessionImageHeight;
                    const ctx = displayCanvas.getContext('2d');
                    ctx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
                    ctx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
                    this.$nextTick(() => {
                        this.refreshCurrentFrameDisplayMarkers(sample.frame_index);
                    });
                };
                img.src = sample.image_b64;
            } else {
                this.sessionImageWidth = 640;
                this.sessionImageHeight = 480;
                displayCanvas.width = 640;
                displayCanvas.height = 480;
                const ctx = displayCanvas.getContext('2d');
                ctx.fillStyle = '#0f172a';
                ctx.fillRect(0, 0, 640, 480);
                this.$nextTick(() => {
                    this.refreshCurrentFrameDisplayMarkers(sample.frame_index);
                });
            }
        },
        select_session_sample(payload) {
            const idx = (payload && payload.index !== undefined) ? Number(payload.index) : Number(payload);
            if (isNaN(idx) || idx < 0 || idx >= this.sessionSamples.length) return;

            if (this.activeMode !== 'session') {
                this.previousMediaMode = this.activeMode;
                if (this.activeMode === 'file' && this.$refs.videoElement) {
                    this.$refs.videoElement.pause();
                }
            }

            this.activeMode = 'session';
            this.activeSessionIndex = idx;
            const sample = this.sessionSamples[idx];
            this.currentFrameIndex = Number(sample.frame_index || 0);
            this.renderSessionSampleOnCanvas(sample);

            this.emitSessionStatus('selected');
            this.$emit('frame_changed', { frame_index: this.currentFrameIndex });
        },
        step_session_sample(payload) {
            if (!this.sessionSamples || this.sessionSamples.length === 0) return;
            const delta = (payload && payload.delta !== undefined) ? Number(payload.delta) : 1;
            const baseIdx = this.activeSessionIndex >= 0 ? this.activeSessionIndex : 0;
            const nextIdx = (baseIdx + delta + this.sessionSamples.length) % this.sessionSamples.length;
            this.select_session_sample({ index: nextIdx });
        },
        delete_session_sample(payload) {
            if (!this.sessionSamples || this.sessionSamples.length === 0) return;
            const targetIdx = (payload && payload.index !== undefined && payload.index !== null)
                ? Number(payload.index)
                : this.activeSessionIndex;
            if (isNaN(targetIdx) || targetIdx < 0 || targetIdx >= this.sessionSamples.length) return;

            this.sessionSamples.splice(targetIdx, 1);

            if (this.sessionSamples.length === 0) {
                this.activeSessionIndex = -1;
                if (this.activeMode === 'session') {
                    this.exit_session_review();
                    return;
                }
            } else if (this.activeSessionIndex >= this.sessionSamples.length) {
                this.activeSessionIndex = this.sessionSamples.length - 1;
            }

            if (this.activeMode === 'session' && this.activeSessionIndex >= 0) {
                this.renderSessionSampleOnCanvas(this.sessionSamples[this.activeSessionIndex]);
            }

            this.emitSessionStatus('deleted');
        },
        load_session_samples(payload) {
            const incoming = (payload && Array.isArray(payload.samples)) ? payload.samples : [];
            const replace = payload && payload.replace !== undefined ? Boolean(payload.replace) : true;
            this.sessionSamples = replace ? [...incoming] : [...this.sessionSamples, ...incoming];

            if (this.sessionSamples.length > 0) {
                this.select_session_sample({ index: 0 });
            } else {
                this.activeSessionIndex = -1;
                this.emitSessionStatus('loaded');
            }
        },
        exit_session_review() {
            if (this.activeMode !== 'session') return;
            const fallbackMode = this.previousMediaMode
                || (this.$refs.videoElement && this.$refs.videoElement.getAttribute('src') ? 'file' : null);
            this.activeMode = fallbackMode;
            if (this.activeMode === 'file') {
                const frameIdx = this.getCurrentFrameIndex();
                this.currentFrameIndex = frameIdx;
                this.refreshCurrentFrameDisplayMarkers(frameIdx);
                this.syncMarkers();
            } else if (!this.activeMode) {
                this.markers = [];
            }
            this.emitSessionStatus('exited_review');
        },
        // --- 100% Frontend ZIP Packaging & Unpacking Methods ---
        crc32(bytes) {
            if (!this._crcTable) {
                const table = new Uint32Array(256);
                for (let n = 0; n < 256; n++) {
                    let c = n;
                    for (let k = 0; k < 8; k++) {
                        c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
                    }
                    table[n] = c >>> 0;
                }
                this._crcTable = table;
            }
            let crc = 0xFFFFFFFF;
            for (let i = 0; i < bytes.length; i++) {
                crc = (crc >>> 8) ^ this._crcTable[(crc ^ bytes[i]) & 0xFF];
            }
            return (crc ^ 0xFFFFFFFF) >>> 0;
        },
        base64ToUint8Array(b64Str) {
            if (!b64Str || typeof b64Str !== 'string') return new Uint8Array(0);
            const clean = b64Str.includes(',') ? b64Str.split(',')[1] : b64Str;
            const binStr = atob(clean);
            const bytes = new Uint8Array(binStr.length);
            for (let i = 0; i < binStr.length; i++) {
                bytes[i] = binStr.charCodeAt(i);
            }
            return bytes;
        },
        uint8ArrayToBase64(bytes) {
            let binary = '';
            const chunkSize = 0x8000;
            for (let i = 0; i < bytes.length; i += chunkSize) {
                const sub = bytes.subarray(i, i + chunkSize);
                binary += String.fromCharCode.apply(null, sub);
            }
            return btoa(binary);
        },
        formatFrameYoloPoseLabel(frameMarkers, classId = 0, bboxPadding = 0.08) {
            const byId = {};
            for (const m of (frameMarkers || [])) {
                if (m && m.id !== undefined) {
                    const id = Number(m.id);
                    if (id >= 0 && id < 17) {
                        byId[id] = m;
                    }
                }
            }
            if (Object.keys(byId).length === 0) return '';

            const xs = [];
            const ys = [];
            const tokens = [];
            for (let kpId = 0; kpId < 17; kpId++) {
                const m = byId[kpId];
                if (!m) {
                    tokens.push('0.000000 0.000000 0');
                    continue;
                }
                const cx = Math.min(1.0, Math.max(0.0, Number(m.x || 0)));
                const cy = Math.min(1.0, Math.max(0.0, Number(m.y || 0)));
                const vis = m.visibility !== undefined ? Number(m.visibility) : 1.0;
                const vFlag = vis <= 0.0 ? 1 : 2;
                xs.push(cx);
                ys.push(cy);
                tokens.push(`${cx.toFixed(6)} ${cy.toFixed(6)} ${vFlag}`);
            }

            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);

            const padX = Math.max(0.04, (maxX - minX) * bboxPadding);
            const padY = Math.max(0.04, (maxY - minY) * bboxPadding);

            const x0 = Math.max(0.0, minX - padX);
            const x1 = Math.min(1.0, maxX + padX);
            const y0 = Math.max(0.0, minY - padY);
            const y1 = Math.min(1.0, maxY + padY);

            const w = Math.max(0.02, Math.min(1.0, x1 - x0));
            const h = Math.max(0.02, Math.min(1.0, y1 - y0));
            const xc = Math.min(1.0, Math.max(0.0, (x0 + x1) / 2.0));
            const yc = Math.min(1.0, Math.max(0.0, (y0 + y1) / 2.0));

            return `${classId} ${xc.toFixed(6)} ${yc.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)} ${tokens.join(' ')}`;
        },
        parseFrameYoloPoseLabel(labelText) {
            if (!labelText || typeof labelText !== 'string') return [];
            const lines = labelText.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
            if (lines.length === 0) return [];
            const tokens = lines[0].split(/\s+/);
            if (tokens.length < 56) return [];

            const markers = [];
            for (let kpId = 0; kpId < 17; kpId++) {
                const base = 5 + kpId * 3;
                const px = parseFloat(tokens[base]);
                const py = parseFloat(tokens[base + 1]);
                const vFlag = Math.round(parseFloat(tokens[base + 2]));
                if (isNaN(px) || isNaN(py) || isNaN(vFlag) || vFlag <= 0) continue;
                markers.push({
                    id: kpId,
                    x: Number(Math.min(1.0, Math.max(0.0, px)).toFixed(6)),
                    y: Number(Math.min(1.0, Math.max(0.0, py)).toFixed(6)),
                    visibility: vFlag >= 2 ? 1.0 : 0.0
                });
            }
            return markers;
        },
        generateYoloPoseDataYaml() {
            return (
                "path: .\n" +
                "train: images/train\n" +
                "val: images/train\n" +
                "kpt_shape: [17, 3]\n" +
                "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n" +
                "names:\n" +
                "  0: person\n"
            );
        },
        formatAnnotationsForExport(samplesList, fps = 30.0) {
            const exported = [];
            (samplesList || []).forEach((s, idx) => {
                const kpts = {};
                for (const m of (s.markers || [])) {
                    const id = Number(m.id);
                    const key = this.keypointKeys[id];
                    if (key) {
                        kpts[key] = {
                            id: id,
                            x: Number(Number(m.x || 0).toFixed(6)),
                            y: Number(Number(m.y || 0).toFixed(6)),
                            visibility: Number(Number(m.visibility !== undefined ? m.visibility : 1.0).toFixed(4))
                        };
                    }
                }
                exported.push({
                    frame_idx: idx,
                    timestamp_ms: Number(((idx / fps) * 1000.0).toFixed(2)),
                    keypoints: kpts
                });
            });
            return exported;
        },
        buildZipArchive(entries) {
            const encoder = new TextEncoder();
            const localChunks = [];
            const centralChunks = [];
            let offset = 0;

            for (const entry of entries) {
                const nameBytes = encoder.encode(entry.name);
                const dataBytes = entry.data instanceof Uint8Array
                    ? entry.data
                    : encoder.encode(String(entry.data || ''));
                const crc = this.crc32(dataBytes);
                const size = dataBytes.length;

                // Local file header (30 bytes + nameBytes)
                const localHeader = new Uint8Array(30 + nameBytes.length);
                const lv = new DataView(localHeader.buffer);
                lv.setUint32(0, 0x04034b50, true); // signature
                lv.setUint16(4, 20, true);         // version needed
                lv.setUint16(6, 0, true);          // flags
                lv.setUint16(8, 0, true);          // compression: 0 (STORE)
                lv.setUint16(10, 0, true);         // mod time
                lv.setUint16(12, 0x21, true);      // mod date
                lv.setUint32(14, crc, true);       // crc32
                lv.setUint32(18, size, true);      // compressed size
                lv.setUint32(22, size, true);      // uncompressed size
                lv.setUint16(26, nameBytes.length, true);
                lv.setUint16(28, 0, true);         // extra length
                localHeader.set(nameBytes, 30);

                localChunks.push(localHeader, dataBytes);

                // Central directory header (46 bytes + nameBytes)
                const centralHeader = new Uint8Array(46 + nameBytes.length);
                const cv = new DataView(centralHeader.buffer);
                cv.setUint32(0, 0x02014b50, true); // signature
                cv.setUint16(4, 20, true);         // version made by
                cv.setUint16(6, 20, true);         // version needed
                cv.setUint16(8, 0, true);          // flags
                cv.setUint16(10, 0, true);         // compression: 0 (STORE)
                cv.setUint16(12, 0, true);         // mod time
                cv.setUint16(14, 0x21, true);      // mod date
                cv.setUint32(16, crc, true);       // crc32
                cv.setUint32(20, size, true);      // compressed size
                cv.setUint32(24, size, true);      // uncompressed size
                cv.setUint16(28, nameBytes.length, true);
                cv.setUint16(30, 0, true);         // extra len
                cv.setUint16(32, 0, true);         // comment len
                cv.setUint16(34, 0, true);         // disk number
                cv.setUint16(36, 0, true);         // internal attrs
                cv.setUint32(38, 0, true);         // external attrs
                cv.setUint32(42, offset, true);    // local header offset
                centralHeader.set(nameBytes, 46);

                centralChunks.push(centralHeader);
                offset += localHeader.length + size;
            }

            let centralSize = 0;
            for (const c of centralChunks) centralSize += c.length;

            // End of Central Directory Record (22 bytes)
            const eocd = new Uint8Array(22);
            const ev = new DataView(eocd.buffer);
            ev.setUint32(0, 0x06054b50, true);
            ev.setUint16(4, 0, true);
            ev.setUint16(6, 0, true);
            ev.setUint16(8, entries.length, true);
            ev.setUint16(10, entries.length, true);
            ev.setUint32(12, centralSize, true);
            ev.setUint32(16, offset, true);
            ev.setUint16(20, 0, true);

            const totalBytes = new Uint8Array(offset + centralSize + 22);
            let pos = 0;
            for (const chunk of localChunks) {
                totalBytes.set(chunk, pos);
                pos += chunk.length;
            }
            for (const chunk of centralChunks) {
                totalBytes.set(chunk, pos);
                pos += chunk.length;
            }
            totalBytes.set(eocd, pos);
            return totalBytes;
        },
        buildSessionZipUint8Array() {
            const sessionName = this.currentVideoName || 'walklab_session';
            const entries = [];
            entries.push({ name: 'data.yaml', data: this.generateYoloPoseDataYaml() });

            const manifestSamples = [];
            for (let i = 0; i < this.sessionSamples.length; i++) {
                const s = this.sessionSamples[i];
                const videoName = s.video_name || sessionName;
                const frameIdx = Number(s.frame_index !== undefined ? s.frame_index : i);
                const sampleId = s.sample_id || `${videoName}_frame_${String(frameIdx).padStart(6, '0')}`;
                const markers = Array.isArray(s.markers) ? s.markers : [];

                const imgRelPath = `images/train/${sampleId}.jpg`;
                const lblRelPath = `labels/train/${sampleId}.txt`;

                const imgBytes = this.base64ToUint8Array(s.image_b64 || '');
                entries.push({ name: imgRelPath, data: imgBytes });

                const labelLine = this.formatFrameYoloPoseLabel(markers);
                entries.push({ name: lblRelPath, data: labelLine ? (labelLine + '\n') : '' });

                manifestSamples.push({
                    sample_id: sampleId,
                    video_name: videoName,
                    frame_index: frameIdx,
                    image_path: imgRelPath,
                    label_path: lblRelPath,
                    markers: markers
                });
            }

            const sessionManifest = {
                session_name: sessionName,
                version: 1,
                samples: manifestSamples
            };
            entries.push({ name: 'session.json', data: JSON.stringify(sessionManifest, null, 2) });
            entries.push({
                name: 'annotations.json',
                data: JSON.stringify(this.formatAnnotationsForExport(this.sessionSamples, 30.0), null, 2)
            });

            return this.buildZipArchive(entries);
        },
        save_session_zip() {
            if (!this.sessionSamples || this.sessionSamples.length === 0) {
                return;
            }
            const zipBytes = this.buildSessionZipUint8Array();
            const blob = new Blob([zipBytes], { type: 'application/zip' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${this.currentVideoName || 'walklab'}_session.zip`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(() => URL.revokeObjectURL(url), 2000);
        },
        async parseZipArchive(zipUint8) {
            const view = new DataView(zipUint8.buffer, zipUint8.byteOffset, zipUint8.byteLength);
            const decoder = new TextDecoder();
            const files = {};

            // Locate End of Central Directory Record (0x06054b50)
            let eocdPos = -1;
            for (let i = zipUint8.length - 22; i >= Math.max(0, zipUint8.length - 65557); i--) {
                if (view.getUint32(i, true) === 0x06054b50) {
                    eocdPos = i;
                    break;
                }
            }
            if (eocdPos < 0) {
                throw new Error('Invalid ZIP archive: EOCD not found.');
            }

            const totalEntries = view.getUint16(eocdPos + 8, true);
            let cdPos = view.getUint32(eocdPos + 16, true);

            for (let e = 0; e < totalEntries; e++) {
                if (view.getUint32(cdPos, true) !== 0x02014b50) break;
                const method = view.getUint16(cdPos + 10, true);
                const compSize = view.getUint32(cdPos + 20, true);
                const nameLen = view.getUint16(cdPos + 28, true);
                const extraLen = view.getUint16(cdPos + 30, true);
                const commentLen = view.getUint16(cdPos + 32, true);
                const localOffset = view.getUint32(cdPos + 42, true);

                const name = decoder.decode(zipUint8.subarray(cdPos + 46, cdPos + 46 + nameLen));
                cdPos += 46 + nameLen + extraLen + commentLen;

                const localNameLen = view.getUint16(localOffset + 26, true);
                const localExtraLen = view.getUint16(localOffset + 28, true);
                const dataStart = localOffset + 30 + localNameLen + localExtraLen;
                const compData = zipUint8.subarray(dataStart, dataStart + compSize);

                if (method === 0) {
                    files[name] = compData;
                } else if (method === 8 && typeof DecompressionStream !== 'undefined') {
                    const ds = new DecompressionStream('deflate-raw');
                    const writer = ds.writable.getWriter();
                    writer.write(compData);
                    writer.close();
                    const resp = new Response(ds.readable);
                    const ab = await resp.arrayBuffer();
                    files[name] = new Uint8Array(ab);
                }
            }
            return files;
        },
        async handleSessionZipUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            try {
                const ab = await file.arrayBuffer();
                const files = await this.parseZipArchive(new Uint8Array(ab));
                const decoder = new TextDecoder();
                const loadedSamples = [];

                if (files['session.json']) {
                    const manifest = JSON.parse(decoder.decode(files['session.json']));
                    for (let i = 0; i < (manifest.samples || []).length; i++) {
                        const item = manifest.samples[i];
                        const sampleId = item.sample_id || `sample_${String(i).padStart(6, '0')}`;
                        const imgPath = item.image_path || `images/train/${sampleId}.jpg`;
                        const lblPath = item.label_path || `labels/train/${sampleId}.txt`;

                        let imageB64 = '';
                        if (files[imgPath]) {
                            imageB64 = 'data:image/jpeg;base64,' + this.uint8ArrayToBase64(files[imgPath]);
                        }
                        let markers = item.markers;
                        if (!Array.isArray(markers)) {
                            markers = files[lblPath]
                                ? this.parseFrameYoloPoseLabel(decoder.decode(files[lblPath]))
                                : [];
                        }
                        loadedSamples.push({
                            sample_id: sampleId,
                            video_name: item.video_name || 'video',
                            frame_index: Number(item.frame_index !== undefined ? item.frame_index : i),
                            image_b64: imageB64,
                            markers: markers
                        });
                    }
                } else {
                    const imgKeys = Object.keys(files)
                        .filter(k => k.startsWith('images/train/') && /\.(jpg|jpeg|png)$/i.test(k))
                        .sort();
                    imgKeys.forEach((imgKey, idx) => {
                        const stem = imgKey.split('/').pop().replace(/\.[^/.]+$/, '');
                        const m = stem.match(/^(.*)_frame_(\d+)$/);
                        const videoName = m ? (m[1] || 'video') : stem;
                        const frameIdx = m ? parseInt(m[2], 10) : idx;
                        const imageB64 = 'data:image/jpeg;base64,' + this.uint8ArrayToBase64(files[imgKey]);
                        const lblKey = `labels/train/${stem}.txt`;
                        const markers = files[lblKey]
                            ? this.parseFrameYoloPoseLabel(decoder.decode(files[lblKey]))
                            : [];
                        loadedSamples.push({
                            sample_id: stem,
                            video_name: videoName,
                            frame_index: frameIdx,
                            image_b64: imageB64,
                            markers: markers
                        });
                    });
                }

                this.load_session_samples({ samples: loadedSamples, replace: true });
            } catch (err) {
                console.error('Failed to unpack session ZIP in browser:', err);
                this.$emit('error', 'Failed to load session ZIP archive');
            } finally {
                event.target.value = '';
            }
        },
        request_session_zip_for_training() {
            const zipBytes = this.buildSessionZipUint8Array();
            const zipB64 = this.uint8ArrayToBase64(zipBytes);
            this.$emit('training_zip_ready', {
                zip_b64: zipB64,
                count: this.sessionSamples.length
            });
        },
        async startWebcam() {
            try {
                this.stream = await navigator.mediaDevices.getUserMedia({ video: true });
                const video = this.$refs.videoElement;
                video.srcObject = this.stream;
                video.removeAttribute('controls'); // hide controls for live stream
                await video.play();

                this.frameIndex = 0;
                this.lastInferredFrameIndex = -this.frameDilation;
                this.frameBuffer = [];
                this.frameMarkersMap = {};
                this.inferredKeyframes = [];
                this.displayedFrameIndex = -1;
                this.currentFrameIndex = 0;
                this.currentVideoName = 'webcam';
                this.markers = [];
                this.activeMode = 'webcam';
                
                // Start capturing frames into the 200ms buffer and dispatching dilated inference
                this.captureFrame();
                // Start synchronized delayed playback and marker interpolation loop
                this.syncMarkers();
            } catch (err) {
                console.error("Error accessing webcam:", err);
                this.$emit('error', 'Could not access webcam');
            }
        },
        async handleFileUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const url = URL.createObjectURL(file);
                const video = this.$refs.videoElement;
                video.srcObject = null;
                video.src = url;
                video.setAttribute('controls', 'true');
                video.load();
                this.activeMode = 'file';
                this.previousMediaMode = 'file';
                this.all_frames_markers = [];
                this.currentFrameIndex = 0;
                this.currentVideoName = file.name.replace(/\.[^/.]+$/, "") || 'video';
                
                this.$emit('file_uploaded_start', { name: file.name });
                
                video.onloadedmetadata = async () => {
                    let duration = video.duration;
                    if (!duration || isNaN(duration) || !isFinite(duration)) {
                        duration = 10; // robust fallback
                    }
                    const assumedFps = 30;
                    const totalFrames = Math.floor(duration * assumedFps);
                    
                    try {
                        const estRes = await fetch(`/api/estimate?frames=${totalFrames}`);
                        const estData = await estRes.json();
                        this.estimatedSeconds = estData.estimated_seconds;
                    } catch (e) {
                        this.estimatedSeconds = totalFrames / 5.23;
                    }
                    
                    this.progressStage = 'Uploading video...';
                    this.progressPercent = 0;
                    
                    const xhr = new XMLHttpRequest();
                    xhr.open('POST', '/api/upload_video', true);
                    
                    let analyzingStarted = false;
                    
                    xhr.upload.onprogress = (e) => {
                        if (e.lengthComputable && this.progressStage === 'Uploading video...') {
                            this.progressPercent = (e.loaded / e.total) * 100;
                            
                            // Transition to Analyzing perfectly when upload hits 100%
                            if (e.loaded >= e.total && !analyzingStarted) {
                                analyzingStarted = true;
                                this.progressStage = 'Analyzing pose...';
                                this.progressPercent = 0;
                                
                                const startTime = performance.now();
                                const durationMs = (this.estimatedSeconds || 10) * 1000;
                                
                                const animateProgress = (now) => {
                                    if (this.progressStage !== 'Analyzing pose...') return;
                                    
                                    const elapsed = now - startTime;
                                    let percent = (elapsed / durationMs) * 100;
                                    if (percent > 99) percent = 99; // Hold at 99%
                                    
                                    this.progressPercent = percent;
                                    requestAnimationFrame(animateProgress);
                                };
                                requestAnimationFrame(animateProgress);
                            }
                        }
                    };
                    
                    xhr.onload = () => {
                        this.progressStage = '';
                        if (xhr.status === 200) {
                            const markers = JSON.parse(xhr.responseText);
                            this.all_frames_markers = markers;
                            this.$emit('file_uploaded_success', {
                                name: file.name,
                                all_frames_markers: markers
                            });
                        } else {
                            console.error('Upload failed');
                            this.$emit('error', 'Video processing failed');
                        }
                    };
                    
                    xhr.onerror = () => {
                        this.progressStage = '';
                        console.error('Network error during upload');
                        this.$emit('error', 'Network error during upload');
                    };
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    xhr.send(formData);
                };
                
                this.syncMarkers();
                event.target.value = '';
            }
        },
        captureFrame() {
            if (this.activeMode !== 'webcam') return;
            
            const video = this.$refs.videoElement;
            const canvas = this.$refs.hiddenCanvas;
            
            if (video && video.readyState === video.HAVE_ENOUGH_DATA && video.videoWidth > 0) {
                const currentIdx = this.frameIndex++;
                const now = performance.now();

                // Store frame snapshot in the real-time display buffer
                const frameCanvas = document.createElement('canvas');
                frameCanvas.width = video.videoWidth;
                frameCanvas.height = video.videoHeight;
                const frameCtx = frameCanvas.getContext('2d');
                frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);

                this.frameBuffer.push({
                    frameIndex: currentIdx,
                    timestamp: now,
                    canvas: frameCanvas
                });

                // Cap buffer size to avoid unbounded memory growth
                if (this.frameBuffer.length > 180) {
                    this.frameBuffer.shift();
                }

                // Frame dilation: only dispatch every N-th frame when backend is idle
                const shouldInfer = (currentIdx - this.lastInferredFrameIndex) >= this.frameDilation;
                if (shouldInfer && !this.isProcessing) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(frameCanvas, 0, 0);

                    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
                    this.isProcessing = true;
                    this.lastInferredFrameIndex = currentIdx;
                    this.$emit('frame_ready', {
                        frame_index: currentIdx,
                        frame_data: dataUrl,
                        buffer_delay_ms: this.bufferDelayMs,
                        dilation_factor: this.frameDilation,
                        interpolation_method: this.interpolationMode
                    });
                }
            }
            
            setTimeout(() => this.captureFrame(), 33);
        },
        calculateVideoBounds() {
            if (this.activeMode === 'session') {
                const displayCanvas = this.$refs.displayCanvas;
                if (!displayCanvas) {
                    return { actualWidth: 0, actualHeight: 0, offsetX: 0, offsetY: 0 };
                }
                const rect = displayCanvas.getBoundingClientRect();
                const imgW = this.sessionImageWidth || displayCanvas.width || 640;
                const imgH = this.sessionImageHeight || displayCanvas.height || 480;
                if (!rect.width || !rect.height || !imgW || !imgH) {
                    return { actualWidth: 0, actualHeight: 0, offsetX: 0, offsetY: 0 };
                }
                const imgRatio = imgW / imgH;
                const containerRatio = rect.width / rect.height;
                let actualWidth, actualHeight, offsetX, offsetY;
                if (imgRatio > containerRatio) {
                    actualWidth = rect.width;
                    actualHeight = rect.width / imgRatio;
                    offsetX = 0;
                    offsetY = (rect.height - actualHeight) / 2;
                } else {
                    actualHeight = rect.height;
                    actualWidth = rect.height * imgRatio;
                    offsetX = (rect.width - actualWidth) / 2;
                    offsetY = 0;
                }
                return { actualWidth, actualHeight, offsetX, offsetY };
            }

            const video = this.$refs.videoElement;
            const displayEl = this.activeMode === 'webcam' ? this.$refs.displayCanvas : video;
            if (!displayEl || !video || !video.videoWidth || !video.videoHeight) {
                return { actualWidth: 0, actualHeight: 0, offsetX: 0, offsetY: 0 };
            }
            const rect = displayEl.getBoundingClientRect();
            const videoRatio = video.videoWidth / video.videoHeight;
            const containerRatio = rect.width / rect.height;
            
            let actualWidth, actualHeight, offsetX, offsetY;
            
            if (videoRatio > containerRatio) {
                actualWidth = rect.width;
                actualHeight = rect.width / videoRatio;
                offsetX = 0;
                offsetY = (rect.height - actualHeight) / 2;
            } else {
                actualHeight = rect.height;
                actualWidth = rect.height * videoRatio;
                offsetX = (rect.width - actualWidth) / 2;
                offsetY = 0;
            }
            
            return { actualWidth, actualHeight, offsetX, offsetY };
        },
        interpolateMarkersLinear(startMarkers, endMarkers, alpha) {
            if (!startMarkers || startMarkers.length === 0) {
                return alpha >= 0.5 ? (endMarkers || []) : [];
            }
            if (!endMarkers || endMarkers.length === 0) {
                return alpha < 0.5 ? startMarkers : [];
            }

            const clampedAlpha = Math.max(0.0, Math.min(1.0, alpha));
            const hasIds = startMarkers[0].id !== undefined || endMarkers[0].id !== undefined;

            if (hasIds) {
                const startById = {};
                const endById = {};
                const allIds = [];

                startMarkers.forEach(m => {
                    if (m.id !== undefined) {
                        startById[m.id] = m;
                        allIds.push(m.id);
                    }
                });
                endMarkers.forEach(m => {
                    if (m.id !== undefined) {
                        endById[m.id] = m;
                        if (startById[m.id] === undefined) {
                            allIds.push(m.id);
                        }
                    }
                });

                const result = [];
                for (const id of allIds) {
                    const m0 = startById[id];
                    const m1 = endById[id];
                    if (m0 && m1) {
                        result.push({
                            id: id,
                            x: m0.x + clampedAlpha * (m1.x - m0.x),
                            y: m0.y + clampedAlpha * (m1.y - m0.y),
                            visibility: m0.visibility !== undefined ? m0.visibility : 1.0
                        });
                    } else if (m0 && clampedAlpha < 0.5) {
                        result.push({
                            id: id,
                            x: m0.x,
                            y: m0.y,
                            visibility: m0.visibility !== undefined ? m0.visibility : 1.0
                        });
                    } else if (m1 && clampedAlpha >= 0.5) {
                        result.push({
                            id: id,
                            x: m1.x,
                            y: m1.y,
                            visibility: m1.visibility !== undefined ? m1.visibility : 1.0
                        });
                    }
                }
                return result;
            }

            const count = Math.min(startMarkers.length, endMarkers.length);
            const result = [];
            for (let i = 0; i < count; i++) {
                result.push({
                    x: startMarkers[i].x + clampedAlpha * (endMarkers[i].x - startMarkers[i].x),
                    y: startMarkers[i].y + clampedAlpha * (endMarkers[i].y - startMarkers[i].y),
                    visibility: startMarkers[i].visibility !== undefined ? startMarkers[i].visibility : 1.0
                });
            }
            return result;
        },
        getMarkersForFrame(targetFrameIdx) {
            if (this.frameMarkersMap[targetFrameIdx] !== undefined) {
                return this.frameMarkersMap[targetFrameIdx];
            }

            if (this.inferredKeyframes.length === 0) {
                return [];
            }

            if (this.interpolationMode === 'none' || this.inferredKeyframes.length === 1) {
                let latest = this.inferredKeyframes[0];
                for (const kf of this.inferredKeyframes) {
                    if (kf.frameIndex <= targetFrameIdx) {
                        latest = kf;
                    } else {
                        break;
                    }
                }
                return latest.markers || [];
            }

            let prevKf = null;
            let nextKf = null;
            for (const kf of this.inferredKeyframes) {
                if (kf.frameIndex <= targetFrameIdx) {
                    prevKf = kf;
                } else if (kf.frameIndex > targetFrameIdx) {
                    nextKf = kf;
                    break;
                }
            }

            if (prevKf && nextKf && nextKf.frameIndex > prevKf.frameIndex) {
                const alpha = (targetFrameIdx - prevKf.frameIndex) / (nextKf.frameIndex - prevKf.frameIndex);
                return this.interpolateMarkersLinear(prevKf.markers, nextKf.markers, alpha);
            }

            if (prevKf) {
                return prevKf.markers || [];
            }
            return nextKf ? (nextKf.markers || []) : [];
        },
        syncMarkers() {
            if (!this.activeMode) return;
            
            if (this.activeMode === 'webcam') {
                const now = performance.now();
                let readyIdx = -1;

                for (let i = 0; i < this.frameBuffer.length; i++) {
                    if (now - this.frameBuffer[i].timestamp >= this.bufferDelayMs) {
                        readyIdx = i;
                    } else {
                        break;
                    }
                }

                if (readyIdx >= 0) {
                    const entry = this.frameBuffer[readyIdx];
                    this.frameBuffer.splice(0, readyIdx);

                    const displayCanvas = this.$refs.displayCanvas;
                    if (displayCanvas && entry.canvas) {
                        if (displayCanvas.width !== entry.canvas.width || displayCanvas.height !== entry.canvas.height) {
                            displayCanvas.width = entry.canvas.width;
                            displayCanvas.height = entry.canvas.height;
                        }
                        const ctx = displayCanvas.getContext('2d');
                        ctx.drawImage(entry.canvas, 0, 0);
                        this.displayedFrameIndex = entry.frameIndex;
                        if (this.currentFrameIndex !== entry.frameIndex) {
                            this.currentFrameIndex = entry.frameIndex;
                            this.$emit('frame_changed', { frame_index: entry.frameIndex });
                        }

                        const bounds = this.calculateVideoBounds();
                        const rawMarkers = this.getMarkersForFrame(entry.frameIndex);
                        this.markers = rawMarkers.map(m => ({
                            id: m.id,
                            x: bounds.offsetX + (m.x * bounds.actualWidth),
                            y: bounds.offsetY + (m.y * bounds.actualHeight),
                            visibility: m.visibility !== undefined ? m.visibility : 1.0
                        }));
                    }
                }
            } else if (this.activeMode === 'file' && this.all_frames_markers && this.all_frames_markers.length > 0) {
                const video = this.$refs.videoElement;
                if (video && video.readyState >= video.HAVE_METADATA && video.videoWidth > 0) {
                    const bounds = this.calculateVideoBounds();
                    
                    const frameIndex = Math.floor(video.currentTime * 30);
                    if (frameIndex !== this.currentFrameIndex) {
                        this.currentFrameIndex = frameIndex;
                        this.$emit('frame_changed', { frame_index: frameIndex });
                    }
                    if (frameIndex < this.all_frames_markers.length && this.draggingKeypointId === null) {
                        const markers = this.all_frames_markers[frameIndex] || [];
                        this.markers = markers.map(m => ({
                            id: m.id,
                            x: bounds.offsetX + (m.x * bounds.actualWidth),
                            y: bounds.offsetY + (m.y * bounds.actualHeight),
                            visibility: m.visibility !== undefined ? m.visibility : 1.0
                        }));
                    }
                }
            }
            
            this.animationFrameId = requestAnimationFrame(this.syncMarkers);
        },
        update_markers(payload) {
            this.isProcessing = false;
            if (!payload) return;

            // Legacy array mode fallback
            if (Array.isArray(payload)) {
                const video = this.$refs.videoElement;
                if (video && video.readyState >= video.HAVE_METADATA && video.videoWidth > 0) {
                    const bounds = this.calculateVideoBounds();
                    this.markers = payload.map(m => ({
                        id: m.id,
                        x: bounds.offsetX + (m.x * bounds.actualWidth),
                        y: bounds.offsetY + (m.y * bounds.actualHeight),
                        visibility: m.visibility !== undefined ? m.visibility : 1.0
                    }));
                }
                return;
            }

            // Indexed real-time payload mode
            if (payload.buffer_delay_ms && payload.buffer_delay_ms !== this.bufferDelayMs) {
                if (payload.buffer_adjusted) {
                    console.warn(
                        `[VideoTracker] Inference time (${(payload.inference_time_ms || 0).toFixed(1)} ms) ` +
                        `exceeded half of buffer delay (${this.bufferDelayMs} ms). ` +
                        `Adjusting buffer delay to ${payload.buffer_delay_ms} ms.`
                    );
                }
                this.bufferDelayMs = payload.buffer_delay_ms;
            }

            if (payload.dilation_factor) {
                this.frameDilation = payload.dilation_factor;
            }
            if (payload.interpolation_method) {
                this.interpolationMode = payload.interpolation_method;
            }

            if (payload.interpolated_frames) {
                for (const [k, v] of Object.entries(payload.interpolated_frames)) {
                    this.frameMarkersMap[ parseInt(k, 10) ] = v;
                }
            }

            if (payload.frame_index !== undefined && Array.isArray(payload.markers)) {
                const fIdx = parseInt(payload.frame_index, 10);
                this.frameMarkersMap[fIdx] = payload.markers;
                this.inferredKeyframes.push({
                    frameIndex: fIdx,
                    markers: payload.markers
                });
                this.inferredKeyframes.sort((a, b) => a.frameIndex - b.frameIndex);
                if (this.inferredKeyframes.length > 60) {
                    this.inferredKeyframes.shift();
                }

                // Prune old frameMarkersMap entries
                const minKeepIdx = fIdx - 300;
                for (const key of Object.keys(this.frameMarkersMap)) {
                    if (parseInt(key, 10) < minKeepIdx) {
                        delete this.frameMarkersMap[key];
                    }
                }
            }
        },
        set_tracking_config(config) {
            if (!config) return;
            if (config.bufferDelayMs !== undefined && config.bufferDelayMs > 0) {
                this.bufferDelayMs = config.bufferDelayMs;
            }
            if (config.frameDilation !== undefined && config.frameDilation >= 1) {
                this.frameDilation = config.frameDilation;
            }
            if (config.interpolationMode !== undefined) {
                this.interpolationMode = config.interpolationMode;
            }
        },
        set_batch_markers(all_markers) {
            this.all_frames_markers = all_markers;
        },
        stop() {
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }
            if (this.animationFrameId) {
                cancelAnimationFrame(this.animationFrameId);
                this.animationFrameId = null;
            }
            if (this.$refs.videoElement) {
                this.$refs.videoElement.pause();
                this.$refs.videoElement.removeAttribute('src');
                this.$refs.videoElement.load();
            }
            this.activeMode = null;
            this.previousMediaMode = null;
            this.markers = [];
            this.frameBuffer = [];
            this.frameMarkersMap = {};
            this.inferredKeyframes = [];
            this.displayedFrameIndex = -1;
            this.draggingKeypointId = null;
        }
    },
    beforeUnmount() {
        this.stop();
    }
}
