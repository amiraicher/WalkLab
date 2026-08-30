from nicegui import ui
from walkinglab.ui.components.video_tracker import VideoTracker

def render():
    ui.label('Annotation Toolkit').classes('text-3xl font-bold mb-6 text-slate-800')
    
    with ui.grid(columns=3).classes('w-full gap-6 h-[700px]'):
        # Left Panel (2/3 width): Unified media frame
        with ui.column().classes('col-span-2 h-full border border-gray-200 rounded-xl shadow-lg bg-white overflow-hidden'):
            tracker = VideoTracker(
                on_upload=lambda e: ui.notify(f'Loaded video {e["name"]} for annotation.', type='info')
            ).classes('w-full h-full')
            
        # Right Panel (1/3 width): Annotation controls
        with ui.column().classes('col-span-1 h-full border border-gray-200 rounded-xl shadow-lg p-6 bg-slate-50 flex flex-col'):
            ui.label('Keypoint Corrections').classes('text-xl font-semibold mb-4 text-slate-700 border-b pb-2 w-full')
            
            ui.label('Use this toolkit to manually review and correct keypoints across frames. Upload a video to begin.').classes('text-sm text-slate-600 mb-6')
            
            # Simulated keypoint list for annotation
            with ui.scroll_area().classes('w-full flex-grow border border-gray-200 bg-white rounded p-2'):
                keypoints = ['Nose', 'L. Shoulder', 'R. Shoulder', 'L. Elbow', 'R. Elbow', 'L. Hip', 'R. Hip', 'L. Knee', 'R. Knee', 'L. Ankle', 'R. Ankle']
                for kp in keypoints:
                    with ui.row().classes('w-full justify-between items-center py-2 border-b border-gray-100 last:border-0 hover:bg-slate-50 px-2 rounded'):
                        ui.label(kp).classes('text-sm font-medium text-slate-700')
                        with ui.row().classes('gap-2'):
                            ui.button(icon='visibility', on_click=lambda k=kp: ui.notify(f'Toggled visibility for {k}')).props('flat dense size=sm color=slate-400 hover:color-blue-500')
                            ui.button(icon='edit', on_click=lambda k=kp: ui.notify(f'Editing {k}')).props('flat dense size=sm color=slate-400 hover:color-blue-500')
            
            with ui.row().classes('w-full mt-4 justify-end gap-2'):
                ui.button('Export Annotations', icon='download', on_click=lambda: ui.notify('Annotations exported.', type='positive')).props('outline color=primary')
