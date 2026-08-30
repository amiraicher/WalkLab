from nicegui import ui
from . import annotation, pose, settings, metrics

def run_app():
    with ui.header().classes('bg-primary text-white'):
        ui.label('WalkingLab Gait Tracking').classes('text-2xl font-bold')
        with ui.tabs() as tabs:
            annotation_tab = ui.tab('Annotation')
            pose_tab = ui.tab('Pose Estimation')
            settings_tab = ui.tab('Settings & Input')
            metrics_tab = ui.tab('Metrics')
            
    with ui.tab_panels(tabs, value=annotation_tab).classes('w-full'):
        with ui.tab_panel(annotation_tab):
            annotation.render()
        with ui.tab_panel(pose_tab):
            pose.render()
        with ui.tab_panel(settings_tab):
            settings.render()
        with ui.tab_panel(metrics_tab):
            metrics.render()

    ui.run(title='WalkingLab')
