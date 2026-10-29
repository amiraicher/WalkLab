"""
Clinical Pose Assessment page view for real-time and offline gait tracking.
"""
from nicegui import run, ui

import walkinglab.backend.inference as inference
from walkinglab.ui.components.video_tracker import VideoTracker


def render() -> None:
    """
    Renders the Clinical Pose Assessment UI panel, video tracker, and guided stepper.
    """
    ui.label("Clinical Pose Assessment").classes(
        "text-3xl font-bold mb-6 text-slate-800"
    )

    with ui.grid(columns=3).classes("w-full gap-6 h-[700px]"):
        # Left Panel (2/3 width): Unified media frame
        with ui.column().classes(
            "col-span-2 h-full border border-gray-200 rounded-xl shadow-lg bg-white overflow-hidden"
        ):
            async def handle_frame(e: dict | str) -> None:
                result = await run.cpu_bound(inference.process_frame, e)
                tracker.run_method("update_markers", result)

            def handle_upload_start(e: dict) -> None:
                ui.notify(
                    f'Batch processing initiated for: {e["name"]}', type="info"
                )

            def handle_upload_success(e: dict) -> None:
                ui.notify("Batch processing complete.", type="positive")

            tracker = VideoTracker(
                on_frame=handle_frame,
                on_file_uploaded_start=handle_upload_start,
                on_file_uploaded_success=handle_upload_success,
            ).classes("w-full h-full")

        # Right Panel (1/3 width): Wizard-style instruction panel
        with ui.column().classes(
            "col-span-1 h-full border border-gray-200 rounded-xl shadow-lg p-6 bg-slate-50"
        ):
            ui.label("Guided Walkthrough").classes(
                "text-xl font-semibold mb-6 text-slate-700 border-b pb-2 w-full"
            )

            with ui.stepper().props("vertical").classes(
                "w-full bg-transparent shadow-none p-0"
            ) as stepper:
                with ui.step("1. Capture Data"):
                    ui.label(
                        "Start the camera for real-time tracking, or upload a video file for offline analysis."
                    ).classes("text-slate-600")
                    with ui.row().classes("mt-4"):
                        ui.button("Next Step", on_click=stepper.next).props(
                            "color=primary"
                        )

                with ui.step("2. Run Inference"):
                    ui.label(
                        "System is extracting biomechanical keypoints from the video source..."
                    ).classes("text-slate-600")
                    with ui.row().classes("mt-4 items-center gap-3"):
                        ui.spinner("dots", size="lg").classes("text-primary")
                        ui.label("Processing").classes(
                            "text-sm text-slate-500 italic"
                        )
                    with ui.row().classes("mt-6"):
                        ui.button("Back", on_click=stepper.previous).props("flat")
                        ui.button("Next Step", on_click=stepper.next).props(
                            "color=primary"
                        )

                with ui.step("3. Review Metrics"):
                    ui.label("Gait analysis is complete.").classes(
                        "mb-4 text-slate-600"
                    )

                    with ui.card().classes(
                        "w-full bg-blue-50/50 border border-blue-100 shadow-sm"
                    ):
                        ui.label("Clinical Metrics").classes(
                            "font-semibold text-blue-900 mb-2"
                        )
                        with ui.row().classes(
                            "w-full justify-between border-b border-blue-100 pb-1"
                        ):
                            ui.label("Stride Length").classes(
                                "text-sm text-slate-600"
                            )
                            ui.label("1.25 m").classes(
                                "font-mono font-medium text-blue-800"
                            )
                        with ui.row().classes(
                            "w-full justify-between border-b border-blue-100 py-1"
                        ):
                            ui.label("Cadence").classes("text-sm text-slate-600")
                            ui.label("108 steps/min").classes(
                                "font-mono font-medium text-blue-800"
                            )
                        with ui.row().classes("w-full justify-between pt-1"):
                            ui.label("Gait Velocity").classes(
                                "text-sm text-slate-600"
                            )
                            ui.label("1.1 m/s").classes(
                                "font-mono font-medium text-blue-800"
                            )

                    with ui.row().classes("mt-6"):
                        ui.button("Back", on_click=stepper.previous).props("flat")
                        ui.button(
                            "Finish Session",
                            on_click=lambda: ui.notify(
                                "Assessment Saved successfully",
                                type="positive",
                                position="top-right",
                            ),
                        ).props("color=positive")
