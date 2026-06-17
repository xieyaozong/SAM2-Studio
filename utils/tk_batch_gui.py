from __future__ import annotations

from pathlib import Path
from utils.batch import process_batch
from utils.config import EXPORT_FORMATS, MODEL_PRESETS, SamBatchConfig

import queue
import threading


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("Value must be greater than 0.")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError("Value must be 0 or greater.")
    return parsed


def run_batch_gui() -> None:
    from tkinter import filedialog, messagebox, ttk
    import tkinter as tk

    root = tk.Tk()
    root.title("SAM 2 Batch Image App")
    root.geometry("760x520")

    messages: queue.Queue[tuple[str, str]] = queue.Queue()
    running = tk.BooleanVar(value=False)

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    model_var = tk.StringVar(value="large")
    device_var = tk.StringVar(value="auto")
    recursive_var = tk.BooleanVar(value=True)
    individual_masks_var = tk.BooleanVar(value=False)
    export_format_var = tk.StringVar(value="yolo")
    class_id_var = tk.StringVar(value="0")
    skip_existing_var = tk.BooleanVar(value=False)
    points_var = tk.StringVar(value="32")

    main = ttk.Frame(root, padding=16)
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(10, weight=1)

    def browse_input() -> None:
        selected = filedialog.askdirectory(title="Choose input image folder")
        if selected:
            input_var.set(selected)
            if not output_var.get():
                output_var.set(str(Path(selected) / "sam2_results"))

    def browse_output() -> None:
        selected = filedialog.askdirectory(title="Choose output folder")
        if selected:
            output_var.set(selected)

    def append_log(message: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", f"{message}\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    ttk.Label(main, text="Input folder").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Entry(main, textvariable=input_var).grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Button(main, text="Browse", command=browse_input).grid(row=0, column=2, padx=(8, 0), pady=4)

    ttk.Label(main, text="Output folder").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Entry(main, textvariable=output_var).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(main, text="Browse", command=browse_output).grid(row=1, column=2, padx=(8, 0), pady=4)

    ttk.Label(main, text="Model").grid(row=2, column=0, sticky="w", pady=4)
    ttk.Combobox(
        main,
        textvariable=model_var,
        values=tuple(MODEL_PRESETS.keys()),
        state="readonly",
        width=16,
    ).grid(row=2, column=1, sticky="w", pady=4)

    ttk.Label(main, text="Device").grid(row=3, column=0, sticky="w", pady=4)
    ttk.Combobox(
        main,
        textvariable=device_var,
        values=("auto", "cuda", "cuda:0", "cpu"),
        state="readonly",
        width=16,
    ).grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(main, text="Points per side").grid(row=4, column=0, sticky="w", pady=4)
    ttk.Entry(main, textvariable=points_var, width=18).grid(row=4, column=1, sticky="w", pady=4)

    ttk.Label(main, text="YOLO class id").grid(row=5, column=0, sticky="w", pady=4)
    ttk.Entry(main, textvariable=class_id_var, width=18).grid(row=5, column=1, sticky="w", pady=4)

    ttk.Label(main, text="Export format").grid(row=6, column=0, sticky="w", pady=4)
    ttk.Combobox(
        main,
        textvariable=export_format_var,
        values=EXPORT_FORMATS,
        state="readonly",
        width=16,
    ).grid(row=6, column=1, sticky="w", pady=4)

    options = ttk.Frame(main)
    options.grid(row=7, column=1, sticky="w", pady=8)
    ttk.Checkbutton(options, text="Recursive", variable=recursive_var).pack(side="left", padx=(0, 12))
    ttk.Checkbutton(options, text="Save individual masks", variable=individual_masks_var).pack(
        side="left", padx=(0, 12)
    )
    ttk.Checkbutton(options, text="Skip existing", variable=skip_existing_var).pack(side="left")

    start_button = ttk.Button(main, text="Start")
    start_button.grid(row=8, column=1, sticky="w", pady=(4, 12))

    ttk.Label(main, text="Log").grid(row=9, column=0, sticky="nw", pady=4)
    log_box = tk.Text(main, height=14, wrap="word", state="disabled")
    log_box.grid(row=10, column=0, columnspan=3, sticky="nsew")

    scrollbar = ttk.Scrollbar(main, orient="vertical", command=log_box.yview)
    scrollbar.grid(row=10, column=3, sticky="ns")
    log_box.configure(yscrollcommand=scrollbar.set)

    def start() -> None:
        if running.get():
            return

        input_path = input_var.get().strip()
        output_dir = output_var.get().strip()
        if not input_path or not output_dir:
            messagebox.showerror("Missing folders", "Choose both input and output folders.")
            return

        try:
            points_per_side = _positive_int(points_var.get().strip())
            class_id = _non_negative_int(class_id_var.get().strip())
        except Exception as exc:
            messagebox.showerror("Invalid setting", str(exc))
            return

        config = SamBatchConfig(
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            model_size=model_var.get(),
            device=device_var.get(),
            recursive=recursive_var.get(),
            points_per_side=points_per_side,
            save_individual_masks=individual_masks_var.get(),
            export_format=export_format_var.get(),
            yolo_class_id=class_id,
            skip_existing=skip_existing_var.get(),
        )

        running.set(True)
        start_button.configure(state="disabled")
        append_log("Starting batch segmentation...")

        def worker() -> None:
            try:
                results = process_batch(
                    config,
                    log=lambda message: messages.put(("log", message)),
                    progress=False,
                )
                failed = sum(1 for result in results if result.status == "error")
                messages.put(("done", f"Finished. Errors: {failed}."))
            except Exception as exc:
                messages.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def poll_messages() -> None:
        try:
            while True:
                message_type, message = messages.get_nowait()
                append_log(message)
                if message_type in {"done", "error"}:
                    running.set(False)
                    start_button.configure(state="normal")
                    if message_type == "error":
                        messagebox.showerror("SAM 2 batch app", message)
        except queue.Empty:
            pass
        root.after(150, poll_messages)

    start_button.configure(command=start)
    poll_messages()
    root.mainloop()
