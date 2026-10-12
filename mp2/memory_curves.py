"""MP2 Problem 4 -- ResNet-18 memory and speed on CUDA or MPS. FULLY PROVIDED.

    python memory_curves.py                     # ResNet-18, three regimes, mean of three runs
    python memory_curves.py --batches 1,4,16,64 # your own batch sizes
    python memory_curves.py --runs 3            # three complete sweeps (default); use at least 3 for the assignment
    python memory_curves.py --layers            # also print ResNet-18 stage outputs and saved storages at batch 1
    python memory_curves.py --device mps        # Mac MPS; auto (default) prefers CUDA, then MPS

Outputs (in --out-dir, default '.'):  memory_curves_resnet.png  (max allocated memory | speed, the three regimes
    on the same axes), memory_curves_resnet.csv (averages), and memory_curves_resnet_runs.csv (individual runs).

For ResNet-18 with a 10-class output and each of three regimes
    eval  : model.eval() + torch.no_grad() forward                     (inference)
    last  : train only the last linear layer; backbone parameters frozen, BatchNorm statistics update
    full  : train every parameter                                        (full fine-tuning)
and each batch size, the first iteration and three continuing iterations are profiled, then warm iterations are timed.
The complete sweep is repeated three times by default; use at least three runs for the assignment.
Each point uses a fresh model and inputs. The plotted memory
is the arithmetic mean of the per-run maxima; throughput is the arithmetic mean of per-run images/second.
    max_allocated_MiB : maximum allocated memory across those four iterations, including resident model state
                and input/labels. Inference measures forward; training measures forward, backward and Adam step.
                CUDA resets the allocator peak counter before each iteration and reads max_memory_allocated()
                afterwards: the allocator tracks the whole interval, not just the final allocation or layer outputs.
                MPS samples current allocations after torch operations and saved-tensor accesses; its maximum is
                approximate and may miss allocations inside backend operations. Timing runs without sampling.
                This is allocated memory tracked by PyTorch, not reserved memory or total device usage.
    params_MiB, buffers_MiB, trainable_MiB, input_MiB, grads_MiB, optimizer_MiB : supporting tensor sizes in the CSV;
                these describe individual components, not an additional memory-usage total. Adam scalar counters
                are excluded from optimizer_MiB. Use --layers for stage outputs and backward-saved storages.
    ms_iter, img_s : median warm iteration time and corresponding images/second within each run, averaged separately
                across runs. Thus mean img_s need not equal batch size divided by mean ms_iter.
    n_runs : successful measurements contributing to each averaged point; skipped or failed points are not zeros.
Inputs and parameters are float32; inputs have shape (batch, 3, 256, 256). Training uses cross-entropy and Adam.
In last-layer training the backbone parameters are frozen, but train-mode BatchNorm statistics still update.
"""
import argparse, csv, gc, os, subprocess, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

MiB = 1024 ** 2
DEV = "cuda"
device = torch.device(DEV)


# ----------------------------------------------------------------------------- model
def make_model():
    from torchvision.models import resnet18
    m = resnet18(weights=None)              # same shapes as Section 8; random weights are fine for memory/speed
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def configure_regime(model, regime):
    if regime not in ("eval", "last", "full"):
        raise ValueError(f"unknown regime: {regime}")
    model.train(regime != "eval")
    model.zero_grad(set_to_none=True)
    for p in model.parameters(): p.requires_grad_(regime == "full")
    if regime == "last":
        for p in model.fc.parameters(): p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


# ----------------------------------------------------------------------------- device helpers
def sync():
    if DEV == "cuda": torch.cuda.synchronize()
    elif DEV == "mps": torch.mps.synchronize()

def clear():
    gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()
    elif DEV == "mps": torch.mps.empty_cache()

def tensor_bytes(ts):
    return sum(t.numel() * t.element_size() for t in ts if torch.is_tensor(t))


def detect_gpu_name():
    if DEV == "cuda":
        return torch.cuda.get_device_name()
    if sys.platform == "darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                stderr=subprocess.DEVNULL, timeout=2).strip()
            if chip.startswith("Apple"):
                return f"{chip} GPU"
        except (OSError, subprocess.SubprocessError):
            pass
    return "Apple GPU"


class PeakSampler(torch.overrides.TorchFunctionMode):
    """Approximate the MPS maximum with samples after torch operations and saved-tensor accesses."""
    def __init__(self):
        super().__init__()
        self.hi = 0
        self.ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.sample()
        return out

    def sample(self):
        sync()
        self.hi = max(self.hi, torch.mps.current_allocated_memory())

    def _pack(self, tensor): self.sample(); return tensor
    def _unpack(self, tensor): self.sample(); return tensor
    def __enter__(self): super().__enter__(); self.ctx.__enter__(); return self
    def __exit__(self, *args): self.ctx.__exit__(*args); super().__exit__(*args)


def saved_for_backward_inventory(model, x, y):
    """Unique saved storages, excluding parameters, buffers and x; a row describes the first saved view.

    Keeping storage objects alive prevents address reuse from merging distinct saved tensors.
    Backend-specific pooling indices and loss auxiliaries are included when saved by autograd.
    """
    keep = {(t.device, t.untyped_storage().data_ptr()) for t in list(model.parameters()) + list(model.buffers()) + [x]}
    seen = {}
    def pack(t):
        s = t.untyped_storage()
        key = (t.device, s.data_ptr())
        if key not in keep and key not in seen:
            seen[key] = (s, dict(shape=tuple(t.shape), dtype=str(t.dtype), bytes=s.nbytes(), MiB=s.nbytes() / MiB))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = F.cross_entropy(model(x), y)
    loss.backward(); model.zero_grad(set_to_none=True)
    return [row for storage, row in seen.values()]


def peak_iteration(iteration):
    """Return the CUDA allocator maximum or approximate MPS maximum for a complete iteration."""
    if DEV == "mps":
        with PeakSampler() as sampler:
            sampler.sample(); iteration(); sampler.sample()
        return sampler.hi
    sync(); torch.cuda.reset_peak_memory_stats()
    iteration(); sync()
    return torch.cuda.max_memory_allocated()


# ----------------------------------------------------------------------------- one measurement
def measure(model, regime, x, y, n_timed=None):
    """Profile the first and three continuing iterations, then time synchronized warm iterations."""
    if n_timed is not None and n_timed < 1:
        raise ValueError("n_timed must be positive")
    trainable = configure_regime(model, regime)
    params = tensor_bytes(model.parameters())
    buffers = tensor_bytes(model.buffers())
    clear()
    r = dict(regime=regime, bs=x.shape[0], params_MiB=params / MiB, buffers_MiB=buffers / MiB,
             trainable_MiB=tensor_bytes(trainable) / MiB, input_MiB=tensor_bytes([x]) / MiB)
    opt = None
    if regime == "eval":
        def iteration():
            with torch.no_grad(): return model(x)
    else:
        opt = torch.optim.Adam(trainable, lr=1e-4)
        def iteration():
            opt.zero_grad(set_to_none=True); loss = F.cross_entropy(model(x), y); loss.backward(); opt.step(); return loss
    peak = max(peak_iteration(iteration) for _ in range(4))  # first step and three continuing steps, including Adam state
    r["max_allocated_MiB"] = peak / MiB
    n = n_timed if n_timed is not None else (3 if x.shape[0] >= 256 else 7)
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter(); iteration(); sync(); ts.append(time.perf_counter() - t0)
    r["ms_iter"] = sorted(ts)[len(ts) // 2] * 1000
    r["img_s"] = x.shape[0] / (r["ms_iter"] / 1000)
    r["grads_MiB"] = tensor_bytes(p.grad for p in trainable) / MiB
    r["optimizer_MiB"] = 0 if opt is None else sum(
        tensor_bytes(st.get(k) for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")) for st in opt.state.values()) / MiB
    return r


# ----------------------------------------------------------------------------- stage table
def layer_table(bs=1):
    """Print ResNet stage outputs and saved-storage inventories using a small diagnostic batch.

    A stage row summarizes its output, not the intermediate tensors inside residual blocks.
    ResNet flattens with torch.flatten, so the fc input hook records that view explicitly.
    """
    model = make_model().to(device); model.eval()
    x = torch.randn(bs, 3, 256, 256, device=device)
    rows, hooks, storage_ptrs = [], [], {}
    for name, module in model.named_children():
        def hook(mod, inp, out, name=name):
            if not torch.is_tensor(out): return
            shared = bool(inp and torch.is_tensor(inp[0]) and
                          out.untyped_storage().data_ptr() == inp[0].untyped_storage().data_ptr())
            rows.append(dict(name=name, type=mod.__class__.__name__, shape=tuple(out.shape), dtype=str(out.dtype),
                             MiB=tensor_bytes([out]) / MiB, parameters=sum(p.numel() for p in mod.parameters()),
                             shares_input_storage=shared))
            storage_ptrs[name] = out.untyped_storage().data_ptr()
        hooks.append(module.register_forward_hook(hook))
    def flatten_hook(mod, inp):
        out = inp[0]
        rows.append(dict(name="flatten", type="torch.flatten", shape=tuple(out.shape), dtype=str(out.dtype),
                         MiB=tensor_bytes([out]) / MiB, parameters=0,
                         shares_input_storage=out.untyped_storage().data_ptr() == storage_ptrs["avgpool"]))
    hooks.append(model.fc.register_forward_pre_hook(flatten_hook))
    try:
        with torch.no_grad(): model(x)
    finally:
        for h in hooks: h.remove()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\nResNet-18: stage outputs at batch {bs}; input {tuple(x.shape)} float32 = {tensor_bytes([x])/MiB:.6f} MiB")
    print(f"{'name':<12}{'stage':<18}{'output shape':<26}{'dtype':<14}{'output MiB':>12}{'parameters':>12}  storage")
    for row in rows:
        alias = "shares input (in-place or view)" if row["shares_input_storage"] else ""
        print(f"{row['name']:<12}{row['type']:<18}{str(row['shape']):<26}{row['dtype']:<14}"
              f"{row['MiB']:>12.6f}{row['parameters']:>12,}  {alias}")
    print("layer1-layer4 summarize residual stages; their internal activations are not listed. Output sizes are not a memory maximum.")
    print("The stem ReLU is in-place; flatten is a view of avgpool. Scale the batch dimension and output MiB for other batch sizes.")
    print(f"parameters: {n_par:,} = {tensor_bytes(model.parameters())/MiB:.6f} MiB; "
          f"buffers: {tensor_bytes(model.buffers())/MiB:.6f} MiB")
    y = torch.randint(0, 10, (bs,), device=device)
    for regime in ("last", "full"):
        trainable = configure_regime(model, regime)
        inventory = saved_for_backward_inventory(model, x, y)
        print(f"\nResNet-18 {regime}, batch {bs}: total {n_par:,}; trainable {sum(p.numel() for p in trainable):,} parameters")
        print("unique saved storages (parameters, buffers and image input excluded; first saved view shown):")
        print(f"{'storage':<10}{'shape':<26}{'dtype':<14}{'bytes':>12}{'MiB':>13}")
        for i, row in enumerate(inventory, 1):
            print(f"{i:<10}{str(row['shape']):<26}{row['dtype']:<14}{row['bytes']:>12,}{row['MiB']:>13.6f}")
        print(f"saved total: {sum(row['bytes'] for row in inventory):,} bytes = {sum(row['MiB'] for row in inventory):.6f} MiB")
    print("Pool indices and loss auxiliaries are included when the backend saves them; dtypes/storage sizes can differ by backend.")
    del model, x, y, trainable; clear()
    return rows


# ----------------------------------------------------------------------------- plotting
def memory_label():
    return "max allocated memory (approx.)" if DEV == "mps" else "max allocated memory"


def memory_note():
    method = ("MPS: approximate allocation samples; may miss allocations inside backend operations."
              if DEV == "mps" else "CUDA allocator maximum; includes model state and input, excludes unused reserved memory.")
    return ("Maximum over the first + next 3 iterations: forward for inference; forward + backward + Adam for training.\n"
            + method)


def plot(rows, out_png, gpu_name=None, run_count=None):
    """ResNet-18 maximum allocated memory and speed, including empty runs."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    colors = {"eval": "#2a78d6", "last": "#2aa876", "full": "#eb6834"}
    labels = {"eval": "inference (eval + no_grad)", "last": "fine-tune last layer only", "full": "full fine-tune"}
    allbs = sorted({r["bs"] for r in rows}) or [1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    for ax, (key, title, unit) in zip(axes, [("max_allocated_MiB", memory_label(), "MiB"), ("img_s", "speed", "images / s")]):
        for reg in ("eval", "last", "full"):
            rs = sorted([r for r in rows if r["regime"] == reg and r.get(key) is not None], key=lambda r: r["bs"])
            if not rs: continue
            style = "--" if (reg == "last" and key == "max_allocated_MiB") else "-"     # distinguish overlapping curves
            z = {"full": 2, "eval": 3, "last": 4}[reg]                                   # coinciding lines: full under, eval, dashed last on top
            ax.plot([r["bs"] for r in rs], [r[key] for r in rs], marker="o", color=colors[reg], lw=2 if reg != "eval" else 3, ms=5, label=labels[reg], zorder=z,
                    linestyle=(0, (4, 3)) if style == "--" else "-")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold"); ax.set_ylabel(unit); ax.set_xlabel("batch size")
        if key == "max_allocated_MiB":
            ax.set_xscale("linear"); ax.set_yscale("linear")
            ax.set_xlim(0, max(allbs) * 1.05)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, integer=True))
        else:
            ax.set_xscale("log", base=2); ax.set_xticks(allbs); ax.set_xticklabels([str(b) for b in allbs])
            ax.set_xlabel("batch size (log\u2082 scale)")
        ax.set_ylim(0, max((r[key] for r in rows if r.get(key) is not None), default=1) * 1.15)
        if not any(r.get(key) is not None for r in rows):
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "no successful measurements", transform=ax.transAxes, ha="center", color="#7a8494")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", color="#e9ecf1"); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles: axes[0].legend(handles, labels, frameon=False, fontsize=9, loc="upper left")
    hardware = f"{gpu_name}; " if gpu_name else ""
    fig.suptitle(f"ResNet-18: memory and speed vs batch size\n{hardware}{DEV}, torch {torch.__version__}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    counts = sorted({r["n_runs"] for r in rows if "n_runs" in r})
    note = memory_note()
    if run_count is not None or counts:
        requested = run_count if run_count is not None else max(counts)
        if not rows:
            average_note = f"Requested {requested} runs; no successful measurements."
        else:
            average_note = (f"Mean of {requested} runs: memory = mean of per-run maxima; "
                            "speed = mean of per-run images/s.")
            if counts and counts != [requested]:
                actual = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
                average_note = (f"{requested} runs requested; {actual} successful runs per point (see CSV n_runs). "
                                "Memory and speed averaged separately.")
        note = average_note + "\n" + note
    fig.text(0.01, 0.01, note, fontsize=8)
    fig.tight_layout(rect=(0, 0.11 if run_count is not None or counts else 0.07, 1, 0.99))
    fig.savefig(out_png, facecolor="white"); plt.close(fig)


# ----------------------------------------------------------------------------- main
def batch_cap(regime):
    """Conservative attempt limit, not a fit guarantee; backend/workspace overhead varies."""
    if DEV == "cuda":
        mem, _ = torch.cuda.mem_get_info()
    else:
        mem = max(0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory())
    per_img = {"eval": 12, "last": 12, "full": 36}[regime] * MiB
    return max(0, int(0.6 * mem / per_img))


def parse_batches(value):
    try:
        batches = sorted({int(part) for part in value.split(",")})
        if not batches or batches[0] <= 0: raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("batches must be comma-separated positive integers") from None
    return batches


def positive_int(value):
    try:
        result = int(value)
        if result < 1: raise ValueError
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("runs must be a positive integer") from None
    return result


def is_oom_error(error):
    if isinstance(error, (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)): return True
    if not isinstance(error, RuntimeError): return False
    message = str(error).lower()
    return message.startswith(("cuda out of memory", "cuda error: out of memory", "mps backend out of memory", "mps out of memory"))


CSV_FIELDS = ["model", "regime", "bs", "max_allocated_MiB", "ms_iter", "img_s",
              "params_MiB", "trainable_MiB", "buffers_MiB", "input_MiB", "grads_MiB", "optimizer_MiB"]
RUN_CSV_FIELDS = CSV_FIELDS + ["run"]
AVERAGE_CSV_FIELDS = CSV_FIELDS + ["n_runs"]


def average_runs(rows):
    """Arithmetic means of successful runs for each model/regime/batch; missing runs are not zeros."""
    groups = {}
    for row in rows:
        key = (row["model"], row["regime"], row["bs"])
        groups.setdefault(key, []).append(row)
    averages = []
    for (model, regime, bs), group in groups.items():
        average = dict(model=model, regime=regime, bs=bs, n_runs=len(group))
        for field in CSV_FIELDS[3:]:
            values = [row[field] for row in group if row[field] is not None]
            average[field] = sum(values) / len(values) if values else None
        averages.append(average)
    return averages


def write_csv(rows, path, fields=None):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS if fields is None else fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    global device, DEV
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", type=parse_batches, default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--runs", type=positive_int, default=3, help="number of complete sweeps to average (default: 3; assignment: at least 3)")
    ap.add_argument("--device", choices=["auto", "cuda", "mps"], default="auto")
    ap.add_argument("--layers", action="store_true", help="also print ResNet-18 stage outputs and saved-storage inventories at batch 1")
    args = ap.parse_args()
    chosen = args.device
    if chosen == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
    if chosen is None:
        ap.error("CUDA or Mac MPS GPU required. In Colab, select Runtime > Change runtime type > GPU; or run on a Mac with MPS-enabled PyTorch.")
    if chosen == "cuda" and not torch.cuda.is_available(): ap.error("CUDA is not available; select a Colab GPU runtime or use --device mps on a supported Mac.")
    if chosen == "mps" and not torch.backends.mps.is_available(): ap.error("MPS is not available; use a supported Mac or select a Colab CUDA GPU runtime.")
    device = torch.device(chosen); DEV = chosen
    gpu = detect_gpu_name()
    print(f"device: {gpu} ({device})  torch {torch.__version__}")
    print("float32 parameters and image input (B, 3, 256, 256); cross-entropy; Adam lr=1e-4, two float32 moments per trainable parameter.")
    print("eval: no_grad, frozen BN statistics; last: backbone parameters frozen, train-mode BN statistics update; full: all parameters train, BN statistics update.")
    print("max allocated memory = maximum over the first + next 3 iterations, including measured model state and input/labels.")
    print("Inference measures forward; training measures forward + backward + Adam step. Timed iterations follow warmup.")
    print(f"Repeat the complete sweep {args.runs} times with a fresh model at each point. Plot arithmetic means of per-run memory maxima and per-run throughput.")
    if DEV == "cuda": print("CUDA: reset_peak_memory_stats() before each iteration, max_memory_allocated() afterwards; the allocator tracks the entire interval.")
    else: print("MPS: approximate maximum sampled after torch operations and saved-tensor accesses; may miss allocations inside backend operations. Timing runs without sampling.")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.layers:
        try:
            layer_table()
        except Exception as error:
            if not is_oom_error(error): raise
            print(f"ResNet-18 stage table skipped after out of memory: {error}", flush=True)
        clear()                          # the exception and its traceback have left scope before cleanup
    rows = []
    csv_path = os.path.join(args.out_dir, "memory_curves_resnet.csv")
    runs_path = os.path.join(args.out_dir, "memory_curves_resnet_runs.csv")
    write_csv([], csv_path, AVERAGE_CSV_FIELDS)
    write_csv([], runs_path, RUN_CSV_FIELDS)
    for run in range(1, args.runs + 1):
        print(f"\nRun {run}/{args.runs}", flush=True)
        for regime in ("eval", "last", "full"):
            cap, described = batch_cap(regime), False
            for bs in args.batches:
                if bs > cap:
                    print(f"resnet {regime:5s} bs={bs:<4} skipped (conservative attempt cap {cap})"); continue
                model = x = y = None
                failure = None
                try:
                    model = make_model().to(device)
                    x = torch.randn(bs, 3, 256, 256, device=device)
                    y = torch.randint(0, 10, (bs,), device=device)
                    r = measure(model, regime, x, y)
                    total = sum(p.numel() for p in model.parameters())
                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                except Exception as error:
                    if not is_oom_error(error): raise
                    failure = str(error)
                finally:
                    model = x = y = None
                clear()                  # no failed tensors or exception traceback retained between attempts
                if failure is not None:
                    print(f"resnet {regime:5s} bs={bs:<4} out of memory: {failure}", flush=True)
                    break                # remaining batches are larger; keep earlier successful CSV rows
                if not described:
                    print(f"ResNet-18 {regime}: total {total:,}; trainable {trainable:,} parameters; buffers {r['buffers_MiB']:.6f} MiB")
                    described = True
                r["model"] = "resnet"; r["run"] = run; rows.append(r)
                write_csv(rows, runs_path, RUN_CSV_FIELDS)
                write_csv(average_runs(rows), csv_path, AVERAGE_CSV_FIELDS)
                value = "    n/a" if r["max_allocated_MiB"] is None else f"{r['max_allocated_MiB']:7.1f} MiB"
                print(f"resnet {regime:5s} bs={bs:<4} {memory_label()} {value}  "
                      f"{r['ms_iter']:8.1f} ms/iter  {r['img_s']:7.0f} img/s", flush=True)
    averages = average_runs(rows)
    incomplete = [r for r in averages if r["n_runs"] != args.runs]
    if incomplete:
        print(f"{len(incomplete)} points have fewer than {args.runs} successful runs; see n_runs in the averaged CSV.")
    png_path = os.path.join(args.out_dir, "memory_curves_resnet.png")
    plot(averages, png_path, gpu_name=gpu, run_count=args.runs)
    print(f"wrote {csv_path}, {runs_path}, and {png_path}")


if __name__ == "__main__":
    main()
