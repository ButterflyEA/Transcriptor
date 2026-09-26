# Host Capacity Warning - Implementation Plan

Warn users when the model they picked cannot fit in the memory of the machine
they are running on. Advisory only: never disable, never block, never refuse a
download.

## Why now: the crash that prompted this

On 2026-09-27 the desktop app was run on Omarchy (Arch) inside a **QEMU VM**
with **8 GB RAM and no VRAM** (no GPU passthrough, so no hardware Vulkan
adapter). Selecting a large model and starting transcription closed every
window with no error message in the terminal.

That signature is a `SIGKILL`, not a panic — the kernel killed the process, so
there is no Rust error to find. The most probable sequence:

1. `try_wgpu_device()` (`backends.rs:41`) probes by allocating a single-element
   tensor. That succeeds against a software adapter (llvmpipe/lavapipe) or
   fails and the code falls back to ndarray on CPU.
2. Either way the weights land in **system RAM**. Software Vulkan is not free
   memory — it is system RAM that the "GPU" borrows.
3. `large-v3` is ~6.2 GB resident as f32. On an 8 GB VM already running a
   desktop compositor, that does not fit.
4. The OOM killer terminated the process. All windows closed.

This is the case the feature exists to catch, and it is worth stating plainly:
**had this check existed, it would have warned before the crash.** It also
proves the RAM-only decision was right — VRAM detection would have reported
"fine", because there is no VRAM at all here. The constraint was RAM the whole
time.

## Global Constraints

- **Advisory only.** No hard block, no disabled options, no refusing to
  download. Estimates are rough and the app must never trap a user who knows
  something we do not.
- **System RAM only.** No VRAM detection. Agreed: free VRAM is not obtainable
  portably, and the case that actually bit us is RAM.
- **No new heavyweight dependency** beyond a single well-scoped crate.
- **Never show a scary "unknown".** If detection fails, the feature renders
  nothing and behaves exactly as it does today.
- Follow the existing patterns: `ModelSize` metadata lives in `config.rs`
  beside `repo_id()`; the UI is fed by the `Defaults` command.

## Decisions

| Question | Decision |
|---|---|
| Warn or block? | Warn. Never block. |
| RAM or VRAM? | System RAM only. |
| Size source | Real cached file size when downloaded, static table otherwise. |
| Dependency | `sysinfo` for cross-platform RAM. |
| Verdict input | `available`, with `total` shown for context. |

## File Structure

- `crates/whisper-burn/src/host.rs` **(new)** — host capacity detection and the
  pure recommendation function.
- `crates/whisper-burn/src/config.rs` — add `approx_resident_bytes()`.
- `crates/whisper-burn/src/download.rs` — expose the cached checkpoint path so
  the real size can be preferred over the table.
- `crates/whisper-burn/src/lib.rs` — export the `host` module.
- `crates/transcriptor/src-tauri/src/app.rs` — extend `Defaults`.
- `crates/transcriptor/ui/src/components/ParamsRail.tsx` — render the advice.

### Task 1: Add the approximate resident-size table

`ModelSize::approx_resident_bytes()` in `config.rs`, returning the f32 resident
footprint. Approximate parameter counts from the published configs:

| Model | Params | f32 resident |
|---|---|---|
| tiny | 39 M | ~0.16 GB |
| base | 74 M | ~0.30 GB |
| small | 244 M | ~0.98 GB |
| medium | 769 M | ~3.08 GB |
| large | 1550 M | ~6.20 GB |
| large-v2 | 1550 M | ~6.20 GB |
| large-v3 | 1550 M | ~6.20 GB |
| large-v3-turbo | 809 M | ~3.24 GB |
| ivrit-hebrew | 1550 M | ~6.20 GB |

Whisper's HF repos publish fp16; we compute in f32, hence 4 bytes/param. This
table is a **floor**, not a total: it excludes activations, the mel spectrogram
and the KV cache. Document that in the doc comment.

Tests: the table is monotonic non-decreasing across `ModelSize::ALL`; `tiny` is
under 1 GB; `large-v3` is over 6 GB. Those two assertions encode the crash we
are preventing.

### Task 2: Prefer the real checkpoint size when available

In `download.rs`, expose the cached `model.safetensors` path for a
`ModelSize` (it already owns `cache_dir`). In `host.rs`, add
`resident_bytes(size) -> u64` that returns the real file size when the
checkpoint is already downloaded, else `approx_resident_bytes()`.

Rationale: for a model the user already has, we should not guess. For one they
have not, the table is what lets us warn *before* a multi-gigabyte download.

Test: with a temp file in place, the real size is returned; with no file, the
table value is returned.

### Task 3: Detect host capacity

`host.rs`:

```rust
pub struct HostCapacity {
    pub total_bytes: u64,
    pub available_bytes: u64,
}
pub fn detect() -> Option<HostCapacity>;
```

`sysinfo`, refreshed twice (the first refresh reports zeroed values on some
platforms). Returns `None` on any failure — never panics, never returns a
plausible-looking zero. A capacity with `total_bytes == 0` is treated as
`None`, because a zero would otherwise recommend `tiny` and imply a broken
machine.

No test: hardware-dependent by nature. The rule is that it cannot fail loudly.

### Task 4: The recommendation function (pure)

```rust
pub fn fits(estimate: u64, available: u64) -> bool;
pub fn recommend(capacity: &HostCapacity, sizes: &[ModelSize]) -> Option<ModelSize>;
```

`fits` requires `estimate <= available * HEADROOM` with `HEADROOM = 0.7`.
Rationale: the table is a floor, activations are not counted, and a
transcription that thrashes swap is not "supported" even if it technically
fits. `recommend` returns the largest model that fits, `None` if even `tiny`
does not.

Both are pure, so both are fully testable with synthetic numbers — no hardware
in the loop. Tests cover: exact boundary, just over boundary, empty input,
every-model-fits, nothing-fits, and the ordering (a bigger model must never be
recommended when a smaller one also fits).

### Task 5: Expose it through `Defaults`

Extend `Defaults` (`app.rs:20`) with:

- `totalRamBytes`, `availableRamBytes`
- `recommendedModel: Option<String>`
- `residentBytes: HashMap<String, u64>` per model

Computed in `get_defaults()`. When `detect()` is `None`, every field is
`None`/empty and the UI shows nothing — the additive-optional shape means
existing UI code keeps working untouched.

### Task 6: Render the advice

`ParamsRail.tsx`, under the model `<select>`:

- A single muted line: `16 GB RAM — 11 GB available · recommended: medium or
  below`
- Options whose estimate exceeds the headroom get a `⚠` suffix and dimmed
  text: `large-v3 ⚠ needs ~6.2 GB`
- The recommendation is advisory text only. The dropdown is never disabled and
  no option is ever removed.
- If capacity is absent, render nothing at all.

Keep it to one line plus suffixes. This is a params rail, not a dashboard.

### Task 7: Verify

- `cargo test --workspace` green, including the new pure-function tests.
- `cargo clippy --workspace --all-targets` clean.
- Manual check on the 8 GB VM: selecting `large-v3` shows the warning;
  selecting `small` does not; nothing is blocked.

## Risks

- **The estimate is a floor.** It will under-predict. Documented, and the
  reason the headroom exists, but a user who is right at the edge can still
  get OOM-killed. The feature reduces the odds; it does not eliminate them.
- **A cgroup or container memory cap makes `available` a lie.** The host may
  report 32 GB free while the process is capped at 4 GB. Not detectable from
  inside without reading cgroup limits; out of scope, named here rather than
  papered over.
- **Swap makes "did it crash" ambiguous.** A machine with enough swap will not
  be OOM-killed, it will just crawl. The warning is still correct.
- **`sysinfo` adds a dependency** to a workspace that is otherwise
  deliberately lean. Accepted in exchange for not hand-rolling two platform
  paths.

## Out of scope

- Blocking, disabling, or gating downloads.
- VRAM / GPU memory detection.
- Live memory sampling during a run.
- cgroup-aware capacity reporting.
- Any change to the CLI's behaviour.
