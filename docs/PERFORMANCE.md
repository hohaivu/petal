# How Petal is fast, and how it stays honest

Petal was tuned in measured rounds: one change at a time, kept only if it beat the previous best on
an A/B benchmark *and* passed a correctness gate. This page summarises what worked, what didn't, and
how the numbers are checked. Figures are from an Apple Silicon MacBook (10 cores, APFS, internal SSD).

## The scan

**Where the time goes.** A full scan is almost entirely kernel time spent reading directory
metadata. Per-file attributes are essentially free once you ask for them in bulk; the cost is per
*folder*. On a whole disk the walk is bound by SSD metadata reads when the cache is cold, and by a
system-wide filesystem lock when it's warm.

| Change | Effect |
|---|---|
| `getattrlistbulk` per folder (names, types, allocated sizes, inode numbers, link counts in one call) instead of `readdir` + an `lstat` per entry | 1.16× |
| Open each folder with `openat(parent_fd, name)` instead of by full path (falls back to a full path on `EMFILE`) | 1.38× cumulative |
| Tally files inline; only subfolders become parallel (rayon) tasks; publish progress once per folder | ~1.17× more |
| 32 KB listing buffer instead of 256 KB | ~1.06× more |
| **Total vs. the first version** | **1.53×** on a 1.5 M-item source tree, **1.79×** on `/System/Library`, about half the CPU |

**Rejected, with reasons:**

- *Trusting `ATTR_DIR_ENTRYCOUNT` to skip the final listing call.* The byte-exact gate caught it:
  APFS reports 0 entries for firmlinked and sealed-volume folders (e.g. `/Applications`), so files
  silently went missing.
- *Skipping the "no more entries" call when a batch came back short.* Up to 4% faster, but it relies
  on undocumented batching behaviour; a failure would mean silently missing files.
- *More threads, several processes, `searchfs` (catalog search), Spotlight.* No gain (the lock is
  system-wide), or slower than the parallel walk, or blind to most of the disk (Spotlight skips
  `~/Library`, hidden folders and system areas).
- *Round 2 bundle (rejected and reverted): entry names as `SharedString`, a presized flatten Vec,
  current-folder updates every 64th folder, and depth-gated skip/hotspot lookups.* The agreed keep
  rule required reproducible `/System/Library` B/A ≤ 0.99 and B faster in ≥ 7/10 pairs, with no
  reproducible median regression on `~/Library`.
  Three independent baseline-vs-bundle re-checks on `/System/Library` gave B/A **1.018 (5/10)**,
  **1.016 (3/10)** and **1.010 (3/10)**. All failed the keep rule, so steps 1–4 were reverted.
  These rounds used `COUNT_TOLERANCE=0` and the default `TOLERANCE=1000000` bytes; each reported
  files=297799, nodes=459670, bytes=28356595712. Exact counts did not require a byte-exact gate.
  The isolated step-2 comparison (steps 1+2 vs step 1) gave 0.990 (6/10) at count tolerance 0 and
  0.960 (3/10) at count tolerance 5; neither met the keep rule.
  Secondary `~/Library` re-checks (`COUNT_TOLERANCE=10`): at the default byte tolerance, two rounds
  aborted on byte drift; 0.532 (5/8) and 1.126 (3/8) came from overlapping runs and are invalid
  performance evidence; 1.010 (3/8) completed at `TOLERANCE=200000000`, which alone does not
  establish reproducible non-regression. The historical exact baseline-fingerprint check could not
  be completed because the trees drifted; matching A/B counts within pairs is not that historical check.
- *Walking one subfolder inline instead of as a rayon task.* Fan-out ≤ 1: 0.977 (6/10), 1.048 (5/10),
  `~/Library` 1.038. Fan-out ≤ 2: 0.891 (8/10), 0.963 (4/10), 1.028 (3/10), 0.934 (8/10), `~/Library` 0.963.
  Pairs won only 23/40: no consistent win.
- *Parallelising flatten.* It is at most ~1% of scan time (45–68 ms of ~5 s on `~/Library`), so it isn't worth the complexity.
- *Reading APFS clone information for every file during the scan.* 12–130% slower. Clone accounting
  moved off the scan path instead (see "Exact savings").

**Never downloads anything.** iCloud Drive's cloud-only ("dataless") folders are detected from their
flags and skipped, and the process sets `IOPOL_MATERIALIZE_DATALESS_FILES` off, so a scan can't stall
on, or trigger, a download.

## The live chart

The question here isn't "how fast is the scan" but "how soon does the picture look right".
`petal --bench-live <path>` samples the live totals every 100 ms and scores each sample against the
final result: when 50% and 90% of the bytes are on screen, when the top three folders are in their
final order, and when the top-level split stays within 5% of the end result.

What made the biggest difference:

1. **Live totals.** Each folder in the top four levels has an atomic counter that the walk adds to
   once per folder; the UI snapshots them ten times a second (~1–3 ms per snapshot).
2. **Tiers.** An *outline* pass lists the first two levels by name (done in ~10 ms), then a *hotspot*
   pass reads the folders that are usually huge (Trash, Downloads, Xcode, simulators, backups, Docker,
   caches, Movies, Mail…) before the main walk, which reuses those results rather than re-reading them.
   Hotspots are exact within ~2.5 s on a full disk.
3. **Final as you go.** A folder is marked final the moment the walk leaves it, and drawn in full
   colour; folders still counting are muted and labelled "≥ size". You can click into the chart
   mid-scan. On a full disk, half the bytes sit in final folders by ~11 s of a ~24 s scan.
   The exception is an incremental rescan: reused folders and their ancestors stay pending until
   hard-link reconciliation (`charge_links`) finishes, so no folder is shown final too early. The
   cost, measured on an incremental `/` rescan (`PETAL_BENCH_INCREMENTAL=1 --bench-live / 3`):
   final50 went from 0.93–0.94 s to 1.34–1.35 s (about +0.4 s, +43–45%); final90 is unchanged or
   better (1.34/1.98 s before, 1.35/1.34 s after); `premature-final` is 0. That misses the plan's
   ~10% threshold, and the user accepted it to keep folders from being shown final early. These are
   incremental measurements; full-scan paths are unchanged.
4. **An exact skeleton for the startup disk.** Scanning `/` reads only the Data volume; every other
   APFS volume in the container (macOS itself, Preboot, VM, Recovery, Update) gets an exact slice
   from `ATTR_VOL_SPACEUSED`, and whatever couldn't be read becomes an exact "Not readable" slice.
   The chart's total therefore equals the disk's used space to the byte.
5. **Motion.** Every segment eases towards its latest target (exponential approach, τ = 0.12 s),
   matched across snapshots by folder path, so the chart glides instead of jumping ten times a second.
   Measured on real frames at 120 Hz: the old chart's edge stood still in 93% of frames and then
   jumped up to 10.7°; now it moves every frame, at most 1.0°.

Rejected: a breadth-first work queue (every live metric got worse: it explores structure before the
leaves where the bytes are) and dropping the barrier between hotspots and the main walk (the
hotspots lost their head start).

**Gates for the live chart:**

- The live totals must equal the finished tree exactly.
- `premature-final` must be 0: no folder may be shown as final and then change size. (This caught a
  real bug: a cloud-only folder deep in the tree marked an *ancestor* final.)
- Early findings must equal their final sizes (`findings wrong 0`).
- The total scan time must not regress (A/B with `bench/ab.sh`).

## Progress and time left

The bar counts items (files + folders) against the volume's object count, which `statfs` reports
instantly and exactly. Bytes would be a poor measure: folders that need Full Disk Access hide far more
bytes than items, so a bytes bar would stall around 80%. "About N s left" comes from an item rate
smoothed over 3 s and is shown only after 1.5 s; in tests it errs on the long side by 0–3 s.

## Exact savings

Petal shows sizes as allocated blocks, like Finder's "on disk". What deleting something *frees* is
different on APFS, where files can be clones sharing blocks, or hard links. So when you collect items
(or when findings resolve), `scan::frees_of` walks just those folders in the background with clone
and link reporting on:

- a family of pure clones is freed only if every member is selected;
- a partially cloned file frees its private bytes (looked up lazily);
- a hard-linked file is freed only if every link is selected.

`cargo test -- --ignored clone_accounting_matches_apfs` checks this against reality: it builds a
throwaway APFS disk image (no admin needed), predicts what deleting each item frees, deletes it and
measures. Prediction equals the space actually freed in every case.

## Incremental rescans

After a completed scan, Petal saves the tree to `~/Library/Caches/io.github.henrydennis.petal/scans/`
(newest 3 roots) with the FSEvents event id taken *before* the scan started. On the next scan
(launch, Open Folder, Rescan) it asks the volume's FSEvents history which folders changed since then,
re-lists only those folders and their ancestors, and reuses every clean cached subtree as is. The
result is identical to a full scan's. Rescanning `~/Library` (about 215k folders) takes about 0.2 s
instead of 5–9 s.

It falls back to a full scan when there's any doubt: external volumes (always full), no FSEvents
history, no cache, a different device UUID or Full Disk Access state than when the cache was written,
dropped or wrapped events, a remount, a must-rescan above the root, a 10 s history timeout, a replaced
root. File ▸ Full Rescan (⌘⇧R) forces a full scan. Folders that
were unreadable are always re-listed, since granting access sends no event. Changing a folder's
mode, owner or ACL sends an event for its parent only, so each cached folder also keeps its change
time (ctime, to the nanosecond, read before it was listed): a clean folder whose change time differs
is walked fresh instead of reused.

`fsevents_incremental_end_to_end` is the gate. It changes a fixture, waits for the events, and
checks that the incremental path was taken and matches a full scan exactly. `--bench-rescan` does
the same on a real tree.

## Reproducing

```sh
cargo build --release
./target/release/petal --bench-scan ~/Library 6          # median of 6 headless scans
./target/release/petal --bench-live / 3                  # live-chart metrics (grant Full Disk Access for /)
./bench/ab.sh old/petal new/petal 8 ~/Library            # interleaved A/B with a result-fingerprint check
./target/release/petal --bench-rescan ~/Library 3        # incremental vs full, with an equality gate
cargo test                                               # unit tests
cargo test -- --ignored                                  # plus the APFS disk-image test
```

`bench/ab.sh` alternates the two binaries (flipping the order each round) so both see the same
background load, and refuses to report a speed-up if the two results differ.

`--bench-rescan` checks the two results for equality. On a live tree like `~/Library`, both scans see
apps writing files. So a MISMATCH there can be real drift: two full scans differ the same way. Hard
links don't cause drift: each inode is charged to its lowest-path link, deterministically (ties broken
by size, then name), so full and incremental scans are byte-exact, and incremental rescans now cover
`/`. The cache is v4, with a per-file identity, about +24% on a `/` cache (204→253 MB). On a quiet tree (`/Applications`)
it reports `gate ok`. It exits 1 if any run mismatched.
