//! Filesystem scanning and the in-memory size tree.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use gpui::SharedString;

use crate::cache;
use crate::disk::{self, DiskLayout};
use crate::dirlist;
use crate::findings;
use crate::fsevents;
use crate::onboarding;
use crate::live::{LIVE_DEPTH, LiveNode};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Dir,
    File,
    /// Not a real folder: an exact amount of space with no browsable contents, such as
    /// the macOS system volume or the part of the disk that couldn't be read.
    Other,
}

#[derive(Clone)]
pub struct Node {
    pub name: SharedString,
    pub size: u64,
    pub kind: Kind,
    pub parent: Option<usize>,
    /// Sorted by size, largest first.
    pub children: Vec<usize>,
    /// Number of files contained (1 for a file).
    pub items: u64,
}

pub struct Tree {
    pub root_path: PathBuf,
    pub nodes: Vec<Node>,
    pub errors: u64,
    /// Cloud-only folders that were skipped rather than downloaded.
    pub cloud_only: u64,
}

impl Tree {
    pub const ROOT: usize = 0;

    pub fn path_of(&self, mut ix: usize) -> PathBuf {
        let mut parts = Vec::new();
        while let Some(parent) = self.nodes[ix].parent {
            parts.push(self.nodes[ix].name.clone());
            ix = parent;
        }
        let mut path = self.root_path.clone();
        for part in parts.iter().rev() {
            path.push(part.as_ref());
        }
        path
    }

    /// The node at `path`, if the scan reached it.
    pub fn find(&self, path: &Path) -> Option<usize> {
        let relative = path.strip_prefix(&self.root_path).ok()?;
        let mut at = Self::ROOT;
        for component in relative.components() {
            let name = component.as_os_str().to_string_lossy();
            at = *self.nodes[at].children.iter().find(|&&c| self.nodes[c].name.as_ref() == name)?;
        }
        Some(at)
    }

    /// Chain of nodes from the root down to `ix`, inclusive.
    pub fn ancestry(&self, mut ix: usize) -> Vec<usize> {
        let mut chain = vec![ix];
        while let Some(parent) = self.nodes[ix].parent {
            chain.push(parent);
            ix = parent;
        }
        chain.reverse();
        chain
    }

    pub fn is_ancestor_or_self(&self, ancestor: usize, mut ix: usize) -> bool {
        loop {
            if ix == ancestor {
                return true;
            }
            match self.nodes[ix].parent {
                Some(parent) => ix = parent,
                None => return false,
            }
        }
    }

    /// Detach a node (after it was trashed) and subtract its size from its ancestors.
    pub fn remove(&mut self, ix: usize) {
        let Some(parent) = self.nodes[ix].parent else {
            return;
        };
        let (size, items) = (self.nodes[ix].size, self.nodes[ix].items);
        self.nodes[parent].children.retain(|&c| c != ix);
        self.nodes[ix].parent = None;

        let mut cur = Some(parent);
        while let Some(p) = cur {
            let node = &mut self.nodes[p];
            node.size = node.size.saturating_sub(size);
            node.items = node.items.saturating_sub(items);
            cur = node.parent;
            // Shrinking a node can change its rank among its siblings.
            if let Some(gp) = cur {
                sort_children(&mut self.nodes, gp);
            }
        }
    }
}

#[derive(Default)]
pub struct Progress {
    pub files: AtomicU64,
    pub bytes: AtomicU64,
    pub errors: AtomicU64,
    pub cloud_only: AtomicU64,
    pub cancelled: AtomicBool,
    pub current: Mutex<String>,
    /// Running directory totals for drawing the chart mid-scan.
    pub live: Arc<LiveNode>,
    /// When the outline and hotspot passes finished (ms after the scan started; 0 = not yet).
    pub outline_done_ms: AtomicU64,
    pub hotspots_done_ms: AtomicU64,
    /// Set when scanning the startup disk: its exact per-volume usage.
    pub layout: std::sync::OnceLock<Arc<DiskLayout>>,
    /// Folders listed so far.
    pub dirs: AtomicU64,
    /// Files and folders on the volume (when scanning a whole volume), for progress.
    pub expected_items: std::sync::OnceLock<u64>,
    /// Hotspot folders as they finish, with exact sizes.
    pub early_findings: Mutex<Vec<findings::Early>>,
    pub started: std::sync::OnceLock<std::time::Instant>,
    /// Folders an incremental rescan took from the cache rather than listing (included in `dirs`).
    pub reused_dirs: AtomicU64,
}

impl Progress {
    /// Forget an abandoned incremental attempt before the full scan starts over.
    fn reset_counts(&self) {
        for counter in [&self.files, &self.bytes, &self.errors, &self.cloud_only, &self.dirs, &self.reused_dirs] {
            counter.store(0, Ordering::Relaxed);
        }
        self.live.reset();
    }
}

struct Raw {
    name: String,
    size: u64,
    kind: Kind,
    items: u64,
    /// Folders only (0 for files).
    ino: u64,
    /// Folders only: `dirlist::Entry::ctime` from just before it was listed.
    stamp: u64,
    children: Vec<Raw>,
}

/// Rare per-folder facts an incremental rescan needs to know about the cached tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mark {
    /// Failed opens and listing errors this folder contributed.
    pub errors: u64,
    /// Skipped as cloud-only.
    pub dataless: bool,
    /// Its listing had a file with more than one link.
    pub hardlinks: bool,
}

/// What a full scan learns beyond the `Tree`, kept out of it so UI trees don't pay for it.
pub struct ScanMeta {
    /// Per node; 0 for files.
    pub inos: Vec<u64>,
    /// Per node, as `Raw::stamp`.
    pub stamps: Vec<u64>,
    /// Sorted by node index.
    pub marks: Vec<(usize, Mark)>,
}

struct Walker<'a> {
    progress: &'a Progress,
    allowed_devices: HashSet<u64>,
    skip: HashSet<PathBuf>,
    hardlinks: Mutex<HashSet<(u64, u64)>>,
    /// Hotspot folders already scanned; the main walk splices these in instead of
    /// reading them again.
    prescanned: Mutex<HashMap<PathBuf, Raw>>,
    /// Read-only copy of the prescanned paths, so the main walk can check membership
    /// without taking a lock for every folder.
    prescanned_paths: HashSet<PathBuf>,
    marks: Mutex<Vec<(PathBuf, Mark)>>,
    /// Any listed folder had a hard-linked file.
    saw_hardlink: AtomicBool,
}

/// Allocated size on disk, which is what actually frees up when a file is deleted.
fn disk_size(meta: &fs::Metadata) -> u64 {
    meta.blocks() * 512
}

impl<'a> Walker<'a> {
    fn new(root: &Path, progress: &'a Progress) -> (Self, Option<fs::Metadata>) {
        let root_meta = fs::symlink_metadata(root).ok();
        let mut allowed_devices: HashSet<u64> = root_meta.iter().map(|m| m.dev()).collect();
        let mut skip: HashSet<PathBuf> = ["/dev", "/Volumes", "/System/Volumes", "/net", "/home"]
            .into_iter()
            .map(PathBuf::from)
            .collect();
        skip.remove(root);

        // On macOS, `/` is a sealed system volume whose user data lives on a separate
        // APFS volume that is stitched in with firmlinks (e.g. /Users, /Applications).
        if root == Path::new("/") {
            if let Ok(data) = fs::metadata("/System/Volumes/Data") {
                allowed_devices.insert(data.dev());
            }
        }
        let walker = Walker {
            progress,
            allowed_devices,
            skip,
            hardlinks: Mutex::new(HashSet::new()),
            prescanned: Mutex::new(HashMap::new()),
            prescanned_paths: HashSet::new(),
            marks: Mutex::new(Vec::new()),
            saw_hardlink: AtomicBool::new(false),
        };
        (walker, root_meta)
    }

    fn admits(&self, path: &Path, entry: &dirlist::Entry) -> bool {
        !self.skip.contains(path) && self.allowed_devices.contains(&entry.dev)
    }

    fn admissible_dir(&self, path: &Path, entry: &dirlist::Entry) -> bool {
        entry.is_dir && !entry.dataless && self.admits(path, entry)
    }

    /// Outline pass: list the top `depth_left` levels breadth-first so every folder there
    /// is named on screen at once. Records structure only; the main walk counts the bytes.
    fn outline(&self, path: &Path, live: &LiveNode, depth_left: usize) {
        if depth_left == 0 || self.progress.cancelled.load(Ordering::Relaxed) {
            return;
        }
        crate::clock::gate();
        let Ok(listing) = dirlist::Dir::open(path).and_then(|dir| dir.list(path, false)) else { return };
        let subdirs: Vec<(PathBuf, Arc<LiveNode>)> = listing
            .entries
            .iter()
            .filter_map(|entry| {
                let child = path.join(&entry.name);
                self.admissible_dir(&child, entry).then(|| (child, live.child(&entry.name)))
            })
            .collect();
        subdirs.par_iter().for_each(|(child, node)| self.outline(child, node, depth_left - 1));
    }

    /// Hotspot pass: scan likely space hogs first so their totals are exact within seconds.
    fn prescan_hotspots(&self, root: &Path, hotspots: &[(PathBuf, usize)]) {
        let results: Vec<(PathBuf, Raw)> = hotspots
            .par_iter()
            .filter_map(|(path, category)| {
                let relative = path.strip_prefix(root).ok()?;
                // Attach to the live tree at the same node the main walk will use.
                let mut live = self.progress.live.clone();
                let mut depth = 0;
                for component in relative.components() {
                    depth += 1;
                    if depth <= LIVE_DEPTH {
                        live = live.child(&component.as_os_str().to_string_lossy());
                    }
                }
                // Read the folder's own allocation now (the same attribute its parent's listing
                // reports), so its total is exact, and final, as soon as this pass ends.
                let own = dirlist::dir_alloc(path).unwrap_or(0);
                // Stamp it before listing, not when its parent's listing comes later: a change
                // in between must leave a stale stamp, so the next rescan lists it again.
                let stamp = fs::symlink_metadata(path).map(|m| dirlist::stamp(m.ctime(), m.ctime_nsec())).unwrap_or(0);
                // `walk_subdir` fills in the inode when it splices this in.
                let mut raw = self.walk_dir(None, path, display_name(path), 0, own, &live, depth);
                raw.stamp = stamp;
                if depth <= LIVE_DEPTH {
                    live.done.store(true, Ordering::Release);
                }
                // Publish the exact size straight away, as a finding.
                self.progress.early_findings.lock().unwrap().push(findings::Early {
                    category: *category,
                    path: path.clone(),
                    size: raw.size,
                    at_ms: self.progress.started.get().map(|s| s.elapsed().as_millis() as u64).unwrap_or(0),
                });
                Some((path.clone(), raw))
            })
            .collect();
        self.prescanned.lock().unwrap().extend(results);
    }

    /// One folder's listing, shared by the full walk and `rescan`: opened relative to
    /// `parent` (full-path fallback), its files tallied in listing order, its subfolders
    /// left to the caller.
    fn read(
        &self,
        parent: Option<&dirlist::Dir>,
        path: &Path,
        name: &str,
        own: u64,
        live: &LiveNode,
    ) -> (Option<dirlist::Dir>, Vec<Raw>, Vec<dirlist::Entry>) {
        crate::clock::gate();
        if let Ok(mut current) = self.progress.current.try_lock() {
            *current = path.to_string_lossy().into_owned();
        }
        self.progress.dirs.fetch_add(1, Ordering::Relaxed);

        let mut mark = Mark::default();
        // Relative open is the fast path; fall back to the full path (e.g. on EMFILE).
        let dir = match parent {
            Some(parent) => parent.open_at(name).or_else(|_| dirlist::Dir::open(path)),
            None => dirlist::Dir::open(path),
        };
        let listing = dir.and_then(|dir| dir.list(path, false).map(|listing| (dir, listing)));
        let (dir, entries) = match listing {
            Ok((dir, listing)) => {
                if listing.errors > 0 {
                    self.progress.errors.fetch_add(listing.errors, Ordering::Relaxed);
                    mark.errors = listing.errors;
                }
                (Some(dir), listing.entries)
            }
            Err(_) => {
                self.progress.errors.fetch_add(1, Ordering::Relaxed);
                mark.errors = 1;
                (None, Vec::new())
            }
        };

        // Files are cheap: tally them inline and publish progress once per directory.
        // Only subdirectories become parallel tasks.
        let mut files: Vec<Raw> = Vec::with_capacity(entries.len());
        let mut subdirs = Vec::new();
        let (mut count, mut bytes) = (0u64, 0u64);
        for entry in entries {
            if entry.is_dir {
                subdirs.push(entry);
                continue;
            }
            let mut size = entry.size;
            if entry.nlink > 1 {
                mark.hardlinks = true;
                if !self.hardlinks.lock().unwrap().insert((entry.dev, entry.ino)) {
                    size = 0;
                }
            }
            count += 1;
            bytes += size;
            files.push(Raw { name: entry.name, size, kind: Kind::File, items: 1, ino: 0, stamp: 0, children: Vec::new() });
        }
        self.progress.files.fetch_add(count, Ordering::Relaxed);
        self.progress.bytes.fetch_add(bytes, Ordering::Relaxed);
        live.record(own + bytes, count);
        if mark != Mark::default() {
            if mark.hardlinks {
                self.saw_hardlink.store(true, Ordering::Relaxed);
            }
            self.marks.lock().unwrap().push((path.to_path_buf(), mark));
        }
        (dir, files, subdirs)
    }

    /// `live` is this directory's own live node if it is within `LIVE_DEPTH`, else its
    /// nearest live ancestor's.
    #[allow(clippy::too_many_arguments)]
    fn walk_dir(
        &self,
        parent: Option<&dirlist::Dir>,
        path: &Path,
        name: String,
        ino: u64,
        own: u64,
        live: &LiveNode,
        depth: usize,
    ) -> Raw {
        if self.progress.cancelled.load(Ordering::Relaxed) {
            return Raw { name, size: 0, kind: Kind::Dir, items: 0, ino, stamp: 0, children: Vec::new() };
        }
        let (dir, mut children, subdirs) = self.read(parent, path, &name, own, live);
        let subdir_results: Vec<Raw> = subdirs
            .into_par_iter()
            .filter_map(|entry| self.walk_subdir(dir.as_ref(), path, entry, live, depth + 1))
            .collect();
        drop(dir);
        children.extend(subdir_results);
        children.sort_by(|a, b| b.size.cmp(&a.size));

        let size = own + children.iter().map(|c| c.size).sum::<u64>();
        let items = children.iter().map(|c| c.items).sum();
        Raw { name, size, kind: Kind::Dir, items, ino, stamp: 0, children }
    }

    fn walk_subdir(
        &self,
        dir: Option<&dirlist::Dir>,
        parent: &Path,
        entry: dirlist::Entry,
        parent_live: &LiveNode,
        depth: usize,
    ) -> Option<Raw> {
        let path = parent.join(&entry.name);
        if !self.admits(&path, &entry) {
            return None;
        }
        let own_live = (depth <= LIVE_DEPTH).then(|| parent_live.child(&entry.name));
        let live = own_live.as_deref().unwrap_or(parent_live);
        if self.prescanned_paths.contains(&path) {
            if let Some(mut raw) = self.prescanned.lock().unwrap().remove(&path) {
                raw.ino = entry.ino;
                return Some(raw);
            }
        }
        if entry.dataless {
            // Its contents live in the cloud and take no space here; don't make macOS fetch them.
            self.progress.cloud_only.fetch_add(1, Ordering::Relaxed);
            live.record(entry.size, 0);
            self.marks.lock().unwrap().push((path, Mark { dataless: true, ..Mark::default() }));
            // Only the folder's own node is final; `live` may be an ancestor's node here.
            if let Some(own) = &own_live {
                own.done.store(true, Ordering::Release);
            }
            return Some(Raw { name: entry.name, size: entry.size, kind: Kind::Dir, items: 0, ino: entry.ino, stamp: entry.ctime, children: Vec::new() });
        }
        let mut raw = self.walk_dir(dir, &path, entry.name, entry.ino, entry.size, live, depth);
        raw.stamp = entry.ctime;
        if let Some(own) = &own_live {
            if !self.progress.cancelled.load(Ordering::Relaxed) {
                own.done.store(true, Ordering::Release);
            }
        }
        Some(raw)
    }
}

/// A plain full scan, no cache I/O.
pub fn scan(root: &Path, progress: &Progress) -> Tree {
    scan_root(root, progress, |root, bases| scan_with_bases(root, progress, bases))
}

/// How `scan_cached` got its tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum How {
    Incremental { relisted: u64, reused: u64 },
    Full(&'static str),
}

/// A finished scan's cache, saved once the final tree is known.
pub struct SaveCache {
    meta: ScanMeta,
    header: cache::Header,
    pub how: How,
}

impl SaveCache {
    pub fn save(&self, tree: &Tree) {
        cache::save(tree, &self.meta, &self.header);
    }
}

/// `scan`, but brought up to date from the cache when FSEvents can say what changed.
/// Only on the startup disk's container: external volumes always scan in full.
pub fn scan_cached(root: &Path, progress: &Progress, force_full: bool) -> (Tree, Option<SaveCache>) {
    let mut save = None;
    let tree = scan_root(root, progress, |root, bases| {
        let (tree, s) = scan_cached_inner(root, progress, bases, force_full);
        save = s;
        tree
    });
    (tree, save)
}

fn scan_cached_inner(root: &Path, progress: &Progress, bases: &findings::Bases, force_full: bool) -> (Tree, Option<SaveCache>) {
    let log = |how: &How| {
        if std::env::var_os("PETAL_PHASES").is_some() {
            match how {
                How::Incremental { relisted, reused } => eprintln!("  incremental: {relisted} dirs re-listed, {reused} reused"),
                How::Full(reason) => eprintln!("  full: {reason}"),
            }
        }
    };
    let internal = disk::container_at(root).is_some_and(|c| Some(c) == disk::container_at(Path::new("/")));
    let meta = fs::symlink_metadata(root).ok();
    let target = meta.filter(|_| internal).and_then(|m| Some((m.dev(), m.ino(), fsevents::device_uuid(m.dev())?)));
    let Some((dev, root_ino, device_uuid)) = target else {
        log(&How::Full(if internal { "no FSEvents history" } else { "external volume" }));
        return (scan_with_bases(root, progress, bases), None);
    };
    // Before any listing: changes made during this scan get later ids, so the next one replays them.
    let event_id = fsevents::current_id();
    let has_fda = onboarding::has_full_disk_access();
    let full = |reason| {
        let (tree, meta) = scan_with_meta(root, progress, bases);
        (tree, meta, How::Full(reason))
    };
    let (tree, meta, how) = 'scan: {
        if force_full {
            break 'scan full("forced");
        }
        let Some(cached) = cache::load(root, &device_uuid, has_fda) else { break 'scan full("no valid cache") };
        let (Ok(canonical), Some(mount)) = (fs::canonicalize(root), mount_of(root)) else { break 'scan full("no mount point") };
        let dirty = match fsevents::changes_since(dev, &mount, &canonical, cached.header.event_id) {
            fsevents::Changes::Reset(reason) => break 'scan full(reason),
            fsevents::Changes::Dirs(dirs) => dirs.into_iter().map(|(rel, recursive)| (root.join(rel), recursive)).collect::<Vec<_>>(),
        };
        match incremental::rescan(root, progress, cached, &dirty) {
            Ok((tree, meta)) => {
                let reused = progress.reused_dirs.load(Ordering::Relaxed);
                let relisted = progress.dirs.load(Ordering::Relaxed) - reused;
                (tree, meta, How::Incremental { relisted, reused })
            }
            Err(fallback) => {
                progress.reset_counts();
                full(match fallback {
                    incremental::Fallback::Root => "root replaced",
                    incremental::Fallback::HardLinks => "hard links on a changed path",
                })
            }
        }
    };
    log(&how);
    let save = (!progress.cancelled.load(Ordering::Relaxed)).then(|| SaveCache {
        meta,
        header: cache::Header { root: root.to_path_buf(), device_uuid, event_id, root_ino, has_fda },
        how,
    });
    (tree, save)
}

/// The startup-disk handling shared by `scan` and `scan_cached`; `inner` scans one folder.
fn scan_root(root: &Path, progress: &Progress, inner: impl FnOnce(&Path, &findings::Bases) -> Tree) -> Tree {
    let _ = progress.started.set(std::time::Instant::now());
    if root == Path::new("/") {
        if let Some(layout) = disk::startup_layout(startup_disk_name()) {
            // Scan only the Data volume: the sealed macOS volume, Preboot, swap and the
            // rest come from APFS's exact per-volume usage instead.
            let layout = Arc::new(layout);
            let _ = progress.layout.set(layout.clone());
            if let Some(items) = volume_items(&layout.data_root) {
                let _ = progress.expected_items.set(items);
            }
            let bases = findings::Bases::for_root(&layout.data_root);
            let mut tree = inner(&layout.data_root, &bases);
            add_volume_slices(&mut tree, &layout);
            return tree;
        }
    }
    if volume_used(root).is_some() {
        if let Some(items) = volume_items(root) {
            let _ = progress.expected_items.set(items);
        }
    }
    inner(root, &findings::Bases::for_root(root))
}

/// Files and folders in use on the volume mounted at `mount` (exact, from APFS).
pub fn volume_items(mount: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(mount.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statfs(c_path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some(stat.f_files.saturating_sub(stat.f_ffree))
}

/// Turn a Data-volume tree into a whole-disk tree: the other volumes become exact slices,
/// and whatever the Data volume holds beyond what could be read becomes "Not readable",
/// so the total matches what Finder reports as used.
fn add_volume_slices(tree: &mut Tree, layout: &DiskLayout) {
    let scanned = tree.nodes[Tree::ROOT].size;
    let unreadable = layout.data_used.saturating_sub(scanned);
    let slices = layout
        .extras
        .iter()
        .cloned()
        .chain((unreadable > 0).then(|| (disk::NOT_READABLE.to_string(), unreadable)));
    let mut added = 0;
    for (name, size) in slices {
        let ix = tree.nodes.len();
        tree.nodes.push(Node { name: name.into(), size, kind: Kind::Other, parent: Some(Tree::ROOT), children: Vec::new(), items: 0 });
        tree.nodes[Tree::ROOT].children.push(ix);
        added += size;
    }
    let root = &mut tree.nodes[Tree::ROOT];
    root.name = layout.name.clone().into();
    root.size += added;
    sort_children(&mut tree.nodes, Tree::ROOT);
}

/// Largest first, except the "not scanned / not readable" remainder, which always
/// comes last so it reads as the tail of the ring.
pub fn sort_children(nodes: &mut [Node], ix: usize) {
    let mut children = std::mem::take(&mut nodes[ix].children);
    children.sort_by_key(|&c| {
        let node = &nodes[c];
        let remainder = node.kind == Kind::Other && (node.name.as_ref() == disk::NOT_SCANNED || node.name.as_ref() == disk::NOT_READABLE);
        (remainder, std::cmp::Reverse(node.size))
    });
    nodes[ix].children = children;
}

/// The startup disk's name as Finder shows it.
pub fn startup_disk_name() -> String {
    fs::read_dir("/Volumes")
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .find(|entry| fs::canonicalize(entry.path()).ok().as_deref() == Some(Path::new("/")))
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .unwrap_or_else(|| "Startup Disk".to_string())
}

/// `bases` locate the hotspot folders (none: no hotspot pass).
fn scan_with_bases(root: &Path, progress: &Progress, bases: &findings::Bases) -> Tree {
    scan_with_meta(root, progress, bases).0
}

fn scan_with_meta(root: &Path, progress: &Progress, bases: &findings::Bases) -> (Tree, ScanMeta) {
    let (mut walker, root_meta) = Walker::new(root, progress);
    let own = root_meta.as_ref().map(disk_size).unwrap_or(0);
    let ino = root_meta.as_ref().map(|m| m.ino()).unwrap_or(0);
    dirlist::raise_fd_limit();
    dirlist::disable_cloud_downloads();
    let walk_start = std::time::Instant::now();
    let elapsed_ms = || walk_start.elapsed().as_millis() as u64;
    walker.outline(root, &progress.live, OUTLINE_DEPTH);
    progress.outline_done_ms.store(elapsed_ms().max(1), Ordering::Relaxed);
    let hotspots = hotspots(root, bases, &walker);
    walker.prescan_hotspots(root, &hotspots);
    walker.prescanned_paths = hotspots.into_iter().map(|(path, _)| path).collect();
    progress.hotspots_done_ms.store(elapsed_ms().max(1), Ordering::Relaxed);
    let mut raw = walker.walk_dir(None, root, display_name(root), ino, own, &progress.live, 0);
    raw.stamp = root_meta.as_ref().map(meta_stamp).unwrap_or(0);
    debug_assert!(walker.prescanned.lock().unwrap().is_empty(), "a hotspot was never spliced in");
    let flatten_start = std::time::Instant::now();

    let mut nodes = Vec::new();
    let (mut inos, mut stamps) = (Vec::new(), Vec::new());
    flatten(raw, None, &mut nodes, &mut inos, &mut stamps);
    if std::env::var_os("PETAL_PHASES").is_some() {
        eprintln!("  walk {:.3}s  flatten {:.3}s", (flatten_start - walk_start).as_secs_f64(), flatten_start.elapsed().as_secs_f64());
    }
    let tree = Tree {
        root_path: root.to_path_buf(),
        nodes,
        errors: progress.errors.load(Ordering::Relaxed),
        cloud_only: progress.cloud_only.load(Ordering::Relaxed),
    };
    let marks = resolve_marks(&tree, walker.marks.into_inner().unwrap());
    (tree, ScanMeta { inos, stamps, marks })
}

fn meta_stamp(meta: &fs::Metadata) -> u64 {
    dirlist::stamp(meta.ctime(), meta.ctime_nsec())
}

/// Path-keyed marks to node indices, sorted. Marks are rare, so `find` is fine.
fn resolve_marks(tree: &Tree, marks: Vec<(PathBuf, Mark)>) -> Vec<(usize, Mark)> {
    let mut out: Vec<(usize, Mark)> = marks.into_iter().filter_map(|(path, mark)| Some((tree.find(&path)?, mark))).collect();
    out.sort_by_key(|m| m.0);
    out
}

/// Incremental rescans from a cached tree.
pub mod incremental {
    use super::*;
    use crate::cache::Cached;

    /// Why `rescan` couldn't run; do a full scan instead.
    #[derive(Debug, PartialEq, Eq)]
    pub enum Fallback {
        /// The root is gone, replaced, or isn't the cached one.
        Root,
        /// Hard links on a changed path: cached per-folder link sizes can't be re-deduplicated.
        // ponytail: v1 policy; exact re-dedup during reuse if this falls back too often.
        HardLinks,
    }

    /// A folder in `rescan`'s result, before it is flattened.
    enum Part {
        Fresh(Raw),
        /// A cached subtree reused as is.
        Reused(usize),
        Dir { name: String, size: u64, items: u64, ino: u64, stamp: u64, children: Vec<Part> },
    }

    /// The cached tree and which of its folders `rescan` must re-list.
    struct Prior {
        cached: Cached,
        /// Dirty, or an ancestor of a dirty folder.
        on_path: Vec<bool>,
        /// Walk fresh, reusing nothing.
        rewalk: Vec<bool>,
    }

    impl Prior {
        fn size(&self, part: &Part) -> (u64, u64) {
            match part {
                Part::Fresh(raw) => (raw.size, raw.items),
                Part::Reused(ix) => (self.cached.nodes[*ix].size, self.cached.nodes[*ix].items),
                Part::Dir { size, items, .. } => (*size, *items),
            }
        }
    }

    /// Bring a cached scan of `root` up to date, re-listing only folders on the path to a
    /// `dirty` one (`true`: everything below it changed too). The result is identical to a
    /// full scan's, as long as `dirty` covers every changed folder.
    pub fn rescan(root: &Path, progress: &Progress, cached: Cached, dirty: &[(PathBuf, bool)]) -> Result<(Tree, ScanMeta), Fallback> {
        let _ = progress.started.set(std::time::Instant::now());
        let (walker, root_meta) = Walker::new(root, progress);
        let root_meta = root_meta
            .filter(|m| m.is_dir() && m.ino() == cached.header.root_ino && cached.header.root == root)
            .ok_or(Fallback::Root)?;

        let nodes = &cached.nodes;
        let (mut on_path, mut rewalk) = (vec![false; nodes.len()], vec![false; nodes.len()]);
        // Unreadable folders may have become readable (e.g. Full Disk Access) with no event.
        let mut dirty_ix: Vec<usize> = cached.marks.iter().filter(|(_, m)| m.errors > 0).map(|m| m.0).collect();
        for (path, recursive) in dirty {
            let Ok(relative) = path.strip_prefix(root) else { continue };
            if path.ancestors().take_while(|a| *a != root).any(|a| walker.skip.contains(a)) {
                continue;
            }
            // The deepest cached folder on the path; a missing one leaves its parent dirty.
            let (mut at, mut exact) = (Tree::ROOT, true);
            for component in relative.components() {
                let name = component.as_os_str().to_string_lossy();
                match nodes[at].children.iter().find(|&&c| nodes[c].kind == Kind::Dir && nodes[c].name.as_ref() == name) {
                    Some(&c) => at = c,
                    None => {
                        exact = false;
                        break;
                    }
                }
            }
            rewalk[at] |= exact && *recursive;
            dirty_ix.push(at);
        }
        on_path[Tree::ROOT] = true;
        for ix in dirty_ix {
            let mut cur = Some(ix);
            while let Some(i) = cur.filter(|&i| !on_path[i]) {
                on_path[i] = true;
                cur = nodes[i].parent;
            }
        }
        if cached.marks.iter().any(|(ix, m)| m.hardlinks && on_path[*ix]) {
            return Err(Fallback::HardLinks);
        }

        dirlist::raise_fd_limit();
        dirlist::disable_cloud_downloads();
        let (own, ino, stamp, name) = (disk_size(&root_meta), root_meta.ino(), meta_stamp(&root_meta), display_name(root));
        let prior = Prior { cached, on_path, rewalk };
        let part = if prior.rewalk[Tree::ROOT] {
            let mut raw = walker.walk_dir(None, root, name, ino, own, &progress.live, 0);
            raw.stamp = stamp;
            Part::Fresh(raw)
        } else {
            walker.merge(&prior, Tree::ROOT, None, root, name, ino, stamp, own, &progress.live, 0)
        };
        if walker.saw_hardlink.load(Ordering::Relaxed) {
            return Err(Fallback::HardLinks);
        }

        let mut cached = prior.cached;
        let (mut nodes, mut meta) = (Vec::new(), ScanMeta { inos: Vec::new(), stamps: Vec::new(), marks: Vec::new() });
        emit(part, None, &mut cached, &mut nodes, &mut meta);
        drop(cached);
        let tree = Tree {
            root_path: root.to_path_buf(),
            nodes,
            errors: progress.errors.load(Ordering::Relaxed),
            cloud_only: progress.cloud_only.load(Ordering::Relaxed),
        };
        meta.marks.extend(resolve_marks(&tree, walker.marks.into_inner().unwrap()));
        meta.marks.sort_by_key(|m| m.0);
        Ok((tree, meta))
    }

    impl Walker<'_> {
        /// `rescan`'s walk of an on-path folder: re-list it, reuse clean cached subfolders,
        /// walk new or replaced ones fresh. Children are assembled exactly like `walk_dir`'s.
        #[allow(clippy::too_many_arguments)]
        fn merge(
            &self,
            prior: &Prior,
            ix: usize,
            parent: Option<&dirlist::Dir>,
            path: &Path,
            name: String,
            ino: u64,
            stamp: u64,
            own: u64,
            live: &LiveNode,
            depth: usize,
        ) -> Part {
            if self.progress.cancelled.load(Ordering::Relaxed) {
                return Part::Fresh(self.walk_dir(parent, path, name, ino, own, live, depth));
            }
            let (dir, files, subdirs) = self.read(parent, path, &name, own, live);
            let cached = &prior.cached;
            let kids: HashMap<&str, usize> = cached.nodes[ix]
                .children
                .iter()
                .filter(|&&c| cached.nodes[c].kind == Kind::Dir)
                .map(|&c| (cached.nodes[c].name.as_ref(), c))
                .collect();
            let subdir_parts: Vec<Part> = subdirs
                .into_par_iter()
                .filter_map(|entry| {
                    let reusable = kids.get(entry.name.as_str()).copied().filter(|&c| {
                        !prior.rewalk[c]
                            && cached.inos[c] == entry.ino
                            && cached.own(c) == entry.size
                            && !entry.dataless
                            && !cached.mark(c).dataless
                            // Off the dirty paths it isn't opened, so a change to its own mode,
                            // owner or ACL (an event for its parent only) must show in its stamp.
                            && (prior.on_path[c] || (entry.ctime != 0 && cached.stamps[c] == entry.ctime))
                    });
                    let Some(c) = reusable else {
                        return self.walk_subdir(dir.as_ref(), path, entry, live, depth + 1).map(Part::Fresh);
                    };
                    let child_path = path.join(&entry.name);
                    if !self.admits(&child_path, &entry) {
                        return None;
                    }
                    let own_live = (depth < LIVE_DEPTH).then(|| live.child(&entry.name));
                    let child_live = own_live.as_deref().unwrap_or(live);
                    let part = if prior.on_path[c] {
                        self.merge(prior, c, dir.as_ref(), &child_path, entry.name, entry.ino, entry.ctime, entry.size, child_live, depth + 1)
                    } else {
                        self.reuse(cached, c, child_live, depth + 1);
                        Part::Reused(c)
                    };
                    if let Some(own) = &own_live {
                        if !self.progress.cancelled.load(Ordering::Relaxed) {
                            own.done.store(true, Ordering::Release);
                        }
                    }
                    Some(part)
                })
                .collect();
            drop(dir);
            let mut children: Vec<Part> = files.into_iter().map(Part::Fresh).collect();
            children.extend(subdir_parts);
            children.sort_by(|a, b| prior.size(b).0.cmp(&prior.size(a).0));

            let (mut size, mut items) = (own, 0);
            for child in &children {
                let (s, i) = prior.size(child);
                size += s;
                items += i;
            }
            Part::Dir { name, size, items, ino, stamp, children }
        }

        /// Count a reused cached subtree as if it had just been walked.
        fn reuse(&self, cached: &Cached, ix: usize, live: &LiveNode, depth: usize) {
            let end = cached.sub_end[ix];
            let (mut files, mut bytes, mut dirs) = (0, 0, 0);
            for node in &cached.nodes[ix..end] {
                if node.kind == Kind::File {
                    files += 1;
                    bytes += node.size;
                } else {
                    dirs += 1;
                }
            }
            let cloud_only = cached.marks_in(ix, end).iter().filter(|m| m.1.dataless).count() as u64;
            self.progress.files.fetch_add(files, Ordering::Relaxed);
            self.progress.bytes.fetch_add(bytes, Ordering::Relaxed);
            // Folders "listed", as the progress bar counts them.
            self.progress.dirs.fetch_add(dirs - cloud_only, Ordering::Relaxed);
            self.progress.cloud_only.fetch_add(cloud_only, Ordering::Relaxed);
            self.progress.reused_dirs.fetch_add(dirs - cloud_only, Ordering::Relaxed);
            seed_live(cached, ix, live, depth);
        }
    }

    /// Record a reused folder in the live tree the way the full walk would have, final at once.
    /// `live` is its own node within `LIVE_DEPTH`, else its nearest live ancestor's.
    fn seed_live(cached: &Cached, ix: usize, live: &LiveNode, depth: usize) {
        let node = &cached.nodes[ix];
        if depth >= LIVE_DEPTH {
            live.record(node.size, node.items);
        } else {
            let (mut size, mut items) = (node.size, node.items);
            for &c in &node.children {
                let child = &cached.nodes[c];
                if child.kind == Kind::Dir {
                    size -= child.size;
                    items -= child.items;
                    seed_live(cached, c, &live.child(&child.name), depth + 1);
                }
            }
            live.record(size, items);
        }
        if depth <= LIVE_DEPTH {
            live.done.store(true, Ordering::Release);
        }
    }

    /// Flatten a `Part` in pre-order. Reused ranges are moved out of `cached`, not cloned.
    fn emit(part: Part, parent: Option<usize>, cached: &mut Cached, nodes: &mut Vec<Node>, meta: &mut ScanMeta) -> usize {
        match part {
            Part::Fresh(raw) => flatten(raw, parent, nodes, &mut meta.inos, &mut meta.stamps),
            Part::Reused(ix) => {
                let (base, end) = (nodes.len(), cached.sub_end[ix]);
                let remap = |i: usize| i - ix + base;
                for k in ix..end {
                    let node = &mut cached.nodes[k];
                    let mut children = std::mem::take(&mut node.children);
                    children.iter_mut().for_each(|c| *c = remap(*c));
                    nodes.push(Node {
                        name: std::mem::replace(&mut node.name, SharedString::new_static("")),
                        size: node.size,
                        kind: node.kind,
                        parent: if k == ix { parent } else { node.parent.map(remap) },
                        children,
                        items: node.items,
                    });
                }
                meta.inos.extend_from_slice(&cached.inos[ix..end]);
                meta.stamps.extend_from_slice(&cached.stamps[ix..end]);
                meta.marks.extend(cached.marks_in(ix, end).iter().map(|&(k, m)| (remap(k), m)));
                base
            }
            Part::Dir { name, size, items, ino, stamp, children } => {
                let ix = nodes.len();
                nodes.push(Node { name: name.into(), size, kind: Kind::Dir, parent, children: Vec::with_capacity(children.len()), items });
                meta.inos.push(ino);
                meta.stamps.push(stamp);
                for child in children {
                    let child_ix = emit(child, Some(ix), cached, nodes, meta);
                    nodes[ix].children.push(child_ix);
                }
                ix
            }
        }
    }

    /// Every node as (path, kind, size, items), in pre-order: equal fingerprints mean equal
    /// trees, children order included.
    pub fn fingerprint(tree: &Tree) -> Vec<(String, Kind, u64, u64)> {
        let mut out = Vec::with_capacity(tree.nodes.len());
        let mut stack = vec![(Tree::ROOT, String::new())];
        while let Some((ix, path)) = stack.pop() {
            let node = &tree.nodes[ix];
            for &c in node.children.iter().rev() {
                stack.push((c, format!("{path}/{}", tree.nodes[c].name)));
            }
            out.push((path, node.kind, node.size, node.items));
        }
        out
    }
}

/// What deleting these paths would actually free, walking them with APFS clone reporting
/// on (in parallel). Cloned data counts only if every file sharing it is among the paths;
/// a partially shared file frees just its private bytes. Runs off the UI thread.
pub fn frees_of(paths: &[PathBuf]) -> u64 {
    #[derive(Default)]
    struct Tally {
        freed: AtomicU64,
        // clone id -> (members found, members in total, bytes)
        families: Mutex<HashMap<u64, (u32, u32, u64)>>,
        // (device, inode) -> (links found, links in total, bytes): a hard-linked file is
        // only freed once every link to it is deleted.
        links: Mutex<HashMap<(u64, u64), (u64, u64, u64)>>,
    }
    fn file(tally: &Tally, path: &Path, size: u64, sharing: Option<dirlist::Sharing>, link: Option<(u64, u64, u64)>) {
        if let Some((dev, ino, nlink)) = link.filter(|l| l.2 > 1) {
            let mut links = tally.links.lock().unwrap();
            let entry = links.entry((dev, ino)).or_insert((0, nlink, size));
            entry.0 += 1;
            return;
        }
        match sharing {
            Some(info) if info.all && info.refs > 1 => {
                let mut families = tally.families.lock().unwrap();
                let family = families.entry(info.clone_id).or_insert((0, info.refs, 0));
                family.0 += 1;
                family.2 = family.2.max(size);
            }
            Some(_) => {
                tally.freed.fetch_add(dirlist::private_size(path).unwrap_or(size).min(size), Ordering::Relaxed);
            }
            None => {
                tally.freed.fetch_add(size, Ordering::Relaxed);
            }
        }
    }
    fn dir(tally: &Tally, path: &Path, dev: u64) {
        let Ok(listing) = dirlist::Dir::open(path).and_then(|d| d.list(path, true)) else { return };
        let mut subdirs = Vec::new();
        for entry in listing.entries {
            if entry.is_dir {
                tally.freed.fetch_add(entry.size, Ordering::Relaxed);
                if !entry.dataless && entry.dev == dev {
                    subdirs.push(path.join(&entry.name));
                }
            } else {
                let link = Some((entry.dev, entry.ino, entry.nlink));
                file(tally, &path.join(&entry.name), entry.size, entry.sharing, link);
            }
        }
        subdirs.par_iter().for_each(|sub| dir(tally, sub, dev));
    }
    let tally = Tally::default();
    paths.par_iter().for_each(|path| {
        let Ok(meta) = fs::symlink_metadata(path) else { return };
        if meta.is_dir() {
            tally.freed.fetch_add(dirlist::dir_alloc(path).unwrap_or(0), Ordering::Relaxed);
            dir(&tally, path, meta.dev());
        } else {
            file(&tally, path, disk_size(&meta), dirlist::file_sharing(path), Some((meta.dev(), meta.ino(), meta.nlink() as u64)));
        }
    });
    let families = tally.families.into_inner().unwrap();
    let links = tally.links.into_inner().unwrap();
    tally.freed.into_inner()
        + families.values().filter(|(found, refs, _)| found >= refs).map(|f| f.2).sum::<u64>()
        + links.values().filter(|(found, total, _)| found >= total).map(|l| l.2).sum::<u64>()
}

/// Levels listed breadth-first before the main walk, so the chart can name them at once.
const OUTLINE_DEPTH: usize = 2;

/// Hotspots that exist under `root`, on a scanned volume, and don't nest inside each other.
/// Each comes with its index in the findings catalog.
fn hotspots(root: &Path, bases: &findings::Bases, walker: &Walker) -> Vec<(PathBuf, usize)> {
    let mut found: Vec<(PathBuf, usize)> = Vec::new();
    for (category, entry) in findings::CATALOG.iter().enumerate() {
        let Some(path) = bases.locate(entry) else { continue };
        if path == root || !path.starts_with(root) || found.iter().any(|(f, _)| path.starts_with(f) || f.starts_with(&path)) {
            continue;
        }
        let Ok(meta) = fs::symlink_metadata(&path) else { continue };
        let dataless = {
            use std::os::macos::fs::MetadataExt as _;
            meta.st_flags() & 0x4000_0000 != 0
        };
        if !meta.is_dir() || dataless || !walker.allowed_devices.contains(&meta.dev()) {
            continue;
        }
        // Only folders the walk itself would skip matter, i.e. those below the scan root.
        if path.ancestors().take_while(|a| *a != root).any(|a| walker.skip.contains(a)) {
            continue;
        }
        found.push((path, category));
    }
    found
}

/// Time `scan` over several runs and print a stable fingerprint of the result.
pub fn bench(root: &Path, runs: usize) {
    let mut times = Vec::new();
    for run in 0..runs {
        let progress = Progress::default();
        let start = std::time::Instant::now();
        let tree = scan(root, &progress);
        let elapsed = start.elapsed().as_secs_f64();
        times.push(elapsed);
        if std::env::var_os("PETAL_PHASES").is_some() {
            let t = std::time::Instant::now();
            let mut found = findings::from_tree(&tree, &findings::Bases::for_root(&tree.root_path));
            let instant = t.elapsed().as_secs_f64();
            for finding in found.iter_mut() {
                let paths: Vec<PathBuf> = finding.nodes.iter().map(|&ix| tree.path_of(ix)).collect();
                let t = std::time::Instant::now();
                let frees = frees_of(&paths);
                eprintln!("  {}: allocated {}, frees {} (worked out in {:.2}s)", finding.title, format_size(finding.size), format_size(frees), t.elapsed().as_secs_f64());
                finding.size = frees;
            }
            eprintln!(
                "  findings {:.2}s: {}",
                instant,
                found.iter().map(|f| format!("{} {}", f.title, format_size(f.size))).collect::<Vec<_>>().join(", ")
            );
        }
        let root_node = &tree.nodes[Tree::ROOT];
        if let Some(layout) = progress.layout.get() {
            let other: u64 = root_node.children.iter().filter(|&&c| tree.nodes[c].kind == Kind::Other).map(|&c| tree.nodes[c].size).sum();
            eprintln!(
                "  disk used {}  data scanned {}  other volumes + not readable {}  chart total {} ({})",
                layout.container_used,
                root_node.size - other,
                other,
                root_node.size,
                if root_node.size == layout.container_used { "matches disk" } else { "MISMATCH" }
            );
        }
        eprintln!(
            "run {run}: {elapsed:.3}s  files={} bytes={} nodes={} errors={} cloud_only={}",
            root_node.items,
            root_node.size,
            tree.nodes.len(),
            tree.errors,
            tree.cloud_only
        );
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("median {:.3}s  min {:.3}s  max {:.3}s", times[times.len() / 2], times[0], times[times.len() - 1]);
}

/// Time incremental rescans against full scans of `root`, checking they agree.
/// False if any run mismatched.
pub fn bench_rescan(root: &Path, runs: usize) -> bool {
    println!("(on a live tree, an occasional MISMATCH can be real changes between the two scans)");
    let (tree, save) = scan_cached(root, &Progress::default(), true);
    let Some(save) = save else {
        println!("no FSEvents history for this volume (or it's external): every scan is full");
        return true;
    };
    save.save(&tree);
    drop(tree);
    let mut all_ok = true;
    for run in 0..runs {
        let start = std::time::Instant::now();
        let (incr, save) = scan_cached(root, &Progress::default(), false);
        let incr_s = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let full = scan(root, &Progress::default());
        let full_s = start.elapsed().as_secs_f64();
        let how = match save.as_ref().map(|s| s.how) {
            Some(How::Incremental { relisted, reused }) => format!("{relisted} relisted, {reused} reused"),
            Some(How::Full(reason)) => format!("fell back to full: {reason}"),
            None => "not saved".into(),
        };
        if let Some(save) = save {
            save.save(&incr);
        }
        let (a, b) = (incremental::fingerprint(&incr), incremental::fingerprint(&full));
        let (sa, sb): (HashSet<_>, HashSet<_>) = (a.iter().collect(), b.iter().collect());
        let differ = sa.symmetric_difference(&sb).count();
        // Paths and item counts only: tells real drift (files coming and going) from size noise.
        let structure = |f: &[(String, Kind, u64, u64)]| f.iter().map(|x| (x.0.clone(), x.1, x.3)).collect::<HashSet<_>>();
        let structural = structure(&a).symmetric_difference(&structure(&b)).count();
        let gate = if a == b && (incr.errors, incr.cloud_only) == (full.errors, full.cloud_only) {
            "gate ok".to_string()
        } else {
            all_ok = false;
            format!("MISMATCH ({differ} paths differ, {structural} in paths or item counts)")
        };
        println!("run {run}: incremental {incr_s:.3} s ({how}) full {full_s:.3} s {gate}");
    }
    all_ok
}

/// Bytes in use on the volume, if `root` is the top of a volume (so a full scan of
/// it should account for roughly that much).
pub fn volume_used(root: &Path) -> Option<u64> {
    if mount_of(root)? != root {
        return None;
    }
    let (total, free) = statvfs(root)?;
    Some(total.saturating_sub(free))
}

/// Where the volume holding `path` is mounted.
fn mount_of(path: &Path) -> Option<PathBuf> {
    use std::ffi::{CStr, CString};
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statfs(c_path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    let mount_point = unsafe { CStr::from_ptr(stat.f_mntonname.as_ptr()) };
    Some(PathBuf::from(std::ffi::OsStr::from_bytes(mount_point.to_bytes())))
}

pub fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

fn flatten(raw: Raw, parent: Option<usize>, nodes: &mut Vec<Node>, inos: &mut Vec<u64>, stamps: &mut Vec<u64>) -> usize {
    let ix = nodes.len();
    inos.push(raw.ino);
    stamps.push(raw.stamp);
    nodes.push(Node {
        name: raw.name.into(),
        size: raw.size,
        kind: raw.kind,
        parent,
        children: Vec::with_capacity(raw.children.len()),
        items: raw.items,
    });
    for child in raw.children {
        let child_ix = flatten(child, Some(ix), nodes, inos, stamps);
        nodes[ix].children.push(child_ix);
    }
    ix
}

pub fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["KB", "MB", "GB", "TB", "PB"];
    if bytes < 1000 {
        return format!("{bytes} bytes");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    value /= 1000.0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if value >= 100.0 {
        format!("{value:.0} {}", UNITS[unit])
    } else if value >= 10.0 {
        format!("{value:.1} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

pub fn format_count(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

pub struct Volume {
    pub name: String,
    pub path: PathBuf,
    pub total: u64,
    pub free: u64,
}

fn statvfs(path: &Path) -> Option<(u64, u64)> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    let frsize = stat.f_frsize as u64;
    Some((stat.f_blocks as u64 * frsize, stat.f_bavail as u64 * frsize))
}

/// The startup disk plus anything mounted under /Volumes.
pub fn volumes() -> Vec<Volume> {
    let mut root_name = String::from("Startup Disk");
    let mut others = Vec::new();
    let root_dev = fs::metadata("/").map(|m| m.dev()).ok();
    let startup_container = disk::container_at(Path::new("/"));
    let data_dev = fs::metadata("/System/Volumes/Data").map(|m| m.dev()).ok();
    if let Ok(entries) = fs::read_dir("/Volumes") {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if fs::canonicalize(&path).ok().as_deref() == Some(Path::new("/")) {
                root_name = name;
                continue;
            }
            // Unmounted placeholders (e.g. /Volumes/Recovery) are plain folders on the startup disk.
            let dev = fs::metadata(&path).map(|m| m.dev()).ok();
            if dev.is_none() || dev == root_dev || dev == data_dev {
                continue;
            }
            // Recovery, Update and the like live in the startup disk's own container: they
            // are part of the startup disk, not separate disks.
            if startup_container.is_some() && disk::container_at(&path) == startup_container {
                continue;
            }
            if let Some((total, free)) = statvfs(&path) {
                if total > 0 {
                    others.push(Volume { name, path, total, free });
                }
            }
        }
    }
    let mut volumes = Vec::new();
    if let Some((total, free)) = statvfs(Path::new("/")) {
        volumes.push(Volume { name: root_name, path: PathBuf::from("/"), total, free });
    }
    others.sort_by(|a, b| a.name.cmp(&b.name));
    volumes.extend(others);
    volumes
}

#[cfg(test)]
mod tests {
    use super::*;
    use incremental::fingerprint;

    fn write(path: &Path, bytes: usize) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, vec![7u8; bytes]).unwrap();
    }

    #[test]
    fn scans_sizes_and_removes_nodes() {
        let dir = std::env::temp_dir().join(format!("petal-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        write(&dir.join("big/a.bin"), 1 << 20);
        write(&dir.join("big/nested/b.bin"), 1 << 19);
        write(&dir.join("small.txt"), 10);
        fs::hard_link(dir.join("big/a.bin"), dir.join("link.bin")).unwrap();

        let tree = scan(&dir, &Progress::default());
        let root = &tree.nodes[Tree::ROOT];
        assert_eq!(root.items, 4);
        // Two links to the same 1 MiB file count once, whichever the parallel walk sees first.
        assert!(root.size < (1 << 20) * 2);
        assert!(root.size >= (1 << 20) + (1 << 19));
        let big = root.children.iter().copied()
            .find(|&c| tree.nodes[c].name.as_ref() == "big").unwrap();
        // Children are sorted largest first and path_of round-trips.
        let sizes: Vec<u64> = root.children.iter().map(|&c| tree.nodes[c].size).collect();
        assert!(sizes.windows(2).all(|w| w[0] >= w[1]));
        assert_eq!(tree.path_of(big), dir.join("big"));

        let mut tree = tree;
        let nested = tree.nodes[big].children.iter().copied()
            .find(|&c| tree.nodes[c].name.as_ref() == "nested").unwrap();
        let (before_root, nested_size) = (tree.nodes[Tree::ROOT].size, tree.nodes[nested].size);
        tree.remove(nested);
        assert_eq!(tree.nodes[Tree::ROOT].size, before_root - nested_size);
        assert_eq!(tree.nodes[Tree::ROOT].items, 3);
        assert!(!tree.nodes[big].children.contains(&nested));

        fs::remove_dir_all(&dir).unwrap();
    }

    /// Hotspots are scanned first and spliced in; the result must be identical to a
    /// plain scan, and the live totals must add up to the same bytes.
    #[test]
    fn hotspot_splice_matches_plain_scan() {
        let dir = std::env::temp_dir().join(format!("petal-hotspots-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let home = dir.join("Users/someone");
        write(&home.join("Downloads/movie.mov"), 3 << 20);
        write(&home.join("Downloads/nested/deeper/a.zip"), 1 << 20);
        write(&home.join(".Trash/old.dmg"), 2 << 20);
        write(&home.join("Library/Caches/com.app/blob"), 1 << 19);
        write(&home.join("Library/Prefs/p.plist"), 100);
        write(&home.join("Documents/notes.txt"), 5000);
        write(&dir.join("Applications/App.app/bin"), 1 << 20);

        let fingerprint = |tree: &Tree| {
            let root = &tree.nodes[Tree::ROOT];
            (root.size, root.items, tree.nodes.len())
        };
        let plain = scan_with_bases(&dir, &Progress::default(), &findings::Bases::default());
        let progress = Progress::default();
        let spliced = scan_with_bases(&dir, &progress, &findings::Bases { home: Some(home.clone()), user_temp: None });
        assert_eq!(fingerprint(&plain), fingerprint(&spliced));
        let live = crate::live::snapshot(&progress.live, &dir, None).tree;
        assert_eq!(live.nodes[Tree::ROOT].size, spliced.nodes[Tree::ROOT].size, "live totals");
        // Same per-folder sizes, not just the same total.
        let downloads = |tree: &Tree| {
            let users = tree.nodes[Tree::ROOT].children.iter().find(|&&c| tree.nodes[c].name.as_ref() == "Users").copied().unwrap();
            let someone = tree.nodes[users].children[0];
            tree.nodes[someone].children.iter().map(|&c| (tree.nodes[c].name.to_string(), tree.nodes[c].size)).collect::<Vec<_>>()
        };
        assert_eq!(downloads(&plain), downloads(&spliced));
        fs::remove_dir_all(&dir).unwrap();
    }

    fn cache_of(tree: &Tree, meta: &ScanMeta) -> crate::cache::Cached {
        let header = crate::cache::Header {
            root: tree.root_path.clone(),
            device_uuid: "test".into(),
            event_id: 0,
            root_ino: meta.inos[Tree::ROOT],
            has_fda: true,
        };
        crate::cache::decode(&crate::cache::encode(tree, meta, &header)).unwrap()
    }

    fn full(dir: &Path) -> (Tree, ScanMeta) {
        scan_with_meta(dir, &Progress::default(), &findings::Bases::default())
    }

    fn rescan(dir: &Path, cached: crate::cache::Cached, dirty: &[&str]) -> (Result<(Tree, ScanMeta), incremental::Fallback>, Progress) {
        let progress = Progress::default();
        let dirty: Vec<(PathBuf, bool)> = dirty.iter().map(|d| (dir.join(d), false)).collect();
        (incremental::rescan(dir, &progress, cached, &dirty), progress)
    }

    #[test]
    fn cache_round_trips() {
        let dir = std::env::temp_dir().join(format!("petal-cache-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        write(&dir.join("a/one.bin"), 5000);
        write(&dir.join("a/b/two.bin"), 70000);
        write(&dir.join("üñí/three"), 1);
        fs::hard_link(dir.join("a/one.bin"), dir.join("a/link.bin")).unwrap();
        write(&dir.join("locked/x"), 1);
        fs::set_permissions(dir.join("locked"), std::os::unix::fs::PermissionsExt::from_mode(0o000)).unwrap();
        let (tree, meta) = full(&dir);
        fs::set_permissions(dir.join("locked"), std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        assert!(meta.marks.iter().any(|m| m.1.errors > 0) && meta.marks.iter().any(|m| m.1.hardlinks));

        let header = crate::cache::Header { root: dir.clone(), device_uuid: "uuid".into(), event_id: 42, root_ino: meta.inos[0], has_fda: false };
        let file = dir.join("scan.cache");
        crate::cache::write_atomic(&file, &crate::cache::encode(&tree, &meta, &header)).unwrap();
        let buf = fs::read(&file).unwrap();
        let cached = crate::cache::decode(&buf).unwrap();
        assert_eq!(cached.header, header);
        assert_eq!(cached.inos, meta.inos);
        assert_eq!(cached.stamps, meta.stamps);
        assert!(meta.stamps[Tree::ROOT] != 0);
        assert_eq!(cached.marks, meta.marks);
        let decoded = Tree { root_path: dir.clone(), nodes: cached.nodes, errors: 0, cloud_only: 0 };
        assert_eq!(fingerprint(&decoded), fingerprint(&tree));
        for (ix, &end) in cached.sub_end.iter().enumerate() {
            assert!((ix + 1..end).all(|k| decoded.is_ancestor_or_self(ix, k)));
        }
        for len in 0..buf.len() {
            assert!(crate::cache::decode(&buf[..len]).is_none());
        }
        for at in 0..buf.len() {
            let mut bad = buf.clone();
            bad[at] ^= 0x10;
            assert!(crate::cache::decode(&bad).is_none());
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn incremental_matches_full() {
        let dir = std::env::temp_dir().join(format!("petal-incremental-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        write(&dir.join("keep/a/b/c/d/deep.bin"), 4096);
        write(&dir.join("keep/sib/x.bin"), 9000);
        write(&dir.join("add/f1"), 100);
        write(&dir.join("grow/g.bin"), 100);
        write(&dir.join("del/sub/x"), 20000);
        write(&dir.join("x/old"), 3000);
        write(&dir.join("y/new"), 7000);
        for name in ["e1", "e2", "e3"] {
            write(&dir.join("tie").join(name), 8192);
        }
        write(&dir.join("tie/grower"), 10);
        for i in 0..20 {
            write(&dir.join(format!("clean/n{i}/f")), 100 * i);
        }
        let (a, meta) = full(&dir);
        let cached = cache_of(&a, &meta);

        write(&dir.join("add/f2"), 50000);
        write(&dir.join("grow/g.bin"), 300000);
        fs::remove_dir_all(dir.join("del/sub")).unwrap();
        write(&dir.join("new/l1/l2/f"), 12345);
        fs::rename(dir.join("x"), dir.join("x.old")).unwrap();
        fs::rename(dir.join("y"), dir.join("x")).unwrap();
        write(&dir.join("tie/grower"), 8192);
        write(&dir.join("keep/a/b/c/d/deep.bin"), 200000);
        let dirty = ["", "add", "grow", "del", "del/sub", "new", "new/l1", "new/l1/l2", "tie", "keep/a/b/c/d"];
        let (b, progress) = rescan(&dir, cached, &dirty);
        let (b, meta_b) = b.unwrap();
        let full_progress = Progress::default();
        let (c, meta_c) = scan_with_meta(&dir, &full_progress, &findings::Bases::default());
        assert_eq!(fingerprint(&b), fingerprint(&c));
        assert_eq!((b.errors, b.cloud_only), (c.errors, c.cloud_only));
        assert_eq!(meta_b.inos, meta_c.inos);
        assert_eq!(meta_b.marks, meta_c.marks);
        assert_eq!(progress.files.load(Ordering::Relaxed), full_progress.files.load(Ordering::Relaxed));
        let live = crate::live::snapshot(&progress.live, &dir, None).tree;
        assert_eq!(live.nodes[Tree::ROOT].size, b.nodes[Tree::ROOT].size, "live totals");

        // Not vacuous: an unreported change in a clean folder stays unseen, so it was reused.
        let cached = cache_of(&b, &meta_b);
        write(&dir.join("clean/n3/f"), 400000);
        let (stale, _) = rescan(&dir, cached, &[]);
        assert_eq!(fingerprint(&stale.unwrap().0), fingerprint(&b));
        assert_ne!(fingerprint(&full(&dir).0), fingerprint(&b));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn incremental_falls_back_on_hardlinks() {
        let dir = std::env::temp_dir().join(format!("petal-incr-links-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        write(&dir.join("a/f"), 5000);
        write(&dir.join("b/g"), 5000);
        let (tree, meta) = full(&dir);
        // A new link in a dirty folder.
        fs::hard_link(dir.join("a/f"), dir.join("a/l")).unwrap();
        let (result, _) = rescan(&dir, cache_of(&tree, &meta), &["a"]);
        assert_eq!(result.err(), Some(incremental::Fallback::HardLinks));
        // A cached link on a changed path.
        let (tree, meta) = full(&dir);
        write(&dir.join("a/h"), 1);
        let (result, _) = rescan(&dir, cache_of(&tree, &meta), &["a"]);
        assert_eq!(result.err(), Some(incremental::Fallback::HardLinks));
        // Links off the changed path are fine.
        let (tree, meta) = full(&dir);
        write(&dir.join("b/h"), 1);
        let (result, _) = rescan(&dir, cache_of(&tree, &meta), &["b"]);
        assert!(result.is_ok());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn incremental_relists_error_dirs() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("petal-incr-errors-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        write(&dir.join("locked/inner/x"), 30000);
        write(&dir.join("open/y"), 1000);
        fs::set_permissions(dir.join("locked"), fs::Permissions::from_mode(0o000)).unwrap();
        let (a, meta) = full(&dir);
        fs::set_permissions(dir.join("locked"), fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(a.errors, 1);
        // No event for it, but now readable: it must be listed again.
        let (b, _) = rescan(&dir, cache_of(&a, &meta), &[]);
        let (b, _) = b.unwrap();
        let c = full(&dir).0;
        assert_eq!(b.errors, 0);
        assert_eq!(fingerprint(&b), fingerprint(&c));
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Undoes `access`'s changes even if a test fails, so the fixture can be deleted.
    struct Restore(PathBuf);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = std::process::Command::new("chmod").args(["-R", "-N"]).arg(&self.0).status();
            let _ = std::process::Command::new("chmod").args(["-R", "u+rwx"]).arg(&self.0).status();
        }
    }

    /// A group this user is in other than `path`'s, so `chgrp` needs no privileges.
    fn other_group(path: &Path) -> u32 {
        let current = fs::metadata(path).unwrap().gid();
        let out = std::process::Command::new("id").arg("-G").output().unwrap();
        String::from_utf8(out.stdout).unwrap().split_whitespace().map(|g| g.parse().unwrap()).find(|&g| g != current).expect("a second group")
    }

    /// Ways to change a folder's access without touching its contents or its parent's listing.
    fn access(change: &str, path: &Path) {
        use std::os::unix::fs::PermissionsExt;
        match change {
            "chmod" => fs::set_permissions(path, fs::Permissions::from_mode(0o000)).unwrap(),
            "chgrp" => std::os::unix::fs::chown(path, None, Some(other_group(path))).unwrap(),
            // An ACL alone: the POSIX mode stays as it was.
            "acl" => assert!(std::process::Command::new("chmod").args(["+a", "everyone deny list,search"]).arg(path).status().unwrap().success()),
            _ => unreachable!(),
        }
    }

    /// The parent's event is all a mode, owner or ACL change of a clean folder brings; its
    /// cached subtree must not be reused, and the result must stay right on later rescans.
    #[test]
    fn incremental_sees_access_changes() {
        for change in ["chmod", "chgrp", "acl"] {
            let dir = std::env::temp_dir().join(format!("petal-incr-access-{change}-{}", std::process::id()));
            let _restore = Restore(dir.clone());
            let _ = fs::remove_dir_all(&dir);
            write(&dir.join("a/b/c/file"), 300_000);
            write(&dir.join("a/b/c/d/deeper"), 5000);
            for i in 0..5 {
                write(&dir.join(format!("a/b/sib{i}/f")), 1000);
            }
            let (a, meta) = full(&dir);
            access(change, &dir.join("a/b/c"));
            let (b, progress) = rescan(&dir, cache_of(&a, &meta), &["a/b"]);
            let (b, meta_b) = b.unwrap();
            let (c, meta_c) = full(&dir);
            assert_eq!(fingerprint(&b), fingerprint(&c), "{change}");
            assert_eq!((b.errors, b.cloud_only), (c.errors, c.cloud_only), "{change}");
            assert_eq!(meta_b.marks, meta_c.marks, "{change}");
            if change != "chgrp" {
                assert!(c.errors > 0 && c.find(&dir.join("a/b/c/file")).is_none(), "{change}: c is unreadable now");
            }
            // root, a, b re-listed, c walked again (and d, if readable); the siblings reused.
            let reused = progress.reused_dirs.load(Ordering::Relaxed);
            assert_eq!(reused, 5, "{change}");
            assert!(progress.dirs.load(Ordering::Relaxed) - reused >= 4, "{change}: c was reused");
            // Saved and loaded again, a quiet rescan stays right.
            let (quiet, _) = rescan(&dir, cache_of(&b, &meta_b), &[]);
            let quiet = quiet.unwrap().0;
            assert_eq!(fingerprint(&quiet), fingerprint(&c), "{change}");
            assert_eq!(quiet.errors, c.errors, "{change}");
            drop(_restore);
            fs::remove_dir_all(&dir).unwrap();
        }
    }

    /// Real FSEvents history: a cached scan, then changes, then an incremental rescan
    /// that must take the incremental path and match a full scan exactly.
    #[test]
    fn fsevents_incremental_end_to_end() {
        let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/target/petal-fsevents-test"));
        let _ = fs::remove_dir_all(&dir);
        write(&dir.join("keep/a/b/deep.bin"), 4096);
        write(&dir.join("grow/g.bin"), 100);
        write(&dir.join("del/sub/x"), 20000);
        for i in 0..10 {
            write(&dir.join(format!("clean/n{i}/f")), 100 * i);
        }
        let dev = fs::metadata(&dir).unwrap().dev();
        if fsevents::device_uuid(dev).is_none() {
            eprintln!("skipped: no FSEvents history on this volume");
            return;
        }
        // SAFETY: the only test that reads PETAL_CACHE_DIR.
        unsafe { std::env::set_var("PETAL_CACHE_DIR", dir.with_extension("cache")) };
        let (tree, save) = scan_cached(&dir, &Progress::default(), true);
        let save = save.expect("internal volume with history");
        assert_eq!(save.how, How::Full("forced"));
        save.save(&tree);
        let since = cache::load(&dir, &fsevents::device_uuid(dev).unwrap(), onboarding::has_full_disk_access()).unwrap().header.event_id;

        write(&dir.join("grow/g.bin"), 300000);
        fs::remove_dir_all(dir.join("del/sub")).unwrap();
        write(&dir.join("new/l1/f"), 12345);
        write(&dir.join("keep/a/b/deep.bin"), 200000);
        // Test-only: wait for fseventsd to have the changes on record.
        let (canonical, mount) = (fs::canonicalize(&dir).unwrap(), mount_of(&dir).unwrap());
        let wanted = ["grow", "del", "new/l1", "keep/a/b"].map(PathBuf::from);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let changes = fsevents::changes_since(dev, &mount, &canonical, since);
            let fsevents::Changes::Dirs(dirs) = &changes else { panic!("{changes:?}") };
            if wanted.iter().all(|w| dirs.iter().any(|d| &d.0 == w)) {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "events never arrived: {dirs:?}");
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        let (incr, save) = scan_cached(&dir, &Progress::default(), false);
        let save = save.unwrap();
        assert!(matches!(save.how, How::Incremental { reused, .. } if reused > 0));
        let full = scan(&dir, &Progress::default());
        assert_eq!(fingerprint(&incr), fingerprint(&full));
        assert_eq!((incr.errors, incr.cloud_only), (full.errors, full.cloud_only));
        save.save(&incr);

        // Access changes: the event names only the parent. Each must converge to a full scan,
        // and stay converged on a quiet rescan from the saved result.
        let _restore = Restore(dir.clone());
        for (change, target) in [("chgrp", "clean/n1"), ("chmod", "clean/n2"), ("acl", "clean/n3")] {
            let since = cache::load(&dir, &fsevents::device_uuid(dev).unwrap(), onboarding::has_full_disk_access()).unwrap().header.event_id;
            access(change, &dir.join(target));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                let changes = fsevents::changes_since(dev, &mount, &canonical, since);
                let fsevents::Changes::Dirs(dirs) = &changes else { panic!("{changes:?}") };
                if dirs.iter().any(|d| d.0 == Path::new("clean") || d.0 == Path::new(target)) {
                    break;
                }
                assert!(std::time::Instant::now() < deadline, "{change}: event never arrived: {dirs:?}");
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            for round in ["after the event", "quiet"] {
                let (incr, save) = scan_cached(&dir, &Progress::default(), false);
                let save = save.unwrap();
                let full = scan(&dir, &Progress::default());
                assert!(matches!(save.how, How::Incremental { .. }), "{change} {round}: {:?}", save.how);
                assert_eq!(fingerprint(&incr), fingerprint(&full), "{change} {round}");
                assert_eq!((incr.errors, incr.cloud_only), (full.errors, full.cloud_only), "{change} {round}");
                save.save(&incr);
            }
            if change != "chgrp" {
                assert!(scan(&dir, &Progress::default()).errors > 0, "{change}: {target} is unreadable now");
            }
        }
        drop(_restore);
        fs::remove_dir_all(&dir).unwrap();
        let _ = fs::remove_dir_all(dir.with_extension("cache"));
    }

    #[test]
    fn formats_sizes() {
        assert_eq!(format_size(512), "512 bytes");
        assert_eq!(format_size(1_500_000), "1.50 MB");
        assert_eq!(format_size(13_400_000_000), "13.4 GB");
        assert_eq!(format_count(1345677), "1,345,677");
    }
    /// Ground truth for clone accounting on a real APFS volume: a disk image with an
    /// original file, three pure clones, an edited clone and a plain file. Checks the
    /// scanned total against the volume's own usage, then predicts what each deletion
    /// frees and compares with what the volume actually frees.
    /// Run with `cargo test -- --ignored clone_accounting_matches_apfs` (needs `hdiutil`).
    #[test]
    #[ignore]
    fn clone_accounting_matches_apfs() {
        use std::process::Command;
        let run = |cmd: &str, args: &[&str]| {
            let out = Command::new(cmd).args(args).output().unwrap();
            assert!(out.status.success(), "{cmd} {args:?}: {}", String::from_utf8_lossy(&out.stderr));
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        let name = format!("PetalClones{}", std::process::id());
        let image = std::env::temp_dir().join(format!("{name}.sparseimage"));
        let _ = fs::remove_file(&image);
        run("hdiutil", &["create", "-size", "300m", "-fs", "APFS", "-volname", &name, "-type", "SPARSE", image.to_str().unwrap()]);
        let attach = run("hdiutil", &["attach", "-nobrowse", image.to_str().unwrap()]);
        let mount = PathBuf::from(attach.lines().last().unwrap().split('\t').last().unwrap().trim());
        let used = || {
            run("sync", &[]);
            std::thread::sleep(std::time::Duration::from_millis(500));
            crate::disk::volume_used(&mount).unwrap()
        };

        let random = |path: &Path, mb: usize| {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut data = vec![0u8; mb << 20];
            let mut x = 0x9e3779b97f4a7c15u64 ^ mb as u64;
            for chunk in data.chunks_mut(8) {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                chunk.copy_from_slice(&x.to_le_bytes()[..chunk.len()]);
            }
            fs::write(path, data).unwrap();
        };
        let before = used();
        random(&mount.join("a/orig.bin"), 40);
        for i in 1..=3 {
            fs::create_dir_all(mount.join("b")).unwrap();
            run("cp", &["-c", mount.join("a/orig.bin").to_str().unwrap(), mount.join(format!("b/clone{i}.bin")).to_str().unwrap()]);
        }
        fs::create_dir_all(mount.join("c")).unwrap();
        run("cp", &["-c", mount.join("a/orig.bin").to_str().unwrap(), mount.join("c/edited.bin").to_str().unwrap()]);
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = fs::OpenOptions::new().write(true).open(mount.join("c/edited.bin")).unwrap();
            f.seek(SeekFrom::Start(10 << 20)).unwrap();
            f.write_all(&vec![7u8; 4 << 20]).unwrap();
        }
        random(&mount.join("d/plain.bin"), 8);
        random(&mount.join("e/linked.bin"), 6);
        fs::create_dir_all(mount.join("f")).unwrap();
        fs::hard_link(mount.join("e/linked.bin"), mount.join("f/link.bin")).unwrap();
        let truth = used() - before;

        let mb = |b: u64| b as f64 / 1e6;
        let tree = scan(&mount, &Progress::default());
        let counted = tree.nodes[Tree::ROOT].size;
        // Sizes are allocation (like Finder): each clone counts its blocks. "Frees" is what
        // must match reality, checked below.
        eprintln!("truth {:.1} MB, allocated (sizes shown) {:.1} MB", mb(truth), mb(counted));

        // Predict, delete, measure; one folder at a time, rescanning in between.
        // e/ first: its file has another link in f/, so deleting e/ frees nothing yet.
        for folder in ["d", "c", "b", "a", "e", "f"] {
            let predicted = frees_of(&[mount.join(folder)]);
            let before = used();
            fs::remove_dir_all(mount.join(folder)).unwrap();
            let actual = before - used();
            eprintln!("delete {folder}/: predicted {:.2} MB, actually freed {:.2} MB", mb(predicted), mb(actual));
            let tolerance = 1 << 20;
            assert!(predicted.abs_diff(actual) <= tolerance, "{folder}: predicted {predicted}, freed {actual}");
        }
        run("hdiutil", &["detach", mount.to_str().unwrap()]);
        fs::remove_file(&image).unwrap();
    }

}
