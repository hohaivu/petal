//! What changed on a volume since a past FSEvents event id, from the volume's own
//! event history (hand-rolled FFI; FSEvents lives in CoreServices).

use std::ffi::{CStr, c_char, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

type CFRef = *const c_void;

#[repr(C)]
struct StreamContext {
    version: isize,
    info: *mut c_void,
    retain: *const c_void,
    release: *const c_void,
    copy_description: *const c_void,
}

type Callback = extern "C" fn(CFRef, *mut c_void, usize, *mut c_void, *const u32, *const u64);

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFTypeArrayCallBacks: c_void;
    fn CFStringCreateWithBytes(alloc: CFRef, bytes: *const u8, len: isize, encoding: u32, external: u8) -> CFRef;
    fn CFArrayCreate(alloc: CFRef, values: *const CFRef, count: isize, callbacks: *const c_void) -> CFRef;
    fn CFRelease(cf: CFRef);
    fn CFUUIDCreateString(alloc: CFRef, uuid: CFRef) -> CFRef;
    fn CFStringGetCString(s: CFRef, buf: *mut c_char, size: isize, encoding: u32) -> u8;
}

#[link(name = "CoreServices", kind = "framework")]
unsafe extern "C" {
    fn FSEventsCopyUUIDForDevice(dev: libc::dev_t) -> CFRef;
    fn FSEventsGetCurrentEventId() -> u64;
    fn FSEventStreamCreateRelativeToDevice(
        alloc: CFRef,
        callback: Callback,
        context: *const StreamContext,
        dev: libc::dev_t,
        paths: CFRef,
        since: u64,
        latency: f64,
        flags: u32,
    ) -> *mut c_void;
    fn FSEventStreamSetDispatchQueue(stream: *mut c_void, queue: *mut c_void);
    fn FSEventStreamStart(stream: *mut c_void) -> u8;
    fn FSEventStreamStop(stream: *mut c_void);
    fn FSEventStreamInvalidate(stream: *mut c_void);
    fn FSEventStreamRelease(stream: *mut c_void);
}

unsafe extern "C" {
    fn dispatch_queue_create(label: *const c_char, attr: *const c_void) -> *mut c_void;
    fn dispatch_sync_f(queue: *mut c_void, context: *mut c_void, work: extern "C" fn(*mut c_void));
    fn dispatch_release(object: *mut c_void);
}

const UTF8: u32 = 0x0800_0100;
const NO_DEFER: u32 = 0x02;
const MUST_SCAN_SUB_DIRS: u32 = 0x01;
const USER_DROPPED: u32 = 0x02;
const KERNEL_DROPPED: u32 = 0x04;
const IDS_WRAPPED: u32 = 0x08;
const HISTORY_DONE: u32 = 0x10;
const ROOT_CHANGED: u32 = 0x20;
const MOUNT: u32 = 0x40;
const UNMOUNT: u32 = 0x80;

#[derive(Debug, PartialEq, Eq)]
pub enum Changes {
    /// Folders whose listing changed, relative to the root (`true`: everything below too).
    Dirs(Vec<(PathBuf, bool)>),
    /// The history can't say: scan everything.
    Reset(&'static str),
}

/// The volume's FSEvents UUID; `None` when it keeps no event history.
pub fn device_uuid(dev: u64) -> Option<String> {
    unsafe {
        let uuid = FSEventsCopyUUIDForDevice(dev as libc::dev_t);
        if uuid.is_null() {
            return None;
        }
        let s = CFUUIDCreateString(std::ptr::null(), uuid);
        CFRelease(uuid);
        let mut buf = [0 as c_char; 64];
        let ok = CFStringGetCString(s, buf.as_mut_ptr(), buf.len() as isize, UTF8) != 0;
        CFRelease(s);
        ok.then(|| CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned())
    }
}

/// The latest event id: everything changed after this gets a later one. Event ids are
/// one system-wide sequence that device streams share. (Empirically,
/// `FSEventsGetLastEventIdForDeviceBeforeTime` returns 0 for the Data volume.)
pub fn current_id() -> u64 {
    unsafe { FSEventsGetCurrentEventId() }
}

/// Where a device-relative event path lands under `root`, if it does. On the startup
/// disk's Data volume, firmlinked folders appear at `/` (`/Users`) as well.
pub fn to_root_relative(event: &Path, mount: &Path, root: &Path) -> Option<PathBuf> {
    candidates(event, mount).into_iter().find_map(|p| p.strip_prefix(root).ok().map(Path::to_path_buf))
}

fn candidates(event: &Path, mount: &Path) -> Vec<PathBuf> {
    let rel = event.strip_prefix("/").unwrap_or(event);
    let mut out = vec![mount.join(rel)];
    if mount == Path::new("/System/Volumes/Data") {
        out.push(Path::new("/").join(rel));
    }
    out
}

#[derive(Default)]
struct Collected {
    dirs: Vec<(PathBuf, bool)>,
    reset: Option<&'static str>,
    done: bool,
}

struct State<'a> {
    mount: &'a Path,
    root: &'a Path,
    collected: Mutex<Collected>,
    signal: Condvar,
}

extern "C" fn on_events(_: CFRef, info: *mut c_void, count: usize, paths: *mut c_void, flags: *const u32, _: *const u64) {
    let state = unsafe { &*(info as *const State) };
    let paths = unsafe { std::slice::from_raw_parts(paths as *const *const c_char, count) };
    let flags = unsafe { std::slice::from_raw_parts(flags, count) };
    let mut c = state.collected.lock().unwrap();
    for (&path, &flag) in paths.iter().zip(flags) {
        if flag & (USER_DROPPED | KERNEL_DROPPED) != 0 {
            c.reset.get_or_insert("events dropped");
        } else if flag & IDS_WRAPPED != 0 {
            c.reset.get_or_insert("event ids wrapped");
        } else if flag & (ROOT_CHANGED | MOUNT | UNMOUNT) != 0 {
            // Also how a replay from before the volume's history begins (Mount|Unmount on "").
            c.reset.get_or_insert("root changed, volume remounted, or history too old");
        }
        if flag & HISTORY_DONE != 0 {
            c.done = true;
            continue;
        }
        let event = Path::new(std::ffi::OsStr::from_bytes(unsafe { CStr::from_ptr(path) }.to_bytes()));
        let recursive = flag & MUST_SCAN_SUB_DIRS != 0;
        if let Some(rel) = to_root_relative(event, state.mount, state.root) {
            c.dirs.push((rel, recursive));
        } else if recursive && candidates(event, state.mount).iter().any(|p| state.root.starts_with(p)) {
            c.reset.get_or_insert("must rescan above the root");
        }
    }
    state.signal.notify_all();
}

extern "C" fn drain(_: *mut c_void) {}

/// Folders under `root` (canonical) on device `dev` mounted at `mount` that changed after
/// event `since`. Waits at most 10 s for the history replay.
pub fn changes_since(dev: u64, mount: &Path, root: &Path, since: u64) -> Changes {
    if since > current_id() {
        return Changes::Reset("event id ahead of the volume's history");
    }
    let state = State { mount, root, collected: Mutex::default(), signal: Condvar::new() };
    let context = StreamContext {
        version: 0,
        info: &state as *const State as *mut c_void,
        retain: std::ptr::null(),
        release: std::ptr::null(),
        copy_description: std::ptr::null(),
    };
    unsafe {
        // "" watches the whole device (empirically: every event on the volume arrives,
        // with paths relative to its mount point and no leading slash).
        let path = CFStringCreateWithBytes(std::ptr::null(), b"".as_ptr(), 0, UTF8, 0);
        let paths = CFArrayCreate(std::ptr::null(), &path, 1, &kCFTypeArrayCallBacks as *const c_void);
        let stream = FSEventStreamCreateRelativeToDevice(std::ptr::null(), on_events, &context, dev as libc::dev_t, paths, since, 0.0, NO_DEFER);
        CFRelease(paths);
        CFRelease(path);
        if stream.is_null() {
            return Changes::Reset("could not open the event stream");
        }
        let queue = dispatch_queue_create(c"petal.fsevents".as_ptr(), std::ptr::null());
        FSEventStreamSetDispatchQueue(stream, queue);
        let started = FSEventStreamStart(stream) != 0;
        let timed_out = started && {
            let guard = state.collected.lock().unwrap();
            let waited = state.signal.wait_timeout_while(guard, Duration::from_secs(10), |c| !c.done && c.reset.is_none()).unwrap();
            waited.1.timed_out()
        };
        if started {
            FSEventStreamStop(stream);
        }
        FSEventStreamInvalidate(stream);
        FSEventStreamRelease(stream);
        // Let any callback already on the queue finish before `state` goes away.
        dispatch_sync_f(queue, std::ptr::null_mut(), drain);
        dispatch_release(queue);
        if !started {
            return Changes::Reset("could not start the event stream");
        }
        if timed_out {
            return Changes::Reset("timeout");
        }
    }
    let c = state.collected.into_inner().unwrap();
    match c.reset {
        Some(reason) => Changes::Reset(reason),
        None => Changes::Dirs(c.dirs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_device_paths_to_root() {
        let rel = |e: &str, m: &str, r: &str| to_root_relative(Path::new(e), Path::new(m), Path::new(r));
        assert_eq!(rel("photos/2024/", "/Volumes/Ext", "/Volumes/Ext"), Some("photos/2024".into()));
        assert_eq!(rel("Users/x/Downloads", "/System/Volumes/Data", "/System/Volumes/Data"), Some("Users/x/Downloads".into()));
        assert_eq!(rel("Users/x/Downloads/", "/System/Volumes/Data", "/Users/x"), Some("Downloads".into()));
        assert_eq!(rel("Users/y/Desktop", "/System/Volumes/Data", "/Users/x"), None);
        assert_eq!(rel("/a/b", "/", "/a"), Some("b".into()));
    }
}
