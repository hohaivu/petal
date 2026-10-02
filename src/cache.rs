//! The scan cache: a full tree plus what an incremental rescan needs, in a compact
//! little-endian binary with LEB128 varints and an FNV-1a checksum trailer.
//!
//! Nodes are stored in pre-order from the root (skipping `Kind::Other` slices), so every
//! subtree decodes to a contiguous index range. Folder sizes and item counts are
//! recomputed on decode; only a folder's own allocation is stored.

use std::fs;
use std::io;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use crate::scan::{Kind, Link, Mark, Node, ScanMeta, Tree};

const MAGIC: &[u8; 8] = b"PETALSC1";
/// Bump whenever scan semantics change.
const VERSION: u32 = 3;
const TAG_FILE: u8 = 0;
const TAG_DIR: u8 = 1;
const TAG_MARKED: u8 = 0x80;
/// A hard-linked file: dev, ino and the file's size follow its charged size.
const TAG_LINK: u8 = 0x40;
const FLAG_DATALESS: u8 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub root: PathBuf,
    /// FSEvents device UUID and event id the cache is current as of.
    pub device_uuid: String,
    pub event_id: u64,
    pub root_ino: u64,
    pub has_fda: bool,
}

pub struct Cached {
    pub header: Header,
    /// Pre-order: node `ix`'s subtree is `ix..sub_end[ix]`.
    pub nodes: Vec<Node>,
    pub inos: Vec<u64>,
    pub stamps: Vec<u64>,
    /// Sorted by node index.
    pub marks: Vec<(usize, Mark)>,
    /// Sorted by node index.
    pub links: Vec<(usize, Link)>,
    pub sub_end: Vec<usize>,
}

/// Entries of `v` (sorted by node index) within `ix..end`.
fn in_range<T>(v: &[(usize, T)], ix: usize, end: usize) -> &[(usize, T)] {
    &v[v.partition_point(|m| m.0 < ix)..v.partition_point(|m| m.0 < end)]
}

impl Cached {
    /// A folder's own allocation (its size minus its children's).
    pub fn own(&self, ix: usize) -> u64 {
        let node = &self.nodes[ix];
        node.size - node.children.iter().map(|&c| self.nodes[c].size).sum::<u64>()
    }

    pub fn mark(&self, ix: usize) -> Mark {
        match self.marks.binary_search_by_key(&ix, |m| m.0) {
            Ok(at) => self.marks[at].1,
            Err(_) => Mark::default(),
        }
    }

    /// Marks within `ix..end`.
    pub fn marks_in(&self, ix: usize, end: usize) -> &[(usize, Mark)] {
        in_range(&self.marks, ix, end)
    }

    /// Hard links within `ix..end`.
    pub fn links_in(&self, ix: usize, end: usize) -> &[(usize, Link)] {
        in_range(&self.links, ix, end)
    }
}

pub fn encode(tree: &Tree, meta: &ScanMeta, header: &Header) -> Vec<u8> {
    let real = |c: &usize| tree.nodes[*c].kind != Kind::Other;
    let mut order = Vec::with_capacity(tree.nodes.len());
    let mut stack = vec![Tree::ROOT];
    while let Some(ix) = stack.pop() {
        order.push(ix);
        stack.extend(tree.nodes[ix].children.iter().rev().filter(|c| real(c)));
    }

    let mut fields = Vec::new();
    put_bytes(&mut fields, header.root.as_os_str().as_bytes());
    put_bytes(&mut fields, header.device_uuid.as_bytes());
    put_varint(&mut fields, header.event_id);
    put_varint(&mut fields, header.root_ino);
    fields.push(header.has_fda as u8);
    put_varint(&mut fields, order.len() as u64);

    let mut out = Vec::with_capacity(16 + fields.len() + order.len() * 28);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    put_bytes(&mut out, &fields);
    for ix in order {
        let node = &tree.nodes[ix];
        let mark = meta.marks.binary_search_by_key(&ix, |m| m.0).ok().map(|at| meta.marks[at].1);
        let link = meta.links.binary_search_by_key(&ix, |l| l.0).ok().map(|at| meta.links[at].1);
        let tag = if node.kind == Kind::File { TAG_FILE } else { TAG_DIR };
        out.push(tag | if mark.is_some() { TAG_MARKED } else { 0 } | if link.is_some() { TAG_LINK } else { 0 });
        put_bytes(&mut out, node.name.as_bytes());
        if node.kind == Kind::File {
            put_varint(&mut out, node.size);
            if let Some(link) = link {
                put_varint(&mut out, link.dev);
                put_varint(&mut out, link.ino);
                put_varint(&mut out, link.size);
            }
        } else {
            // Own allocation: all children count here, including any volume slices.
            let own = node.size.saturating_sub(node.children.iter().map(|&c| tree.nodes[c].size).sum());
            put_varint(&mut out, own);
            put_varint(&mut out, node.children.iter().filter(|c| real(c)).count() as u64);
            put_varint(&mut out, meta.inos.get(ix).copied().unwrap_or(0));
            put_varint(&mut out, meta.stamps.get(ix).copied().unwrap_or(0));
        }
        if let Some(mark) = mark {
            put_varint(&mut out, mark.errors);
            out.push(if mark.dataless { FLAG_DATALESS } else { 0 });
        }
    }
    let sum = fnv1a(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

/// `None` for anything malformed; never panics.
pub fn decode(buf: &[u8]) -> Option<Cached> {
    let (body, sum) = buf.split_at(buf.len().checked_sub(8)?);
    if fnv1a(body) != u64::from_le_bytes(sum.try_into().ok()?) {
        return None;
    }
    let mut r = Reader { buf: body, pos: 0 };
    if r.take(8)? != MAGIC || u32::from_le_bytes(r.take(4)?.try_into().ok()?) != VERSION {
        return None;
    }
    let mut f = Reader { buf: r.bytes()?, pos: 0 };
    let root = PathBuf::from(std::ffi::OsString::from_vec(f.bytes()?.to_vec()));
    let device_uuid = String::from_utf8(f.bytes()?.to_vec()).ok()?;
    let (event_id, root_ino, has_fda) = (f.varint()?, f.varint()?, f.byte()? != 0);
    let count = usize::try_from(f.varint()?).ok()?;
    // Every node takes at least 3 bytes, so a bigger count is corrupt.
    if count == 0 || count > body.len() / 3 {
        return None;
    }

    struct Frame {
        ix: usize,
        left: u64,
        own: u64,
    }
    let mut nodes: Vec<Node> = Vec::with_capacity(count);
    let mut inos = Vec::with_capacity(count);
    let mut stamps = Vec::with_capacity(count);
    let mut sub_end = vec![0; count];
    let mut marks = Vec::new();
    let mut links = Vec::new();
    let mut stack: Vec<Frame> = Vec::new();
    while nodes.len() < count {
        let ix = nodes.len();
        if ix > 0 && stack.is_empty() {
            return None;
        }
        let tag = r.byte()?;
        let kind = match tag & !(TAG_MARKED | TAG_LINK) {
            TAG_FILE if ix > 0 => Kind::File,
            TAG_DIR => Kind::Dir,
            _ => return None,
        };
        let name = std::str::from_utf8(r.bytes()?).ok()?;
        let parent = stack.last_mut().map(|frame| {
            frame.left -= 1;
            frame.ix
        });
        if let Some(parent) = parent {
            nodes[parent].children.push(ix);
        }
        let (size, items, ino) = if kind == Kind::File { (r.varint()?, 1, 0) } else { (0, 0, 0) };
        if tag & TAG_LINK != 0 {
            if kind != Kind::File {
                return None;
            }
            links.push((ix, Link { dev: r.varint()?, ino: r.varint()?, size: r.varint()?, fresh: false }));
        }
        nodes.push(Node { name: name.to_owned().into(), size, kind, parent, children: Vec::new(), items });
        if kind == Kind::Dir {
            let (own, left, ino, stamp) = (r.varint()?, r.varint()?, r.varint()?, r.varint()?);
            inos.push(ino);
            stamps.push(stamp);
            stack.push(Frame { ix, left, own });
        } else {
            inos.push(ino);
            stamps.push(0);
            sub_end[ix] = ix + 1;
        }
        if tag & TAG_MARKED != 0 {
            let (errors, flags) = (r.varint()?, r.byte()?);
            marks.push((ix, Mark { errors, dataless: flags & FLAG_DATALESS != 0 }));
        }
        // Close every folder whose children are all in.
        while let Some(frame) = stack.last().filter(|frame| frame.left == 0) {
            let (mut size, mut items) = (frame.own, 0u64);
            for &c in &nodes[frame.ix].children {
                size = size.checked_add(nodes[c].size)?;
                items = items.checked_add(nodes[c].items)?;
            }
            let node = &mut nodes[frame.ix];
            (node.size, node.items) = (size, items);
            sub_end[frame.ix] = nodes.len();
            stack.pop();
        }
    }
    if !stack.is_empty() || r.pos != body.len() {
        return None;
    }
    let header = Header { root, device_uuid, event_id, root_ino, has_fda };
    Some(Cached { header, nodes, inos, stamps, marks, links, sub_end })
}

/// Bundle id, so the unbundled binary shares the app's cache.
const BUNDLE_ID: &str = "io.github.henrydennis.petal";
/// Caches kept, newest first.
// ponytail: fixed count; size-based budget if users scan many roots.
const KEEP: usize = 3;

/// `PETAL_CACHE_DIR` overrides it (for tests).
fn dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("PETAL_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    Some(PathBuf::from(std::env::var_os("HOME")?).join("Library/Caches").join(BUNDLE_ID).join("scans"))
}

fn file_for(root: &Path) -> Option<PathBuf> {
    Some(dir()?.join(format!("{:016x}.bin", fnv1a(root.as_os_str().as_bytes()))))
}

/// Save and prune to the newest `KEEP`. Errors are ignored: it's only a cache.
pub fn save(tree: &Tree, meta: &ScanMeta, header: &Header) {
    let Some(file) = file_for(&header.root) else { return };
    let Some(dir) = file.parent() else { return };
    let _ = fs::create_dir_all(dir);
    if write_atomic(&file, &encode(tree, meta, header)).is_err() {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else { return };
    let mut bins: Vec<(std::time::SystemTime, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .filter_map(|p| Some((fs::metadata(&p).ok()?.modified().ok()?, p)))
        .collect();
    bins.sort_by_key(|b| std::cmp::Reverse(b.0));
    for (_, old) in bins.into_iter().skip(KEEP) {
        let _ = fs::remove_file(old);
    }
}

/// The cache for `root`, if it's still valid for this volume history and access level.
pub fn load(root: &Path, device_uuid: &str, has_fda: bool) -> Option<Cached> {
    let cached = decode(&fs::read(file_for(root)?).ok()?)?;
    let h = &cached.header;
    (h.root == root && h.device_uuid == device_uuid && h.has_fda == has_fda).then_some(cached)
}

/// Write via a temp file and `rename`, so a crash never leaves a half-written cache.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let out = self.buf.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(out)
    }

    fn byte(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn varint(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            // The tenth byte holds bit 63 only; anything more doesn't fit a u64.
            if shift == 63 && b > 1 {
                return None;
            }
            v |= ((b & 0x7f) as u64).checked_shl(shift)?;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }

    fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = usize::try_from(self.varint()?).ok()?;
        self.take(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(bytes: &[u8]) -> Option<u64> {
        let mut r = Reader { buf: bytes, pos: 0 };
        r.varint().filter(|_| r.pos == bytes.len())
    }

    #[test]
    fn varint_rejects_overflow() {
        let mut max = Vec::new();
        put_varint(&mut max, u64::MAX);
        assert_eq!(max, [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]);
        assert_eq!(varint(&max), Some(u64::MAX));
        let tenth = |b: u8| [0x80; 9].into_iter().chain([b]).collect::<Vec<_>>();
        assert_eq!(varint(&tenth(0x01)), Some(1 << 63));
        assert_eq!(varint(&tenth(0x02)), None, "bit 64");
        assert_eq!(varint(&tenth(0x7f)), None);
        assert_eq!(varint(&tenth(0x81)), None, "an eleventh byte");
        assert_eq!(varint(&[0x80; 3]), None, "truncated");
    }

    /// A one-folder cache with a valid checksum around `own`, the folder's raw varint bytes.
    fn one_dir(own: &[u8]) -> Vec<u8> {
        let mut fields = Vec::new();
        put_bytes(&mut fields, b"/x");
        put_bytes(&mut fields, b"uuid");
        put_varint(&mut fields, 1);
        put_varint(&mut fields, 2);
        fields.push(1);
        put_varint(&mut fields, 1);
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&VERSION.to_le_bytes());
        put_bytes(&mut out, &fields);
        out.push(TAG_DIR);
        put_bytes(&mut out, b"x");
        out.extend_from_slice(own);
        for v in [0, 3, 4] {
            put_varint(&mut out, v);
        }
        let sum = fnv1a(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        out
    }

    #[test]
    fn decode_rejects_overflowing_varint() {
        let cached = decode(&one_dir(&[0x05])).unwrap();
        assert_eq!((cached.nodes[0].size, cached.inos[0], cached.stamps[0]), (5, 3, 4));
        let mut overflow = vec![0x80; 9];
        overflow.push(0x02);
        assert!(decode(&one_dir(&overflow)).is_none());
    }
}
