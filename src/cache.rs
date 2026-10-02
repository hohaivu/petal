//! The scan cache: a full tree plus what an incremental rescan needs, in a compact
//! little-endian binary with LEB128 varints and an FNV-1a checksum trailer.
//!
//! Nodes are stored in pre-order from the root (skipping `Kind::Other` slices), so every
//! subtree decodes to a contiguous index range. Folder sizes and item counts are
//! recomputed on decode; only a folder's own allocation is stored.

// ponytail: only the tests use this until wave 2 wires up load/save.
#![cfg_attr(not(test), allow(dead_code))]

use std::fs;
use std::io;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use crate::scan::{Kind, Mark, Node, ScanMeta, Tree};

const MAGIC: &[u8; 8] = b"PETALSC1";
/// Bump whenever scan semantics change.
const VERSION: u32 = 1;
const TAG_FILE: u8 = 0;
const TAG_DIR: u8 = 1;
const TAG_MARKED: u8 = 0x80;
const FLAG_DATALESS: u8 = 1;
const FLAG_HARDLINKS: u8 = 2;

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
    /// Sorted by node index.
    pub marks: Vec<(usize, Mark)>,
    pub sub_end: Vec<usize>,
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
        let lo = self.marks.partition_point(|m| m.0 < ix);
        let hi = self.marks.partition_point(|m| m.0 < end);
        &self.marks[lo..hi]
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
        let tag = if node.kind == Kind::File { TAG_FILE } else { TAG_DIR };
        out.push(tag | if mark.is_some() { TAG_MARKED } else { 0 });
        put_bytes(&mut out, node.name.as_bytes());
        if node.kind == Kind::File {
            put_varint(&mut out, node.size);
        } else {
            // Own allocation: all children count here, including any volume slices.
            let own = node.size.saturating_sub(node.children.iter().map(|&c| tree.nodes[c].size).sum());
            put_varint(&mut out, own);
            put_varint(&mut out, node.children.iter().filter(|c| real(c)).count() as u64);
            put_varint(&mut out, meta.inos.get(ix).copied().unwrap_or(0));
        }
        if let Some(mark) = mark {
            put_varint(&mut out, mark.errors);
            out.push(if mark.dataless { FLAG_DATALESS } else { 0 } | if mark.hardlinks { FLAG_HARDLINKS } else { 0 });
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
    let mut sub_end = vec![0; count];
    let mut marks = Vec::new();
    let mut stack: Vec<Frame> = Vec::new();
    while nodes.len() < count {
        let ix = nodes.len();
        if ix > 0 && stack.is_empty() {
            return None;
        }
        let tag = r.byte()?;
        let kind = match tag & !TAG_MARKED {
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
        nodes.push(Node { name: name.to_owned().into(), size, kind, parent, children: Vec::new(), items });
        if kind == Kind::Dir {
            let (own, left, ino) = (r.varint()?, r.varint()?, r.varint()?);
            inos.push(ino);
            stack.push(Frame { ix, left, own });
        } else {
            inos.push(ino);
            sub_end[ix] = ix + 1;
        }
        if tag & TAG_MARKED != 0 {
            let (errors, flags) = (r.varint()?, r.byte()?);
            marks.push((ix, Mark { errors, dataless: flags & FLAG_DATALESS != 0, hardlinks: flags & FLAG_HARDLINKS != 0 }));
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
    Some(Cached { header, nodes, inos, marks, sub_end })
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
