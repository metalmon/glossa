use encoding_rs::{Encoding, UTF_16BE, UTF_16LE, UTF_8};

/// Detect the charset of a text file from a prefix (first ~64 KiB). Returns None if the bytes look
/// binary (a NUL byte, or >10% C0 control bytes other than tab/newline/carriage-return).
pub fn detect(prefix: &[u8]) -> Option<&'static Encoding> {
    if prefix.is_empty() {
        return Some(UTF_8);
    }
    // BOM sniffing.
    if prefix.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Some(UTF_8);
    }
    if prefix.starts_with(&[0xFF, 0xFE]) {
        return Some(UTF_16LE);
    }
    if prefix.starts_with(&[0xFE, 0xFF]) {
        return Some(UTF_16BE);
    }
    // Binary sniff (only meaningful for non-UTF-16 content).
    let mut control = 0usize;
    for &b in prefix {
        if b == 0 {
            return None;
        }
        if b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r' {
            control += 1;
        }
    }
    if control * 10 > prefix.len() {
        return None;
    }
    // Strict UTF-8 over the prefix wins; else guess (cp1251 / koi8-r / latin-*).
    if std::str::from_utf8(prefix).is_ok() {
        return Some(UTF_8);
    }
    let mut det = chardetng::EncodingDetector::new();
    det.feed(prefix, true);
    Some(det.guess(None, true))
}

/// Decode a whole buffer to UTF-8 text, or None if it looks binary.
pub fn decode_all(bytes: &[u8]) -> Option<String> {
    let enc = detect(bytes)?;
    let (text, _, _) = enc.decode(bytes);
    Some(text.into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_utf8_and_bom() {
        assert_eq!(decode_all("héllo".as_bytes()).unwrap(), "héllo");
        let mut bom = vec![0xEF, 0xBB, 0xBF];
        bom.extend_from_slice("hi".as_bytes());
        assert_eq!(decode_all(&bom).unwrap(), "hi");
    }

    #[test]
    fn decodes_utf16_le_and_be() {
        let le = [0xFF, 0xFE, b'h', 0x00, b'i', 0x00];
        assert_eq!(decode_all(&le).unwrap(), "hi");
        let be = [0xFE, 0xFF, 0x00, b'h', 0x00, b'i'];
        assert_eq!(decode_all(&be).unwrap(), "hi");
    }

    #[test]
    fn decodes_windows_1251_russian() {
        // "Привет" in Windows-1251.
        let cp1251 = [0xCF, 0xF0, 0xE8, 0xE2, 0xE5, 0xF2];
        assert_eq!(decode_all(&cp1251).unwrap(), "Привет");
    }

    #[test]
    fn binary_is_none() {
        assert!(decode_all(&[0x89, b'P', b'N', b'G', 0x00, 0x1A]).is_none()); // NUL present
        let mut ctrl = vec![0x01u8; 100];
        ctrl[50] = b'x';
        assert!(decode_all(&ctrl).is_none()); // mostly control
    }
}
