use encoding_rs::{Encoding, UTF_16BE, UTF_16LE, UTF_8, WINDOWS_1252};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyMsoExtract {
    pub kind: &'static str,
    pub text: String,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextBlock {
    pub section: String,
    pub text: String,
    pub source_offset: Option<usize>,
}

const OLE_MAGIC: &[u8] = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1";
const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;

pub fn extract_legacy_mso_text(input: &[u8]) -> Option<LegacyMsoExtract> {
    if !input.starts_with(OLE_MAGIC) {
        return None;
    }
    let (capped_input, capped) = cap_input_slice(input, MAX_INPUT_BYTES);
    let kind = detect_kind(capped_input);
    let mut warnings = Vec::new();
    let probe = String::from_utf8_lossy(capped_input).to_ascii_lowercase();
    if probe.contains("vba") || probe.contains("macros") {
        warnings.push("ole-macro-present".to_string());
    }
    if probe.contains("ole10native") || probe.contains("embedded object") {
        warnings.push("ole-embedded-object".to_string());
    }
    let streams = parse_ole_streams(capped_input);
    if streams.is_none() {
        warnings.push("Corrupted".to_string());
    }
    if kind == "ole-unknown" {
        warnings.push("Unsupported".to_string());
    }
    let (text, structured_ok) = extract_text_by_kind(capped_input, kind, streams.as_ref());
    if (matches!(kind, "doc" | "xls" | "ppt") && !structured_ok && !text.trim().is_empty()) || capped {
        warnings.push("PartialExtracted".to_string());
    }
    Some(LegacyMsoExtract {
        kind,
        text,
        warnings,
    })
}

fn cap_input_slice<'a>(input: &'a [u8], max_len: usize) -> (&'a [u8], bool) {
    if input.len() > max_len {
        (&input[..max_len], true)
    } else {
        (input, false)
    }
}

pub fn build_text_blocks(kind: &str, text: &str) -> Vec<TextBlock> {
    let mut blocks = Vec::new();
    if text.trim().is_empty() {
        return blocks;
    }
    if kind == "xls" || kind == "ppt" {
        let mut current_section = String::new();
        let mut current_lines: Vec<String> = Vec::new();
        let mut offset = 0usize;
        for line in text.lines() {
            let is_header = line.starts_with("Sheet ") || line.starts_with("Slide ");
            if is_header {
                if !current_lines.is_empty() {
                    blocks.push(TextBlock {
                        section: if current_section.is_empty() {
                            "Section".to_string()
                        } else {
                            current_section.clone()
                        },
                        text: current_lines.join("\n"),
                        source_offset: Some(offset),
                    });
                    current_lines.clear();
                }
                current_section = line.trim().to_string();
            } else if !line.trim().is_empty() {
                current_lines.push(line.trim().to_string());
            }
            offset += line.len() + 1;
        }
        if !current_lines.is_empty() {
            blocks.push(TextBlock {
                section: if current_section.is_empty() {
                    "Section".to_string()
                } else {
                    current_section
                },
                text: current_lines.join("\n"),
                source_offset: Some(offset),
            });
        }
        if !blocks.is_empty() {
            return blocks;
        }
    }
    blocks.push(TextBlock {
        section: kind.to_string(),
        text: text.trim().to_string(),
        source_offset: Some(0),
    });
    blocks
}

fn detect_kind(input: &[u8]) -> &'static str {
    let lower = String::from_utf8_lossy(input).to_ascii_lowercase();
    if lower.contains("worddocument") {
        "doc"
    } else if lower.contains("workbook") || lower.contains("book") {
        "xls"
    } else if lower.contains("powerpoint document") {
        "ppt"
    } else if lower.contains("ownerfile") || lower.contains("~$") {
        "msoffice-ownerfile"
    } else {
        "ole-unknown"
    }
}

fn extract_text_by_kind(input: &[u8], kind: &str, streams: Option<&Vec<OleStream>>) -> (String, bool) {
    if kind == "doc"
        && let Some(streams) = streams
        && let Some(text) = extract_doc_text_structured(streams)
        && !text.trim().is_empty()
        && is_high_confidence_text(&text, kind)
    {
        return (text, true);
    }
    if kind == "xls"
        && let Some(streams) = streams
        && let Some(text) = extract_xls_text_structured(streams)
        && !text.trim().is_empty()
        && is_high_confidence_text(&text, kind)
    {
        return (text, true);
    }
    if kind == "ppt"
        && let Some(streams) = streams
        && let Some(text) = extract_ppt_text_structured(streams)
        && !text.trim().is_empty()
        && is_high_confidence_text(&text, kind)
    {
        return (text, true);
    }
    let scan_bytes = select_scan_bytes(input, kind, streams);
    let mut lines = Vec::new();
    let min_ascii = if kind == "xls" { 2 } else { 3 };
    for s in extract_utf16le_strings(scan_bytes, 2, 48 * 1024) {
        lines.push(s);
    }
    for s in extract_ascii_strings(scan_bytes, min_ascii, 48 * 1024) {
        lines.push(s);
    }
    for s in extract_latin1_strings(scan_bytes, 4, 48 * 1024) {
        lines.push(repair_utf8_mojibake(&s).unwrap_or(s));
    }

    let mut out = Vec::new();
    for raw in lines {
        let normalized = normalize_line(&raw);
        if normalized.is_empty() {
            continue;
        }
        if !looks_like_content_line(&normalized, kind) {
            continue;
        }
        if out.last() == Some(&normalized) {
            continue;
        }
        out.push(normalized);
        if out.len() >= 800 {
            break;
        }
    }
    let mut cleaned = trim_leading_noise(out);
    cleaned = truncate_doc_tail_noise(cleaned, kind);
    cleaned = trim_trailing_noise(cleaned, kind);
    (cleaned.join("\n"), false)
}

fn is_high_confidence_text(text: &str, kind: &str) -> bool {
    score_text_quality(text, kind) >= 45
}

fn score_text_quality(text: &str, kind: &str) -> u8 {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return 0;
    }
    let lines: Vec<&str> = trimmed.lines().collect();
    let mut score: i32 = 40;
    if lines.len() >= 2 {
        score += 15;
    }
    let mut meaningful = 0usize;
    for line in &lines {
        let t = line.trim();
        if t.len() >= 2 && looks_like_content_line(t, kind) {
            meaningful += 1;
        }
    }
    if !lines.is_empty() {
        score += ((meaningful * 40) / lines.len()) as i32;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.contains("root entry")
        || lower.contains("summaryinformation")
        || lower.contains("documentsummaryinformation")
    {
        score -= 35;
    }
    score.clamp(0, 100) as u8
}

fn extract_ppt_text_structured(streams: &[OleStream]) -> Option<String> {
    let doc = streams
        .iter()
        .find(|s| s.name.eq_ignore_ascii_case("PowerPoint Document"))?
        .data
        .as_slice();
    let records = walk_ppt_records(doc, 16, 8 * 1024 * 1024)?;
    if records.is_empty() {
        return None;
    }
    let slide_blocks = group_ppt_slide_text(&records);
    let mut out = Vec::new();
    for (idx, lines) in slide_blocks.into_iter().enumerate() {
        if lines.is_empty() {
            continue;
        }
        out.push(format!("Slide {}", idx + 1));
        out.extend(lines);
    }
    if out.is_empty() {
        for rec in records.iter().take(10) {
            out.push(format!(
                "type=0x{:04X} depth={} len={}",
                rec.rec_type, rec.depth, rec.rec_len
            ));
        }
    }
    Some(out.join("\n"))
}

fn group_ppt_slide_text(records: &[PptRecord]) -> Vec<Vec<String>> {
    let mut slides: Vec<Vec<String>> = Vec::new();
    let mut current = Vec::<String>::new();
    let mut seen_slide = false;
    for rec in records {
        if is_ppt_slide_container(rec.rec_type) {
            if seen_slide && !current.is_empty() {
                slides.push(std::mem::take(&mut current));
            }
            seen_slide = true;
            continue;
        }
        if let Some(text) = decode_ppt_text_atom(rec.rec_type, &rec.data) {
            let line = normalize_line(&text);
            if line.is_empty() {
                continue;
            }
            if current.last() == Some(&line) {
                continue;
            }
            current.push(line);
        }
    }
    if !current.is_empty() {
        slides.push(current);
    }
    slides
}

fn is_ppt_slide_container(rec_type: u16) -> bool {
    // Common PPT binary container type ids for slide-like scopes.
    matches!(rec_type, 0x03EE | 0x03F3)
}

#[derive(Debug, Clone)]
struct PptRecord {
    rec_type: u16,
    rec_len: u32,
    depth: usize,
    data: Vec<u8>,
}

fn walk_ppt_records(input: &[u8], max_depth: usize, max_record_len: usize) -> Option<Vec<PptRecord>> {
    let mut out = Vec::new();
    walk_ppt_records_inner(input, 0, max_depth, max_record_len, &mut out)?;
    Some(out)
}

fn walk_ppt_records_inner(
    input: &[u8],
    depth: usize,
    max_depth: usize,
    max_record_len: usize,
    out: &mut Vec<PptRecord>,
) -> Option<()> {
    if depth > max_depth {
        return None;
    }
    let mut pos = 0usize;
    while pos + 8 <= input.len() {
        let rec_ver_inst = le_u16(input, pos)?;
        let rec_ver = (rec_ver_inst & 0x000F) as u8;
        let rec_type = le_u16(input, pos + 2)?;
        let rec_len = le_u32(input, pos + 4)? as usize;
        if rec_len > max_record_len {
            return None;
        }
        let data_start = pos + 8;
        let data_end = data_start.saturating_add(rec_len);
        if data_end > input.len() {
            break;
        }
        out.push(PptRecord {
            rec_type,
            rec_len: rec_len as u32,
            depth,
            data: input[data_start..data_end].to_vec(),
        });
        if rec_ver == 0x0F {
            walk_ppt_records_inner(&input[data_start..data_end], depth + 1, max_depth, max_record_len, out)?;
        }
        pos = data_end;
    }
    Some(())
}

fn decode_ppt_text_atom(rec_type: u16, data: &[u8]) -> Option<String> {
    match rec_type {
        // TextBytesAtom
        0x0FA8 => Some(decode_bytes_with_strategy(data, None)),
        // TextCharsAtom
        0x0FA0 => {
            if data.len() < 2 {
                return None;
            }
            let mut units = Vec::with_capacity(data.len() / 2);
            for ch in data.chunks_exact(2) {
                units.push(u16::from_le_bytes([ch[0], ch[1]]));
            }
            Some(String::from_utf16_lossy(&units))
        }
        // CString
        0x0FBA => {
            if data.is_empty() {
                return None;
            }
            if data.len() % 2 == 0 {
                let mut units = Vec::with_capacity(data.len() / 2);
                for ch in data.chunks_exact(2) {
                    units.push(u16::from_le_bytes([ch[0], ch[1]]));
                }
                Some(String::from_utf16_lossy(&units))
            } else {
                Some(decode_bytes_with_strategy(data, None))
            }
        }
        _ => None,
    }
}

fn decode_bytes_with_strategy(bytes: &[u8], preferred: Option<&'static Encoding>) -> String {
    if bytes.is_empty() {
        return String::new();
    }
    if let Some((enc, skip)) = detect_bom_encoding(bytes) {
        let (decoded, _, _) = enc.decode(&bytes[skip..]);
        return decoded.into_owned();
    }
    if let Some(enc) = preferred {
        let (decoded, _, had_errors) = enc.decode(bytes);
        if !had_errors {
            let s = decoded.into_owned();
            if looks_like_mojibake(&s) && std::str::from_utf8(bytes).is_ok() {
                return String::from_utf8_lossy(bytes).to_string();
            }
            return s;
        }
    }
    if std::str::from_utf8(bytes).is_ok() {
        return String::from_utf8_lossy(bytes).to_string();
    }
    if looks_like_utf16le(bytes) {
        let (decoded, _, _) = UTF_16LE.decode(bytes);
        return decoded.into_owned();
    }
    let (decoded, _, _) = WINDOWS_1252.decode(bytes);
    decoded.into_owned()
}

fn looks_like_mojibake(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let suspicious = s
        .chars()
        .filter(|c| matches!(*c, 'Ã' | 'Â' | 'â' | '€' | '™' | 'œ' | 'ž' | '¤' | 'ä' | 'å' | 'æ'))
        .count();
    suspicious * 10 >= s.chars().count()
}

fn detect_bom_encoding(bytes: &[u8]) -> Option<(&'static Encoding, usize)> {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Some((UTF_8, 3));
    }
    if bytes.starts_with(&[0xFF, 0xFE]) {
        return Some((UTF_16LE, 2));
    }
    if bytes.starts_with(&[0xFE, 0xFF]) {
        return Some((UTF_16BE, 2));
    }
    None
}

fn looks_like_utf16le(bytes: &[u8]) -> bool {
    if bytes.len() < 4 || bytes.len() % 2 != 0 {
        return false;
    }
    let mut pairs = 0usize;
    let mut zero_high = 0usize;
    for pair in bytes.chunks_exact(2) {
        pairs += 1;
        if pair[1] == 0 {
            zero_high += 1;
        }
    }
    zero_high * 100 / pairs >= 35
}

fn extract_xls_text_structured(streams: &[OleStream]) -> Option<String> {
    let workbook = streams
        .iter()
        .find(|s| s.name.eq_ignore_ascii_case("Workbook") || s.name.eq_ignore_ascii_case("Book"))?
        .data
        .as_slice();
    let records = parse_biff_records(workbook)?;
    let mut bof_count = 0usize;
    let mut eof_count = 0usize;
    let mut sheets = Vec::new();
    let mut sst_preview = Vec::new();
    let mut sst_strings = Vec::new();
    let mut preferred_encoding = None::<&'static Encoding>;
    for rec in &records {
        match rec.id {
            0x0809 => bof_count += 1, // BOF
            0x000A => eof_count += 1, // EOF
            0x0042 => {
                preferred_encoding = parse_xls_codepage(rec.data).or(preferred_encoding);
            }
            0x0085 => {
                if let Some(name) = parse_boundsheet_name(rec.data)
                    && !name.trim().is_empty()
                {
                    sheets.push(name);
                }
            }
            0x00FC => {
                if let Some(strings) = parse_sst_strings_with_continue(workbook, rec.offset, preferred_encoding) {
                    sst_strings = strings.clone();
                    sst_preview = strings.into_iter().take(3).collect();
                }
            }
            _ => {}
        }
    }
    if bof_count == 0 || eof_count == 0 || sheets.is_empty() {
        return None;
    }
    let mut lines = sheets
        .iter()
        .enumerate()
        .map(|(i, name)| format!("Sheet{}: {}", i + 1, name))
        .collect::<Vec<_>>();
    for s in sst_preview {
        if !s.trim().is_empty() {
            lines.push(s);
        }
    }
    for block in format_sheet_blocks(parse_xls_cell_values(
        &records,
        &sheets,
        &sst_strings,
        preferred_encoding,
    )) {
        if !block.trim().is_empty() {
            lines.push(block);
        }
    }
    Some(lines.join("\n"))
}

#[derive(Debug, Clone, Copy)]
struct BiffRecord<'a> {
    id: u16,
    data: &'a [u8],
    offset: usize,
}

fn parse_biff_records(input: &[u8]) -> Option<Vec<BiffRecord<'_>>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= input.len() {
        let id = le_u16(input, pos)?;
        let len = le_u16(input, pos + 2)? as usize;
        let start = pos + 4;
        let end = start.saturating_add(len);
        if end > input.len() {
            break;
        }
        out.push(BiffRecord {
            id,
            data: &input[start..end],
            offset: pos,
        });
        pos = end;
    }
    if out.is_empty() { None } else { Some(out) }
}

fn parse_boundsheet_name(data: &[u8]) -> Option<String> {
    if data.len() < 8 {
        return None;
    }
    let cch = data[6] as usize;
    let flags = data[7];
    let high_byte = (flags & 0x01) != 0;
    let name_data = data.get(8..)?;
    if high_byte {
        let byte_len = cch.saturating_mul(2);
        let bytes = name_data.get(0..byte_len)?;
        let mut units = Vec::with_capacity(cch);
        for ch in bytes.chunks_exact(2) {
            units.push(u16::from_le_bytes([ch[0], ch[1]]));
        }
        Some(String::from_utf16_lossy(&units))
    } else {
        let bytes = name_data.get(0..cch)?;
        Some(String::from_utf8_lossy(bytes).to_string())
    }
}

fn parse_sst_strings_with_continue(
    workbook: &[u8],
    sst_record_offset: usize,
    preferred: Option<&'static Encoding>,
) -> Option<Vec<String>> {
    if sst_record_offset + 4 > workbook.len() {
        return None;
    }
    let sst_len = le_u16(workbook, sst_record_offset + 2)? as usize;
    let sst_data_start = sst_record_offset + 4;
    let sst_data_end = sst_data_start.saturating_add(sst_len);
    if sst_data_end > workbook.len() || sst_len < 8 {
        return None;
    }
    let cst_unique = le_u32(workbook, sst_data_start + 4)? as usize;
    let mut bytes = workbook[sst_data_start + 8..sst_data_end].to_vec();
    let mut next = sst_data_end;
    while next + 4 <= workbook.len() {
        let id = le_u16(workbook, next)?;
        let len = le_u16(workbook, next + 2)? as usize;
        let data_start = next + 4;
        let data_end = data_start.saturating_add(len);
        if data_end > workbook.len() || id != 0x003C {
            break;
        }
        bytes.extend_from_slice(&workbook[data_start..data_end]);
        next = data_end;
    }
    parse_sst_plain_strings(&bytes, cst_unique, preferred)
}

fn parse_sst_plain_strings(
    bytes: &[u8],
    limit: usize,
    preferred: Option<&'static Encoding>,
) -> Option<Vec<String>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 3 <= bytes.len() && out.len() < limit {
        let cch = le_u16(bytes, pos)? as usize;
        let flags = bytes[pos + 2];
        pos += 3;
        let is_16 = (flags & 0x01) != 0;
        if (flags & 0x08) != 0 {
            if pos + 2 > bytes.len() {
                break;
            }
            let run_count = le_u16(bytes, pos)? as usize;
            pos += 2;
            let skip = run_count.saturating_mul(4);
            if pos + skip > bytes.len() {
                break;
            }
            pos += skip;
        }
        if (flags & 0x04) != 0 {
            if pos + 4 > bytes.len() {
                break;
            }
            pos += 4;
        }
        if is_16 {
            let byte_len = cch.saturating_mul(2);
            if pos + byte_len > bytes.len() {
                break;
            }
            let mut units = Vec::with_capacity(cch);
            for ch in bytes[pos..pos + byte_len].chunks_exact(2) {
                units.push(u16::from_le_bytes([ch[0], ch[1]]));
            }
            out.push(String::from_utf16_lossy(&units));
            pos += byte_len;
        } else {
            if pos + cch > bytes.len() {
                break;
            }
            out.push(decode_bytes_with_strategy(&bytes[pos..pos + cch], preferred));
            pos += cch;
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

#[derive(Debug, Clone)]
struct XlsCell {
    sheet: String,
    row: u16,
    col: u16,
    value: String,
}

fn parse_xls_cell_values(
    records: &[BiffRecord<'_>],
    sheet_names: &[String],
    sst: &[String],
    preferred: Option<&'static Encoding>,
) -> Vec<XlsCell> {
    let mut out = Vec::new();
    let mut current_sheet = None::<usize>;
    let mut i = 0usize;
    while i < records.len() {
        let rec = &records[i];
        if rec.id == 0x0809 {
            if rec.data.len() >= 4 {
                let bof_type = u16::from_le_bytes([rec.data[2], rec.data[3]]);
                if bof_type == 0x0010 {
                    let next = current_sheet.map(|n| n + 1).unwrap_or(0);
                    current_sheet = Some(next);
                }
            }
            i += 1;
            continue;
        }
        let Some(sheet_idx) = current_sheet else {
            i += 1;
            continue;
        };
        let sheet_name = sheet_names
            .get(sheet_idx)
            .map(String::as_str)
            .unwrap_or("Sheet");
        if rec.id == 0x0006
            && let Some((row, col, cached)) = parse_formula_cached_record(rec.data)
        {
            let value = match cached {
                FormulaCachedValue::Number(v) => Some(v),
                FormulaCachedValue::Boolean(v) => Some(v),
                FormulaCachedValue::Error(v) => Some(v),
                FormulaCachedValue::StringPending => {
                    if i + 1 < records.len() && records[i + 1].id == 0x0207 {
                        i += 1;
                        parse_string_record(records[i].data)
                    } else {
                        None
                    }
                }
                FormulaCachedValue::Blank => None,
            };
            if let Some(value) = value {
                out.push(XlsCell {
                    sheet: sheet_name.to_string(),
                    row,
                    col,
                    value,
                });
            }
            i += 1;
            continue;
        }
        if let Some(cell) = parse_cell_record_line(rec, sheet_name, sst, preferred) {
            out.push(cell);
        }
        if out.len() >= 200 {
            break;
        }
        i += 1;
    }
    out
}

fn parse_cell_record_line(
    rec: &BiffRecord<'_>,
    sheet_name: &str,
    sst: &[String],
    preferred: Option<&'static Encoding>,
) -> Option<XlsCell> {
    match rec.id {
        0x0204 => parse_label_record(rec.data, preferred).map(|(r, c, v)| XlsCell {
            sheet: sheet_name.to_string(),
            row: r,
            col: c,
            value: v,
        }),
        0x00FD => parse_labelsst_record(rec.data, sst).map(|(r, c, v)| XlsCell {
            sheet: sheet_name.to_string(),
            row: r,
            col: c,
            value: v,
        }),
        0x0203 => parse_number_record(rec.data).map(|(r, c, v)| XlsCell {
            sheet: sheet_name.to_string(),
            row: r,
            col: c,
            value: v,
        }),
        0x027E => parse_rk_record(rec.data).map(|(r, c, v)| XlsCell {
            sheet: sheet_name.to_string(),
            row: r,
            col: c,
            value: v,
        }),
        _ => None,
    }
}

fn format_sheet_blocks(mut cells: Vec<XlsCell>) -> Vec<String> {
    if cells.is_empty() {
        return Vec::new();
    }
    cells.sort_by(|a, b| {
        a.sheet
            .cmp(&b.sheet)
            .then(a.row.cmp(&b.row))
            .then(a.col.cmp(&b.col))
    });
    let mut blocks = Vec::new();
    let mut current_sheet = String::new();
    let mut current_lines = Vec::new();
    for cell in cells {
        if current_sheet != cell.sheet {
            if !current_lines.is_empty() {
                blocks.push(format!("{current_sheet}\n{}", current_lines.join("\n")));
                current_lines.clear();
            }
            current_sheet = cell.sheet.clone();
        }
        current_lines.push(format!("R{}C{}={}", cell.row + 1, cell.col + 1, cell.value));
    }
    if !current_lines.is_empty() {
        blocks.push(format!("{current_sheet}\n{}", current_lines.join("\n")));
    }
    blocks
}

fn parse_label_record(data: &[u8], preferred: Option<&'static Encoding>) -> Option<(u16, u16, String)> {
    if data.len() < 8 {
        return None;
    }
    let row = le_u16(data, 0)?;
    let col = le_u16(data, 2)?;
    let cch = le_u16(data, 6)? as usize;
    let bytes = data.get(8..8 + cch)?;
    Some((row, col, decode_bytes_with_strategy(bytes, preferred)))
}

fn parse_labelsst_record(data: &[u8], sst: &[String]) -> Option<(u16, u16, String)> {
    if data.len() < 10 {
        return None;
    }
    let row = le_u16(data, 0)?;
    let col = le_u16(data, 2)?;
    let idx = le_u32(data, 6)? as usize;
    let value = sst.get(idx)?.clone();
    Some((row, col, value))
}

fn parse_number_record(data: &[u8]) -> Option<(u16, u16, String)> {
    if data.len() < 14 {
        return None;
    }
    let row = le_u16(data, 0)?;
    let col = le_u16(data, 2)?;
    let n = f64::from_le_bytes(data.get(6..14)?.try_into().ok()?);
    Some((row, col, format_excel_number(n)))
}

fn parse_rk_record(data: &[u8]) -> Option<(u16, u16, String)> {
    if data.len() < 10 {
        return None;
    }
    let row = le_u16(data, 0)?;
    let col = le_u16(data, 2)?;
    let rk = le_u32(data, 6)?;
    Some((row, col, format_excel_number(decode_rk_value(rk))))
}

enum FormulaCachedValue {
    Number(String),
    StringPending,
    Boolean(String),
    Error(String),
    Blank,
}

fn parse_formula_cached_record(data: &[u8]) -> Option<(u16, u16, FormulaCachedValue)> {
    if data.len() < 14 {
        return None;
    }
    let row = le_u16(data, 0)?;
    let col = le_u16(data, 2)?;
    let result = data.get(6..14)?;
    if result[6] == 0xFF && result[7] == 0xFF {
        let value = match result[0] {
            0x00 => FormulaCachedValue::StringPending,
            0x01 => FormulaCachedValue::Boolean(if result[2] == 0 { "FALSE" } else { "TRUE" }.to_string()),
            0x02 => FormulaCachedValue::Error(decode_excel_error(result[2]).to_string()),
            0x03 => FormulaCachedValue::Blank,
            _ => FormulaCachedValue::Blank,
        };
        return Some((row, col, value));
    }
    let n = f64::from_le_bytes(result.try_into().ok()?);
    Some((row, col, FormulaCachedValue::Number(format_excel_number(n))))
}

fn parse_string_record(data: &[u8]) -> Option<String> {
    if data.len() < 3 {
        return None;
    }
    let cch = le_u16(data, 0)? as usize;
    let flags = data[2];
    let mut pos = 3usize;
    if (flags & 0x08) != 0 {
        let run_count = le_u16(data, pos)? as usize;
        pos = pos.saturating_add(2 + run_count.saturating_mul(4));
    }
    if (flags & 0x04) != 0 {
        pos = pos.saturating_add(4);
    }
    let is_16 = (flags & 0x01) != 0;
    if is_16 {
        let byte_len = cch.saturating_mul(2);
        let bytes = data.get(pos..pos + byte_len)?;
        let mut units = Vec::with_capacity(cch);
        for ch in bytes.chunks_exact(2) {
            units.push(u16::from_le_bytes([ch[0], ch[1]]));
        }
        Some(String::from_utf16_lossy(&units))
    } else {
        Some(String::from_utf8_lossy(data.get(pos..pos + cch)?).to_string())
    }
}

fn decode_excel_error(code: u8) -> &'static str {
    match code {
        0x00 => "#NULL!",
        0x07 => "#DIV/0!",
        0x0F => "#VALUE!",
        0x17 => "#REF!",
        0x1D => "#NAME?",
        0x24 => "#NUM!",
        0x2A => "#N/A",
        _ => "#ERROR",
    }
}

fn parse_xls_codepage(data: &[u8]) -> Option<&'static Encoding> {
    if data.len() < 2 {
        return None;
    }
    let cp = le_u16(data, 0)?;
    let label = match cp {
        65001 => "utf-8",
        1200 => "utf-16le",
        1201 => "utf-16be",
        936 => "gbk",
        950 => "big5",
        932 => "shift_jis",
        949 => "euc-kr",
        1250 => "windows-1250",
        1251 => "windows-1251",
        1252 => "windows-1252",
        1253 => "windows-1253",
        1254 => "windows-1254",
        1255 => "windows-1255",
        1256 => "windows-1256",
        1257 => "windows-1257",
        874 => "windows-874",
        _ => return None,
    };
    Encoding::for_label(label.as_bytes())
}

fn format_excel_number(n: f64) -> String {
    if n == 0.0 {
        return "0".to_string();
    }
    if !n.is_finite() {
        return n.to_string();
    }
    if (n.fract()).abs() < f64::EPSILON {
        format!("{:.0}", n)
    } else {
        let mut s = format!("{:.15}", n);
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        s
    }
}

fn decode_rk_value(rk: u32) -> f64 {
    let is_mult_100 = (rk & 0x01) != 0;
    let is_integer = (rk & 0x02) != 0;
    let mut v = if is_integer {
        ((rk as i32) >> 2) as f64
    } else {
        let raw = ((rk & 0xFFFF_FFFCu32) as u64) << 32;
        f64::from_bits(raw)
    };
    if is_mult_100 {
        v /= 100.0;
    }
    v
}

fn extract_doc_text_structured(streams: &[OleStream]) -> Option<String> {
    let fib = parse_doc_fib(
        streams
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case("WordDocument"))?
            .data
            .as_slice(),
    )?;
    let word = streams
        .iter()
        .find(|s| s.name.eq_ignore_ascii_case("WordDocument"))?
        .data
        .as_slice();
    let mut table_candidates: Vec<(&str, &[u8])> = streams
        .iter()
        .filter(|s| s.name.eq_ignore_ascii_case("1Table") || s.name.eq_ignore_ascii_case("0Table"))
        .map(|s| (s.name.as_str(), s.data.as_slice()))
        .collect();
    if table_candidates.is_empty() {
        return None;
    }
    table_candidates.sort_by_key(|(name, _)| {
        if fib.use_1table {
            (!name.eq_ignore_ascii_case("1Table")) as u8
        } else {
            (!name.eq_ignore_ascii_case("0Table")) as u8
        }
    });
    let mut best = String::new();
    for (_, table) in table_candidates {
        if let Some(text) = extract_doc_text_from_table(word, table, fib)
            && text.len() > best.len()
        {
            best = text;
        }
    }
    if best.trim().is_empty() {
        None
    } else {
        Some(best)
    }
}

fn extract_doc_text_from_table(word: &[u8], table: &[u8], fib: DocFib) -> Option<String> {
    let clx = locate_clx(table, fib)?;
    let plc = parse_clx_piece_table(clx)?;
    let mut text = String::new();
    for piece in plc {
        let cp_chars = piece.cp_end.saturating_sub(piece.cp_start) as usize;
        if cp_chars == 0 {
            continue;
        }
        if piece.compressed {
            let fc = piece.fc as usize;
            let end = fc.saturating_add(ansi_piece_byte_len(cp_chars, fib));
            if end > word.len() {
                continue;
            }
            text.push_str(&decode_ansi_piece(&word[fc..end], fib));
        } else {
            let fc = piece.fc as usize;
            let byte_len = cp_chars.saturating_mul(2);
            let end = fc.saturating_add(byte_len);
            if end > word.len() {
                continue;
            }
            let mut units = Vec::with_capacity(cp_chars);
            for ch in word[fc..end].chunks_exact(2) {
                units.push(u16::from_le_bytes([ch[0], ch[1]]));
            }
            text.push_str(&String::from_utf16_lossy(&units));
        }
    }
    Some(clean_doc_text(&text))
}

fn locate_clx(table: &[u8], fib: DocFib) -> Option<&[u8]> {
    if fib.lcb_clx >= 8 {
        let start = fib.fc_clx as usize;
        let end = start.saturating_add(fib.lcb_clx as usize);
        if end <= table.len() {
            return table.get(start..end);
        }
    }
    scan_for_clx(table)
}

#[derive(Debug, Clone, Copy)]
struct DocFib {
    use_1table: bool,
    fc_clx: u32,
    lcb_clx: u32,
    lid: u16,
    chs: u16,
    chs_tables: u16,
}

fn parse_doc_fib(word: &[u8]) -> Option<DocFib> {
    if word.len() < 0x1AA {
        return None;
    }
    let w_ident = le_u16(word, 0x00)?;
    if w_ident != 0xA5EC {
        return None;
    }
    let n_fib = le_u16(word, 0x02)?;
    // Accept common WW8-era FIB versions.
    if n_fib < 101 {
        return None;
    }
    let flags = le_u16(word, 0x0A)?;
    let use_1table = (flags & 0x0200) != 0;
    let lid = le_u16(word, 0x06)?;
    let chs = le_u16(word, 0x12)?;
    let chs_tables = le_u16(word, 0x14)?;
    let fc_clx = le_u32(word, 0x1A2)?;
    let lcb_clx = le_u32(word, 0x1A6)?;
    if lcb_clx == 0 {
        return None;
    }
    Some(DocFib {
        use_1table,
        fc_clx,
        lcb_clx,
        lid,
        chs,
        chs_tables,
    })
}

fn decode_ansi_piece(bytes: &[u8], fib: DocFib) -> String {
    let fallback_label = std::env::var("VECTRAPARSE_DOC_ANSI_FALLBACK")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "windows-1252".to_string());
    let encoding = select_ansi_encoding(fib)
        .or_else(|| Encoding::for_label(fallback_label.as_bytes()))
        .unwrap_or(encoding_rs::WINDOWS_1252);
    let (decoded, _, _) = encoding.decode(bytes);
    decoded.into_owned()
}

fn ansi_piece_byte_len(cp_chars: usize, fib: DocFib) -> usize {
    if is_dbcs_ansi(fib) {
        cp_chars.saturating_mul(2)
    } else {
        cp_chars
    }
}

fn is_dbcs_ansi(fib: DocFib) -> bool {
    matches!(fib.chs_tables, 128 | 129 | 134 | 136)
        || matches!(fib.chs, 128 | 129 | 134 | 136)
        || matches!(fib.lid, 0x0411 | 0x0412 | 0x0404 | 0x0C04 | 0x1404 | 0x0804 | 0x1004)
}

fn select_ansi_encoding(fib: DocFib) -> Option<&'static Encoding> {
    charset_to_encoding(fib.chs_tables)
        .or_else(|| charset_to_encoding(fib.chs))
        .or_else(|| lcid_to_encoding(fib.lid))
}

fn charset_to_encoding(chs: u16) -> Option<&'static Encoding> {
    let label = match chs {
        0 | 1 => "windows-1252",
        128 => "shift_jis",
        129 => "euc-kr",
        134 => "gbk",
        136 => "big5",
        161 => "windows-1253",
        162 => "windows-1254",
        177 => "windows-1255",
        178 => "windows-1256",
        186 => "windows-1257",
        204 => "windows-1251",
        222 => "windows-874",
        238 => "windows-1250",
        _ => return None,
    };
    Encoding::for_label(label.as_bytes())
}

fn lcid_to_encoding(lid: u16) -> Option<&'static Encoding> {
    let lang = lid & 0x03FF;
    let label = match lid {
        0x0404 | 0x0C04 | 0x1404 => "big5",
        0x0804 | 0x1004 => "gbk",
        0x0411 => "shift_jis",
        0x0412 => "euc-kr",
        _ => match lang {
            0x19 => "windows-1251", // Russian
            0x08 => "windows-1253", // Greek
            0x1F => "windows-1254", // Turkish
            0x0D => "windows-1255", // Hebrew
            0x01 => "windows-1256", // Arabic
            0x15 => "windows-1250", // Polish
            0x09 => "windows-1252", // English
            _ => return None,
        },
    };
    Encoding::for_label(label.as_bytes())
}

fn scan_for_clx(table: &[u8]) -> Option<&[u8]> {
    let limit = table.len().saturating_sub(8);
    for i in 0..limit {
        if table[i] != 0x02 {
            continue;
        }
        let lcb = le_u32(table, i + 1)? as usize;
        if lcb < 16 || (lcb - 4) % 12 != 0 {
            continue;
        }
        let end = i.saturating_add(5).saturating_add(lcb);
        if end <= table.len() {
            return table.get(i..end);
        }
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct PlcPiece {
    cp_start: u32,
    cp_end: u32,
    fc: u32,
    compressed: bool,
}

fn parse_clx_piece_table(clx: &[u8]) -> Option<Vec<PlcPiece>> {
    let mut pos = 0usize;
    while pos + 5 <= clx.len() {
        let kind = clx[pos];
        if kind == 0x01 {
            let cb = le_u16(clx, pos + 1)? as usize;
            pos = pos.saturating_add(3).saturating_add(cb);
            continue;
        }
        if kind == 0x02 {
            let lcb = le_u32(clx, pos + 1)? as usize;
            let start = pos + 5;
            let end = start.saturating_add(lcb);
            if end > clx.len() || lcb < 16 || (lcb - 4) % 12 != 0 {
                return None;
            }
            let plc = &clx[start..end];
            let n = (lcb - 4) / 12;
            let cp_bytes = 4 * (n + 1);
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let cp_start = le_u32(plc, i * 4)?;
                let cp_end = le_u32(plc, (i + 1) * 4)?;
                let pcd_off = cp_bytes + i * 8;
                let fc_raw = le_u32(plc, pcd_off + 2)?;
                let compressed = (fc_raw & 0x4000_0000) != 0;
                let fc = fc_raw & 0x3FFF_FFFF;
                let fc = if compressed { fc / 2 } else { fc };
                out.push(PlcPiece {
                    cp_start,
                    cp_end,
                    fc,
                    compressed,
                });
            }
            return Some(out);
        }
        break;
    }
    None
}

fn clean_doc_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            // Word binary control marks we keep as layout boundaries.
            '\r' | '\u{000B}' | '\u{000C}' | '\u{2028}' | '\u{2029}' => out.push('\n'),
            '\u{0007}' | '\u{001E}' => out.push('\n'),
            '\u{0009}' | '\u{001F}' => out.push('\t'),
            '\u{0000}'..='\u{0008}' | '\u{000E}'..='\u{001D}' => {}
            _ => out.push(ch),
        }
    }
    normalize_doc_whitespace(&out)
}

fn normalize_doc_whitespace(s: &str) -> String {
    let mut normalized = String::with_capacity(s.len());
    let mut last_newline = false;
    for ch in s.chars() {
        if ch == '\n' {
            if !last_newline {
                normalized.push('\n');
            }
            last_newline = true;
            continue;
        }
        last_newline = false;
        normalized.push(ch);
    }
    normalized
        .lines()
        .map(|line| line.trim_matches(|c: char| c.is_whitespace()))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_line(line: &str) -> String {
    let t = line.replace('\0', "");
    let compact = t.split_whitespace().collect::<Vec<_>>().join(" ");
    compact.trim().to_string()
}

fn looks_like_content_line(line: &str, kind: &str) -> bool {
    if line.len() < 2 {
        return false;
    }
    let lower = line.to_ascii_lowercase();
    let blocked_literals = [
        "cambria",
        "calibri",
        "ms gothic",
        "arial",
        "times new roman",
        "宋体",
        "黑体",
        "楷体",
        "wps office emf_",
    ];
    if blocked_literals
        .iter()
        .any(|v| lower == *v || line == *v || lower.starts_with(v))
    {
        return false;
    }
    if matches!(
        lower.as_str(),
        "root entry"
            | "summaryinformation"
            | "documentsummaryinformation"
            | "worddocument"
            | "powerpoint document"
            | "current user"
            | "workbook"
            | "book"
            | "normal.dotm"
            | "ksoProductBuildVer"
    ) {
        return false;
    }
    if lower.contains("root entry")
        || lower.contains("summaryinformation")
        || lower.contains("documentsummaryinformation")
        || lower.contains("kso")
    {
        return false;
    }
    if lower.contains("ihdr")
        || lower.contains("idat")
        || lower.contains("phys")
        || lower.contains("wmfc")
    {
        return false;
    }
    let alpha_like = line
        .chars()
        .filter(|c| {
            c.is_alphanumeric()
                || c.is_whitespace()
                || ('\u{4E00}'..='\u{9FFF}').contains(c)
                || ('\u{3040}'..='\u{30FF}').contains(c)
                || ('\u{AC00}'..='\u{D7AF}').contains(c)
        })
        .count();
    let ratio = alpha_like * 100 / line.chars().count();
    if ratio < 45 {
        return false;
    }
    let has_hangul = line.chars().any(|c| ('\u{AC00}'..='\u{D7AF}').contains(&c));
    let has_cjk = line.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c));
    let has_ascii_alpha = line.chars().any(|c| c.is_ascii_alphabetic());
    if kind == "doc" && has_hangul {
        return false;
    }
    if has_hangul && !has_ascii_alpha && line.chars().count() <= 8 {
        return false;
    }
    if line.chars().count() <= 3 && !has_ascii_alpha && !has_cjk {
        return false;
    }
    let hex_like = line.chars().all(|c| c.is_ascii_hexdigit()) && line.len() >= 12;
    if hex_like {
        return false;
    }
    let compact: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.len() == 32 && compact.chars().all(|c| c.is_ascii_hexdigit()) {
        return false;
    }
    if kind == "xls" {
        return ratio >= 35;
    }
    let has_latin_word = line
        .split_whitespace()
        .any(|w| w.chars().filter(|c| c.is_ascii_alphabetic()).count() >= 3);
    if !(has_cjk || has_latin_word) {
        return false;
    }
    true
}

#[derive(Debug, Clone)]
struct OleStream {
    name: String,
    data: Vec<u8>,
}

fn select_scan_bytes<'a>(input: &'a [u8], kind: &str, streams: Option<&'a Vec<OleStream>>) -> &'a [u8] {
    let Some(streams) = streams else {
        return input;
    };
    let targets: &[&str] = match kind {
        "doc" => &["WordDocument", "1Table", "0Table", "Data"],
        "xls" => &["Workbook", "Book"],
        "ppt" => &["PowerPoint Document", "Current User"],
        _ => &[],
    };
    if targets.is_empty() {
        return input;
    }
    for stream in streams {
        if targets.iter().any(|t| stream.name.eq_ignore_ascii_case(t)) {
            return &stream.data;
        }
    }
    input
}

fn parse_ole_streams(input: &[u8]) -> Option<Vec<OleStream>> {
    let cfb = Cfb::parse(input)?;
    let mut out = Vec::new();
    for entry in &cfb.dir_entries {
        if entry.obj_type != 2 {
            continue;
        }
        if entry.name.is_empty() {
            continue;
        }
        let bytes = cfb.read_stream(entry).ok()?;
        out.push(OleStream {
            name: entry.name.clone(),
            data: bytes,
        });
    }
    Some(out)
}

#[derive(Debug, Clone)]
struct DirEntry {
    name: String,
    obj_type: u8,
    start_sector: u32,
    stream_size: u64,
}

struct Cfb<'a> {
    input: &'a [u8],
    sector_size: usize,
    mini_stream_cutoff: u32,
    fat: Vec<u32>,
    mini_fat: Vec<u32>,
    mini_stream: Vec<u8>,
    dir_entries: Vec<DirEntry>,
}

impl<'a> Cfb<'a> {
    fn parse(input: &'a [u8]) -> Option<Self> {
        if input.len() < 512 || !input.starts_with(OLE_MAGIC) {
            return None;
        }
        let sector_shift = le_u16(input, 0x1E)?;
        let sector_size = 1usize.checked_shl(u32::from(sector_shift))?;
        if !(512..=4096).contains(&sector_size) {
            return None;
        }
        let first_dir_sector = le_u32(input, 0x30)?;
        let mini_stream_cutoff = le_u32(input, 0x38)?;
        let first_difat_sector = le_u32(input, 0x44)?;
        let num_difat_sectors = le_u32(input, 0x48)?;
        let num_fat_sectors = le_u32(input, 0x2C)? as usize;
        let first_mini_fat_sector = le_u32(input, 0x3C)?;
        let num_mini_fat_sectors = le_u32(input, 0x40)? as usize;

        let mut difat = Vec::new();
        for i in 0..109 {
            let v = le_u32(input, 0x4C + i * 4)?;
            if !is_free_sector(v) {
                difat.push(v);
            }
        }
        let mut next = first_difat_sector;
        for _ in 0..num_difat_sectors {
            if is_end_of_chain(next) || is_free_sector(next) {
                break;
            }
            let sec = sector(input, sector_size, next)?;
            let per = (sector_size / 4).saturating_sub(1);
            for i in 0..per {
                let v = le_u32(sec, i * 4)?;
                if !is_free_sector(v) {
                    difat.push(v);
                }
            }
            next = le_u32(sec, per * 4)?;
        }
        if difat.len() > num_fat_sectors.saturating_add(8_192) {
            return None;
        }
        let mut fat = Vec::new();
        for fat_sector in difat.iter().take(num_fat_sectors.max(difat.len())) {
            let sec = sector(input, sector_size, *fat_sector)?;
            for i in 0..(sector_size / 4) {
                fat.push(le_u32(sec, i * 4)?);
            }
        }
        if fat.is_empty() {
            return None;
        }
        let mut cfb = Cfb {
            input,
            sector_size,
            mini_stream_cutoff,
            fat,
            mini_fat: Vec::new(),
            mini_stream: Vec::new(),
            dir_entries: Vec::new(),
        };
        cfb.dir_entries = cfb.parse_directory(first_dir_sector)?;
        cfb.mini_fat = cfb.parse_mini_fat(first_mini_fat_sector, num_mini_fat_sectors)?;
        cfb.mini_stream = cfb.parse_mini_stream();
        Some(cfb)
    }

    fn parse_directory(&self, first_sector: u32) -> Option<Vec<DirEntry>> {
        let bytes = self.read_chain(first_sector, 4 * 1024 * 1024)?;
        let mut out = Vec::new();
        for chunk in bytes.chunks_exact(128) {
            let name_len = le_u16(chunk, 0x40)? as usize;
            if name_len < 2 || name_len > 64 {
                continue;
            }
            let name_u16_len = name_len / 2 - 1;
            let mut name_vec = Vec::new();
            for i in 0..name_u16_len {
                name_vec.push(le_u16(chunk, i * 2)?);
            }
            let name = String::from_utf16(&name_vec).ok()?;
            let obj_type = chunk[0x42];
            let start_sector = le_u32(chunk, 0x74)?;
            let stream_size = le_u64(chunk, 0x78)?;
            out.push(DirEntry {
                name,
                obj_type,
                start_sector,
                stream_size,
            });
        }
        Some(out)
    }

    fn read_stream(&self, entry: &DirEntry) -> Result<Vec<u8>, ()> {
        if entry.stream_size == 0 {
            return Ok(Vec::new());
        }
        if entry.stream_size < u64::from(self.mini_stream_cutoff) {
            return self.read_mini_stream(entry);
        }
        let mut bytes = self
            .read_chain(entry.start_sector, entry.stream_size.min(32 * 1024 * 1024) as usize)
            .ok_or(())?;
        bytes.truncate(entry.stream_size as usize);
        Ok(bytes)
    }

    fn parse_mini_fat(&self, first_sector: u32, sector_count: usize) -> Option<Vec<u32>> {
        if sector_count == 0 || is_end_of_chain(first_sector) || is_free_sector(first_sector) {
            return Some(Vec::new());
        }
        let max_len = sector_count.saturating_mul(self.sector_size);
        let bytes = self.read_chain(first_sector, max_len)?;
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Some(out)
    }

    fn parse_mini_stream(&self) -> Vec<u8> {
        let Some(root) = self
            .dir_entries
            .iter()
            .find(|entry| entry.obj_type == 5 && entry.name.eq_ignore_ascii_case("Root Entry"))
        else {
            return Vec::new();
        };
        if root.stream_size == 0 {
            return Vec::new();
        }
        let max_len = root.stream_size.min(32 * 1024 * 1024) as usize;
        let mut bytes = self.read_chain(root.start_sector, max_len).unwrap_or_default();
        bytes.truncate(root.stream_size as usize);
        bytes
    }

    fn read_mini_stream(&self, entry: &DirEntry) -> Result<Vec<u8>, ()> {
        let mut bytes = read_mini_chain(
            &self.mini_stream,
            &self.mini_fat,
            entry.start_sector,
            entry.stream_size as usize,
        )?;
        bytes.truncate(entry.stream_size as usize);
        Ok(bytes)
    }

    fn read_chain(&self, start_sector: u32, max_len: usize) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        let mut visited = 0usize;
        let mut sec = start_sector;
        while !is_end_of_chain(sec) && !is_free_sector(sec) {
            let s = sector(self.input, self.sector_size, sec)?;
            out.extend_from_slice(s);
            if out.len() >= max_len {
                break;
            }
            let idx = sec as usize;
            sec = *self.fat.get(idx)?;
            visited += 1;
            if visited > self.fat.len() {
                break;
            }
        }
        Some(out)
    }
}

fn sector(input: &[u8], sector_size: usize, sector_id: u32) -> Option<&[u8]> {
    let offset = 512usize.checked_add((sector_id as usize).checked_mul(sector_size)?)?;
    let end = offset.checked_add(sector_size)?;
    input.get(offset..end)
}

fn read_mini_chain(
    mini_stream: &[u8],
    mini_fat: &[u32],
    start_sector: u32,
    max_len: usize,
) -> Result<Vec<u8>, ()> {
    if max_len == 0 {
        return Ok(Vec::new());
    }
    if mini_stream.is_empty() || mini_fat.is_empty() {
        return Err(());
    }
    const MINI_SECTOR_SIZE: usize = 64;
    let mut out = Vec::new();
    let mut visited = 0usize;
    let mut sec = start_sector;
    while !is_end_of_chain(sec) && !is_free_sector(sec) {
        let idx = sec as usize;
        let off = idx.checked_mul(MINI_SECTOR_SIZE).ok_or(())?;
        let end = off.checked_add(MINI_SECTOR_SIZE).ok_or(())?;
        let chunk = mini_stream.get(off..end).ok_or(())?;
        out.extend_from_slice(chunk);
        if out.len() >= max_len {
            break;
        }
        sec = *mini_fat.get(idx).ok_or(())?;
        visited += 1;
        if visited > mini_fat.len() {
            return Err(());
        }
    }
    Ok(out)
}

fn le_u16(input: &[u8], offset: usize) -> Option<u16> {
    let b = input.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn le_u32(input: &[u8], offset: usize) -> Option<u32> {
    let b = input.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn le_u64(input: &[u8], offset: usize) -> Option<u64> {
    let b = input.get(offset..offset + 8)?;
    Some(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}

fn is_end_of_chain(v: u32) -> bool {
    v >= 0xFFFFFFF8
}

fn is_free_sector(v: u32) -> bool {
    v == 0xFFFFFFFF
}

fn extract_ascii_strings(input: &[u8], min_len: usize, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    let mut consumed = 0usize;
    for &b in input {
        let printable = (0x20..=0x7e).contains(&b) || b == b'\t' || b == b' ';
        if printable {
            buf.push(b);
            continue;
        }
        if buf.len() >= min_len {
            let s = String::from_utf8_lossy(&buf).to_string();
            consumed += s.len();
            out.push(s);
        }
        buf.clear();
        if consumed >= max_chars {
            break;
        }
    }
    out
}

fn extract_latin1_strings(input: &[u8], min_len: usize, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    let mut consumed = 0usize;
    for &b in input {
        let printable = (0x20..=0x7e).contains(&b) || (0xa0..=0xff).contains(&b);
        if printable {
            buf.push(b);
            continue;
        }
        if buf.len() >= min_len {
            let s = buf.iter().map(|c| *c as char).collect::<String>();
            consumed += s.len();
            out.push(s);
        }
        buf.clear();
        if consumed >= max_chars {
            break;
        }
    }
    out
}

fn extract_utf16le_strings(input: &[u8], min_len: usize, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = Vec::<u16>::new();
    let mut consumed = 0usize;
    for chunk in input.chunks_exact(2) {
        let u = u16::from_le_bytes([chunk[0], chunk[1]]);
        let ch = char::from_u32(u as u32).unwrap_or('\0');
        let printable = ch.is_ascii_graphic()
            || ch.is_ascii_whitespace()
            || ('\u{4E00}'..='\u{9FFF}').contains(&ch)
            || ('\u{3040}'..='\u{30FF}').contains(&ch)
            || ('\u{AC00}'..='\u{D7AF}').contains(&ch);
        if printable {
            buf.push(u);
            continue;
        }
        if buf.len() >= min_len && let Ok(s) = String::from_utf16(&buf) {
            consumed += s.len();
            out.push(s);
        }
        buf.clear();
        if consumed >= max_chars {
            break;
        }
    }
    out
}

fn repair_utf8_mojibake(s: &str) -> Option<String> {
    if s.is_empty() {
        return None;
    }
    let suspicious = s
        .chars()
        .filter(|c| matches!(*c, 'Ã' | 'Â' | 'æ' | 'å' | 'è' | 'é' | 'ç' | 'ï' | 'ð'))
        .count();
    if suspicious * 8 < s.chars().count() {
        return None;
    }
    let mut bytes = Vec::with_capacity(s.len());
    for c in s.chars() {
        if (c as u32) > 0xff {
            return None;
        }
        bytes.push(c as u8);
    }
    String::from_utf8(bytes).ok()
}

fn trim_leading_noise(lines: Vec<String>) -> Vec<String> {
    let mut trimmed: Vec<String> = lines;
    while let Some(first) = trimmed.first() {
        if is_leading_garbage(first) {
            trimmed.remove(0);
        } else {
            break;
        }
    }
    if let Some(anchor) = trimmed.iter().position(|line| is_body_anchor(line)) {
        trimmed = trimmed.into_iter().skip(anchor).collect();
    }
    if let Some(anchor) = trimmed.iter().position(|line| {
        line.chars()
            .filter(|c| ('\u{4E00}'..='\u{9FFF}').contains(c))
            .count()
            >= 6
    }) {
        trimmed = trimmed.into_iter().skip(anchor.saturating_sub(2)).collect();
    }
    trimmed
}

fn trim_trailing_noise(lines: Vec<String>, kind: &str) -> Vec<String> {
    let mut trimmed = lines;
    while let Some(last) = trimmed.last() {
        if is_trailing_garbage(last, kind) {
            trimmed.pop();
        } else {
            break;
        }
    }
    trimmed
}

fn truncate_doc_tail_noise(lines: Vec<String>, kind: &str) -> Vec<String> {
    if kind != "doc" || lines.is_empty() {
        return lines;
    }
    let mut out = Vec::new();
    let mut short_noise_streak = 0usize;
    for line in lines {
        if is_doc_noise_anchor(&line) {
            break;
        }
        if is_doc_short_noise_line(&line) || is_trailing_garbage(&line, kind) {
            short_noise_streak += 1;
            if short_noise_streak >= 3 {
                break;
            }
            continue;
        }
        short_noise_streak = 0;
        out.push(line);
    }
    out
}

fn is_body_anchor(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return false;
    }
    if t.eq_ignore_ascii_case("python-docx") || t.eq_ignore_ascii_case("generated by python-docx") {
        return false;
    }
    let cjk = t
        .chars()
        .filter(|c| ('\u{4E00}'..='\u{9FFF}').contains(c))
        .count();
    let hangul = t
        .chars()
        .filter(|c| ('\u{AC00}'..='\u{D7AF}').contains(c))
        .count();
    let ascii_upper = t.chars().filter(|c| c.is_ascii_uppercase()).count();
    let punct = t.chars().filter(|c| c.is_ascii_punctuation()).count();
    cjk >= 3 && hangul == 0 && ascii_upper <= 2 && punct <= 3
}

fn is_leading_garbage(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return true;
    }
    if t.eq_ignore_ascii_case("python-docx") || t.eq_ignore_ascii_case("generated by python-docx") {
        return false;
    }
    if t.len() <= 2 {
        return true;
    }
    if t.chars().all(|c| c.is_ascii_hexdigit()) && t.len() >= 8 {
        return true;
    }
    let cjk = t
        .chars()
        .filter(|c| ('\u{4E00}'..='\u{9FFF}').contains(c))
        .count();
    let hangul = t
        .chars()
        .filter(|c| ('\u{AC00}'..='\u{D7AF}').contains(c))
        .count();
    let ascii_alpha = t.chars().filter(|c| c.is_ascii_alphabetic()).count();
    if hangul > 0 && ascii_alpha == 0 && cjk == 0 {
        return true;
    }
    cjk == 0 && ascii_alpha == 0
}

fn is_trailing_garbage(line: &str, kind: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return true;
    }
    if !looks_like_content_line(t, kind) {
        return true;
    }
    let cjk = t
        .chars()
        .filter(|c| ('\u{4E00}'..='\u{9FFF}').contains(c))
        .count();
    let ascii_alpha = t.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let digit = t.chars().filter(|c| c.is_ascii_digit()).count();

    // Typical mojibake tail in .doc fallback: mostly CJK-looking noise + a few ascii letters.
    if kind == "doc" && cjk >= 2 && ascii_alpha > 0 && t.chars().count() <= 10 {
        return true;
    }
    if kind == "doc" && cjk >= 2 && ascii_alpha == 0 && t.chars().count() <= 4 {
        return true;
    }
    // Isolated short symbol-like tails (e.g. "耀(")
    if kind == "doc" && cjk == 1 && ascii_alpha == 0 && digit == 0 && t.chars().count() <= 2 {
        return true;
    }
    false
}

fn is_doc_noise_anchor(line: &str) -> bool {
    let t = line.trim().to_ascii_lowercase();
    t == "0table"
        || t == "data"
        || t.contains("wpscustomdata")
        || t.contains("img_")
        || t.contains("图片")
        || t.contains("worddocument")
}

fn is_doc_short_noise_line(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return true;
    }
    let len = t.chars().count();
    if !(2..=6).contains(&len) {
        return false;
    }
    looks_like_mojibake(t)
}

#[cfg(test)]
mod tests {
    use super::{
        ansi_piece_byte_len, build_text_blocks, clean_doc_text, decode_ansi_piece, decode_ppt_text_atom, detect_kind,
        decode_bytes_with_strategy, extract_doc_text_from_table, extract_legacy_mso_text,
        extract_ppt_text_structured, extract_xls_text_structured, format_excel_number, format_sheet_blocks,
        group_ppt_slide_text, locate_clx, looks_like_mojibake, parse_boundsheet_name, parse_clx_piece_table,
        parse_doc_fib, parse_sst_strings_with_continue, read_mini_chain, repair_utf8_mojibake,
        score_text_quality, trim_trailing_noise, truncate_doc_tail_noise, walk_ppt_records, DocFib, OleStream,
        PptRecord, XlsCell,
    };

    #[test]
    fn detects_doc_kind() {
        let data = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1....WordDocument....";
        assert_eq!(detect_kind(data), "doc");
    }

    #[test]
    fn repairs_utf8_mojibake() {
        let fixed = repair_utf8_mojibake("è¿\u{00a0}æ").unwrap_or_default();
        assert!(fixed.is_empty() || fixed.chars().any(|c| !c.is_ascii()));
    }

    #[test]
    fn reads_mini_chain_across_multiple_sectors() {
        let mut mini_stream = vec![0u8; 3 * 64];
        mini_stream[0..64].fill(b'A');
        mini_stream[64..128].fill(b'B');
        mini_stream[128..192].fill(b'C');
        let mini_fat = vec![2, 0xFFFF_FFFF, 1];
        let out = read_mini_chain(&mini_stream, &mini_fat, 0, 130).expect("read mini chain");
        assert_eq!(out.len(), 192);
        assert_eq!(out[0], b'A');
        assert_eq!(out[64], b'C');
        assert_eq!(out[128], b'B');
    }

    #[test]
    fn mini_chain_fails_on_out_of_bounds_sector() {
        let mini_stream = vec![0u8; 64];
        let mini_fat = vec![0xFFFF_FFFF];
        let err = read_mini_chain(&mini_stream, &mini_fat, 3, 32).err();
        assert!(err.is_some());
    }

    #[test]
    fn parses_doc_fib_and_clx_offsets() {
        let mut word = vec![0u8; 0x1AA];
        word[0x00..0x02].copy_from_slice(&0xA5ECu16.to_le_bytes());
        word[0x02..0x04].copy_from_slice(&0x00C1u16.to_le_bytes());
        word[0x0A..0x0C].copy_from_slice(&0x0200u16.to_le_bytes());
        word[0x1A2..0x1A6].copy_from_slice(&12u32.to_le_bytes());
        word[0x1A6..0x1AA].copy_from_slice(&20u32.to_le_bytes());

        let fib = parse_doc_fib(&word).expect("valid fib");
        assert!(fib.use_1table);
        assert_eq!(fib.fc_clx, 12);
        assert_eq!(fib.lcb_clx, 20);
        assert_eq!(fib.lid, 0);
        assert_eq!(fib.chs, 0);
        assert_eq!(fib.chs_tables, 0);
    }

    #[test]
    fn rejects_non_word_fib() {
        let mut word = vec![0u8; 0x1AA];
        word[0x00..0x02].copy_from_slice(&0xFFFFu16.to_le_bytes());
        word[0x02..0x04].copy_from_slice(&0x00C1u16.to_le_bytes());
        word[0x1A6..0x1AA].copy_from_slice(&21u32.to_le_bytes());
        assert!(parse_doc_fib(&word).is_none());
    }

    #[test]
    fn locate_clx_prefers_fib_offsets_then_fallback_scan() {
        let fib = DocFib {
            use_1table: false,
            fc_clx: 8,
            lcb_clx: 16,
            lid: 0,
            chs: 0,
            chs_tables: 0,
        };
        let mut table = vec![0u8; 32];
        table[8] = 0x02;
        table[9..13].copy_from_slice(&11u32.to_le_bytes());
        let clx = locate_clx(&table, fib).expect("clx from fib offsets");
        assert_eq!(clx.len(), 16);

        let fib_bad = DocFib {
            use_1table: false,
            fc_clx: 1000,
            lcb_clx: 16,
            lid: 0,
            chs: 0,
            chs_tables: 0,
        };
        let mut scan_table = vec![0u8; 24];
        scan_table[2] = 0x02;
        scan_table[3..7].copy_from_slice(&16u32.to_le_bytes());
        let fallback = locate_clx(&scan_table, fib_bad).expect("fallback scan clx");
        assert_eq!(fallback.len(), 21);
    }

    #[test]
    fn parses_clx_piece_table() {
        // CLX: [0x02][lcb=16] + PlcPcd (n=1): cp[0]=0, cp[1]=5, pcd.fc=0x40000000(ANSI, fc=0)
        let mut clx = vec![0x02];
        clx.extend_from_slice(&16u32.to_le_bytes());
        clx.extend_from_slice(&0u32.to_le_bytes());
        clx.extend_from_slice(&5u32.to_le_bytes());
        clx.extend_from_slice(&[0, 0]); // pcd metadata
        clx.extend_from_slice(&0x4000_0000u32.to_le_bytes());
        clx.extend_from_slice(&[0, 0]); // pcd tail
        let pieces = parse_clx_piece_table(&clx).expect("parse piece table");
        assert_eq!(pieces.len(), 1);
        assert!(pieces[0].compressed);
        assert_eq!(pieces[0].cp_start, 0);
        assert_eq!(pieces[0].cp_end, 5);
        assert_eq!(pieces[0].fc, 0);
    }

    #[test]
    fn extracts_doc_text_by_piece_ranges() {
        let mut word = vec![0u8; 0x1AA];
        word[0x00..0x02].copy_from_slice(&0xA5ECu16.to_le_bytes());
        word[0x02..0x04].copy_from_slice(&0x00C1u16.to_le_bytes());
        word[0x0A..0x0C].copy_from_slice(&0u16.to_le_bytes());
        word[0x1A2..0x1A6].copy_from_slice(&0u32.to_le_bytes());
        word[0x1A6..0x1AA].copy_from_slice(&21u32.to_le_bytes());

        word.extend_from_slice(b"Hello");

        let mut table = Vec::new();
        table.push(0x02);
        table.extend_from_slice(&16u32.to_le_bytes());
        table.extend_from_slice(&0u32.to_le_bytes());
        table.extend_from_slice(&5u32.to_le_bytes());
        table.extend_from_slice(&[0, 0]);
        table.extend_from_slice(&0x4000_0354u32.to_le_bytes()); // compressed, fc=(0x354/2)=>0x1AA
        table.extend_from_slice(&[0, 0]);

        let fib = parse_doc_fib(&word).expect("fib");
        let text = extract_doc_text_from_table(&word, &table, fib).expect("doc text");
        assert_eq!(text, "Hello");
    }

    #[test]
    fn decodes_ansi_piece_with_charset_mapping() {
        let fib = DocFib {
            use_1table: false,
            fc_clx: 0,
            lcb_clx: 16,
            lid: 0x0804,
            chs: 0,
            chs_tables: 134,
        };
        let s = decode_ansi_piece(&[0xD6, 0xD0, 0xCE, 0xC4], fib);
        assert_eq!(s, "中文");
    }

    #[test]
    fn clean_doc_text_normalizes_controls_and_newlines() {
        let raw = "A\rB\u{0007}\u{001E}C\t \u{001F}D\u{0001}\n\nE";
        let cleaned = clean_doc_text(raw);
        assert_eq!(cleaned, "A\nB\nC\t \tD\nE");
    }

    #[test]
    fn extracts_doc_mixed_zh_en_text_sample() {
        let mut word = vec![0u8; 0x1AA];
        word[0x00..0x02].copy_from_slice(&0xA5ECu16.to_le_bytes());
        word[0x02..0x04].copy_from_slice(&0x00C1u16.to_le_bytes());
        word[0x0A..0x0C].copy_from_slice(&0u16.to_le_bytes());
        word[0x14..0x16].copy_from_slice(&0u16.to_le_bytes()); // chsTables=1252
        word[0x1A2..0x1A6].copy_from_slice(&0u32.to_le_bytes());
        word[0x1A6..0x1AA].copy_from_slice(&33u32.to_le_bytes());

        // ANSI piece: "ABC" (single-byte)
        let ansi_offset = word.len() as u32;
        word.extend_from_slice(b"ABC");
        // Unicode piece: "中文"
        let unicode_offset = word.len() as u32;
        for u in "中文".encode_utf16() {
            word.extend_from_slice(&u.to_le_bytes());
        }

        // CLX + PlcPcd: cp=[0,3,5], piece0 ANSI(3 chars), piece1 Unicode(2 chars)
        let mut table = Vec::new();
        table.push(0x02);
        table.extend_from_slice(&28u32.to_le_bytes()); // lcb
        table.extend_from_slice(&0u32.to_le_bytes());
        table.extend_from_slice(&3u32.to_le_bytes());
        table.extend_from_slice(&5u32.to_le_bytes());
        table.extend_from_slice(&[0, 0]);
        table.extend_from_slice(&(0x4000_0000u32 | (ansi_offset * 2)).to_le_bytes());
        table.extend_from_slice(&[0, 0]);
        table.extend_from_slice(&[0, 0]);
        table.extend_from_slice(&unicode_offset.to_le_bytes());
        table.extend_from_slice(&[0, 0]);

        let fib = parse_doc_fib(&word).expect("fib");
        let text = extract_doc_text_from_table(&word, &table, fib).expect("mixed text");
        assert_eq!(text, "ABC中文");
    }

    #[test]
    fn extracts_doc_dbcs_ansi_piece_text() {
        let mut word = vec![0u8; 0x1AA];
        word[0x00..0x02].copy_from_slice(&0xA5ECu16.to_le_bytes());
        word[0x02..0x04].copy_from_slice(&0x00C1u16.to_le_bytes());
        word[0x0A..0x0C].copy_from_slice(&0u16.to_le_bytes());
        word[0x14..0x16].copy_from_slice(&134u16.to_le_bytes()); // chsTables=GBK
        word[0x1A2..0x1A6].copy_from_slice(&0u32.to_le_bytes());
        word[0x1A6..0x1AA].copy_from_slice(&21u32.to_le_bytes());

        let ansi_offset = word.len() as u32;
        word.extend_from_slice(&[0xD6, 0xD0, 0xCE, 0xC4]); // 中文 (GBK)

        let mut table = Vec::new();
        table.push(0x02);
        table.extend_from_slice(&16u32.to_le_bytes());
        table.extend_from_slice(&0u32.to_le_bytes());
        table.extend_from_slice(&2u32.to_le_bytes()); // cp chars
        table.extend_from_slice(&[0, 0]);
        table.extend_from_slice(&(0x4000_0000u32 | (ansi_offset * 2)).to_le_bytes());
        table.extend_from_slice(&[0, 0]);

        let fib = parse_doc_fib(&word).expect("fib");
        let text = extract_doc_text_from_table(&word, &table, fib).expect("dbcs ansi text");
        assert_eq!(text, "中文");
    }

    #[test]
    fn ansi_piece_len_uses_dbcs_width_when_needed() {
        let fib = DocFib {
            use_1table: false,
            fc_clx: 0,
            lcb_clx: 16,
            lid: 0x0804,
            chs: 0,
            chs_tables: 134,
        };
        assert_eq!(ansi_piece_byte_len(3, fib), 6);
    }

    #[test]
    fn parses_boundsheet_ascii_name() {
        let mut rec = vec![0u8; 8];
        rec[6] = 6; // cch
        rec[7] = 0; // compressed 8-bit
        rec.extend_from_slice(b"Sheet1");
        let name = parse_boundsheet_name(&rec).expect("sheet name");
        assert_eq!(name, "Sheet1");
    }

    #[test]
    fn extracts_xls_sheet_names_from_biff_records() {
        let mut workbook = Vec::new();
        // BOF
        workbook.extend_from_slice(&0x0809u16.to_le_bytes());
        workbook.extend_from_slice(&0x0008u16.to_le_bytes());
        workbook.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
        // BOUNDSHEET (unicode "数据")
        let mut bs = vec![0u8; 8];
        bs[6] = 2;
        bs[7] = 1;
        for u in "数据".encode_utf16() {
            bs.extend_from_slice(&u.to_le_bytes());
        }
        workbook.extend_from_slice(&0x0085u16.to_le_bytes());
        workbook.extend_from_slice(&(bs.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&bs);
        // EOF
        workbook.extend_from_slice(&0x000Au16.to_le_bytes());
        workbook.extend_from_slice(&0u16.to_le_bytes());

        let streams = vec![OleStream {
            name: "Workbook".to_string(),
            data: workbook,
        }];
        let text = extract_xls_text_structured(&streams).expect("xls text");
        assert_eq!(text, "Sheet1: 数据");
    }

    #[test]
    fn parses_sst_with_continue_chain() {
        let mut workbook = Vec::new();
        // SST: cstTotal=2, cstUnique=2, payload starts with first full string "Hello"
        let mut sst = Vec::new();
        sst.extend_from_slice(&2u32.to_le_bytes());
        sst.extend_from_slice(&2u32.to_le_bytes());
        sst.extend_from_slice(&5u16.to_le_bytes());
        sst.push(0);
        sst.extend_from_slice(b"Hello");
        workbook.extend_from_slice(&0x00FCu16.to_le_bytes());
        workbook.extend_from_slice(&(sst.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&sst);
        // CONTINUE: second string "World"
        let mut cont = Vec::new();
        cont.extend_from_slice(&5u16.to_le_bytes());
        cont.push(0);
        cont.extend_from_slice(b"World");
        workbook.extend_from_slice(&0x003Cu16.to_le_bytes());
        workbook.extend_from_slice(&(cont.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&cont);

        let strings = parse_sst_strings_with_continue(&workbook, 0, None).expect("sst strings");
        assert_eq!(strings, vec!["Hello".to_string(), "World".to_string()]);
    }

    #[test]
    fn extracts_xls_cell_value_records() {
        let mut workbook = Vec::new();
        // Globals BOF
        workbook.extend_from_slice(&0x0809u16.to_le_bytes());
        workbook.extend_from_slice(&0x0008u16.to_le_bytes());
        workbook.extend_from_slice(&[0, 0, 0x05, 0x00, 0, 0, 0, 0]);
        // BOUNDSHEET "SheetA"
        let mut bs = vec![0u8; 8];
        bs[6] = 6;
        bs[7] = 0;
        bs.extend_from_slice(b"SheetA");
        workbook.extend_from_slice(&0x0085u16.to_le_bytes());
        workbook.extend_from_slice(&(bs.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&bs);
        // SST with one string: "Foo"
        let mut sst = Vec::new();
        sst.extend_from_slice(&1u32.to_le_bytes());
        sst.extend_from_slice(&1u32.to_le_bytes());
        sst.extend_from_slice(&3u16.to_le_bytes());
        sst.push(0);
        sst.extend_from_slice(b"Foo");
        workbook.extend_from_slice(&0x00FCu16.to_le_bytes());
        workbook.extend_from_slice(&(sst.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&sst);
        // Globals EOF
        workbook.extend_from_slice(&0x000Au16.to_le_bytes());
        workbook.extend_from_slice(&0u16.to_le_bytes());
        // Worksheet BOF
        workbook.extend_from_slice(&0x0809u16.to_le_bytes());
        workbook.extend_from_slice(&0x0008u16.to_le_bytes());
        workbook.extend_from_slice(&[0, 0, 0x10, 0x00, 0, 0, 0, 0]);
        // LABELSST row0 col0 idx0
        let mut labelsst = Vec::new();
        labelsst.extend_from_slice(&0u16.to_le_bytes());
        labelsst.extend_from_slice(&0u16.to_le_bytes());
        labelsst.extend_from_slice(&0u16.to_le_bytes());
        labelsst.extend_from_slice(&0u32.to_le_bytes());
        workbook.extend_from_slice(&0x00FDu16.to_le_bytes());
        workbook.extend_from_slice(&(labelsst.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&labelsst);
        // NUMBER row1 col1 = 42
        let mut number = Vec::new();
        number.extend_from_slice(&1u16.to_le_bytes());
        number.extend_from_slice(&1u16.to_le_bytes());
        number.extend_from_slice(&0u16.to_le_bytes());
        number.extend_from_slice(&42f64.to_le_bytes());
        workbook.extend_from_slice(&0x0203u16.to_le_bytes());
        workbook.extend_from_slice(&(number.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&number);
        // Worksheet EOF
        workbook.extend_from_slice(&0x000Au16.to_le_bytes());
        workbook.extend_from_slice(&0u16.to_le_bytes());

        let streams = vec![OleStream {
            name: "Workbook".to_string(),
            data: workbook,
        }];
        let text = extract_xls_text_structured(&streams).expect("xls text");
        assert!(text.contains("Sheet1: SheetA"));
        assert!(text.contains("SheetA\nR1C1=Foo\nR2C2=42"));
    }

    #[test]
    fn xls_codepage_guides_label_decoding() {
        let mut workbook = Vec::new();
        // Globals BOF
        workbook.extend_from_slice(&0x0809u16.to_le_bytes());
        workbook.extend_from_slice(&0x0008u16.to_le_bytes());
        workbook.extend_from_slice(&[0, 0, 0x05, 0x00, 0, 0, 0, 0]);
        // CODEPAGE = 936 (GBK)
        workbook.extend_from_slice(&0x0042u16.to_le_bytes());
        workbook.extend_from_slice(&2u16.to_le_bytes());
        workbook.extend_from_slice(&936u16.to_le_bytes());
        // BOUNDSHEET "S"
        let mut bs = vec![0u8; 8];
        bs[6] = 1;
        bs[7] = 0;
        bs.extend_from_slice(b"S");
        workbook.extend_from_slice(&0x0085u16.to_le_bytes());
        workbook.extend_from_slice(&(bs.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&bs);
        // Globals EOF
        workbook.extend_from_slice(&0x000Au16.to_le_bytes());
        workbook.extend_from_slice(&0u16.to_le_bytes());
        // Worksheet BOF
        workbook.extend_from_slice(&0x0809u16.to_le_bytes());
        workbook.extend_from_slice(&0x0008u16.to_le_bytes());
        workbook.extend_from_slice(&[0, 0, 0x10, 0x00, 0, 0, 0, 0]);
        // LABEL row0 col0 -> 中文 (GBK bytes)
        let mut label = Vec::new();
        label.extend_from_slice(&0u16.to_le_bytes());
        label.extend_from_slice(&0u16.to_le_bytes());
        label.extend_from_slice(&0u16.to_le_bytes());
        label.extend_from_slice(&4u16.to_le_bytes());
        label.extend_from_slice(&[0xD6, 0xD0, 0xCE, 0xC4]);
        workbook.extend_from_slice(&0x0204u16.to_le_bytes());
        workbook.extend_from_slice(&(label.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&label);
        // Worksheet EOF
        workbook.extend_from_slice(&0x000Au16.to_le_bytes());
        workbook.extend_from_slice(&0u16.to_le_bytes());

        let streams = vec![OleStream {
            name: "Workbook".to_string(),
            data: workbook,
        }];
        let text = extract_xls_text_structured(&streams).expect("xls text");
        assert!(text.contains("S\nR1C1=中文"));
    }

    #[test]
    fn sheet_block_output_is_sorted_by_row_col() {
        let cells = vec![
            XlsCell {
                sheet: "S1".to_string(),
                row: 3,
                col: 0,
                value: "d".to_string(),
            },
            XlsCell {
                sheet: "S1".to_string(),
                row: 0,
                col: 2,
                value: "a".to_string(),
            },
            XlsCell {
                sheet: "S1".to_string(),
                row: 0,
                col: 1,
                value: "b".to_string(),
            },
        ];
        let blocks = format_sheet_blocks(cells);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], "S1\nR1C2=b\nR1C3=a\nR4C1=d");
    }

    #[test]
    fn excel_number_format_avoids_scientific_notation() {
        assert_eq!(format_excel_number(1200000000000.0), "1200000000000");
        assert_eq!(format_excel_number(0.00000123), "0.00000123");
        assert_eq!(format_excel_number(-0.0), "0");
    }

    #[test]
    fn extracts_xls_formula_cached_non_numeric_values() {
        let mut workbook = Vec::new();
        // Globals BOF + BOUNDSHEET + EOF
        workbook.extend_from_slice(&0x0809u16.to_le_bytes());
        workbook.extend_from_slice(&0x0008u16.to_le_bytes());
        workbook.extend_from_slice(&[0, 0, 0x05, 0x00, 0, 0, 0, 0]);
        let mut bs = vec![0u8; 8];
        bs[6] = 2;
        bs[7] = 0;
        bs.extend_from_slice(b"S1");
        workbook.extend_from_slice(&0x0085u16.to_le_bytes());
        workbook.extend_from_slice(&(bs.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&bs);
        workbook.extend_from_slice(&0x000Au16.to_le_bytes());
        workbook.extend_from_slice(&0u16.to_le_bytes());
        // Worksheet BOF
        workbook.extend_from_slice(&0x0809u16.to_le_bytes());
        workbook.extend_from_slice(&0x0008u16.to_le_bytes());
        workbook.extend_from_slice(&[0, 0, 0x10, 0x00, 0, 0, 0, 0]);

        // FORMULA string pending at R1C1
        let mut formula_str = Vec::new();
        formula_str.extend_from_slice(&0u16.to_le_bytes());
        formula_str.extend_from_slice(&0u16.to_le_bytes());
        formula_str.extend_from_slice(&0u16.to_le_bytes());
        formula_str.extend_from_slice(&[0x00, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF]);
        formula_str.extend_from_slice(&[0u8; 6]);
        workbook.extend_from_slice(&0x0006u16.to_le_bytes());
        workbook.extend_from_slice(&(formula_str.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&formula_str);
        // STRING record -> "OK"
        let mut str_rec = Vec::new();
        str_rec.extend_from_slice(&2u16.to_le_bytes());
        str_rec.push(0);
        str_rec.extend_from_slice(b"OK");
        workbook.extend_from_slice(&0x0207u16.to_le_bytes());
        workbook.extend_from_slice(&(str_rec.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&str_rec);

        // FORMULA boolean TRUE at R1C2
        let mut formula_bool = Vec::new();
        formula_bool.extend_from_slice(&0u16.to_le_bytes());
        formula_bool.extend_from_slice(&1u16.to_le_bytes());
        formula_bool.extend_from_slice(&0u16.to_le_bytes());
        formula_bool.extend_from_slice(&[0x01, 0, 0x01, 0, 0, 0xFF, 0xFF, 0xFF]);
        formula_bool.extend_from_slice(&[0u8; 6]);
        workbook.extend_from_slice(&0x0006u16.to_le_bytes());
        workbook.extend_from_slice(&(formula_bool.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&formula_bool);

        // FORMULA error #DIV/0! at R1C3
        let mut formula_err = Vec::new();
        formula_err.extend_from_slice(&0u16.to_le_bytes());
        formula_err.extend_from_slice(&2u16.to_le_bytes());
        formula_err.extend_from_slice(&0u16.to_le_bytes());
        formula_err.extend_from_slice(&[0x02, 0, 0x07, 0, 0, 0xFF, 0xFF, 0xFF]);
        formula_err.extend_from_slice(&[0u8; 6]);
        workbook.extend_from_slice(&0x0006u16.to_le_bytes());
        workbook.extend_from_slice(&(formula_err.len() as u16).to_le_bytes());
        workbook.extend_from_slice(&formula_err);

        // Worksheet EOF
        workbook.extend_from_slice(&0x000Au16.to_le_bytes());
        workbook.extend_from_slice(&0u16.to_le_bytes());

        let streams = vec![OleStream {
            name: "Workbook".to_string(),
            data: workbook,
        }];
        let text = extract_xls_text_structured(&streams).expect("xls text");
        assert!(text.contains("S1\nR1C1=OK\nR1C2=TRUE\nR1C3=#DIV/0!"));
    }

    #[test]
    fn walks_ppt_container_records_recursively() {
        // Outer container (recVer=0xF, recType=0x03E8) with one child atom (type=0x0FA0, len=3)
        let mut child = Vec::new();
        child.extend_from_slice(&0x0000u16.to_le_bytes()); // recVer=0
        child.extend_from_slice(&0x0FA0u16.to_le_bytes());
        child.extend_from_slice(&3u32.to_le_bytes());
        child.extend_from_slice(&[1, 2, 3]);

        let mut outer = Vec::new();
        outer.extend_from_slice(&0x000Fu16.to_le_bytes()); // recVer=0xF container
        outer.extend_from_slice(&0x03E8u16.to_le_bytes());
        outer.extend_from_slice(&(child.len() as u32).to_le_bytes());
        outer.extend_from_slice(&child);

        let records = walk_ppt_records(&outer, 8, 1024).expect("walk records");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].rec_type, 0x03E8);
        assert_eq!(records[0].depth, 0);
        assert_eq!(records[1].rec_type, 0x0FA0);
        assert_eq!(records[1].depth, 1);
    }

    #[test]
    fn ppt_walker_rejects_oversized_record() {
        let mut rec = Vec::new();
        rec.extend_from_slice(&0x0000u16.to_le_bytes());
        rec.extend_from_slice(&0x03F3u16.to_le_bytes());
        rec.extend_from_slice(&4097u32.to_le_bytes());
        rec.extend_from_slice(&[0u8; 8]);
        let walked = walk_ppt_records(&rec, 4, 4096);
        assert!(walked.is_none());
    }

    #[test]
    fn decodes_ppt_text_atoms() {
        let bytes = decode_ppt_text_atom(0x0FA8, b"Title").expect("textbytes");
        assert_eq!(bytes, "Title");

        let mut chars = Vec::new();
        for u in "正文".encode_utf16() {
            chars.extend_from_slice(&u.to_le_bytes());
        }
        let decoded = decode_ppt_text_atom(0x0FA0, &chars).expect("textchars");
        assert_eq!(decoded, "正文");
    }

    #[test]
    fn extracts_ppt_text_from_container_records() {
        // child1: TextBytesAtom "Slide 1"
        let mut child1 = Vec::new();
        child1.extend_from_slice(&0x0000u16.to_le_bytes());
        child1.extend_from_slice(&0x0FA8u16.to_le_bytes());
        child1.extend_from_slice(&7u32.to_le_bytes());
        child1.extend_from_slice(b"Slide 1");

        // child2: TextCharsAtom "内容"
        let mut textchars_payload = Vec::new();
        for u in "内容".encode_utf16() {
            textchars_payload.extend_from_slice(&u.to_le_bytes());
        }
        let mut child2 = Vec::new();
        child2.extend_from_slice(&0x0000u16.to_le_bytes());
        child2.extend_from_slice(&0x0FA0u16.to_le_bytes());
        child2.extend_from_slice(&(textchars_payload.len() as u32).to_le_bytes());
        child2.extend_from_slice(&textchars_payload);

        let mut container_payload = Vec::new();
        container_payload.extend_from_slice(&child1);
        container_payload.extend_from_slice(&child2);

        let mut ppt_stream = Vec::new();
        ppt_stream.extend_from_slice(&0x000Fu16.to_le_bytes());
        ppt_stream.extend_from_slice(&0x03E8u16.to_le_bytes());
        ppt_stream.extend_from_slice(&(container_payload.len() as u32).to_le_bytes());
        ppt_stream.extend_from_slice(&container_payload);

        let streams = vec![OleStream {
            name: "PowerPoint Document".to_string(),
            data: ppt_stream,
        }];
        let text = extract_ppt_text_structured(&streams).expect("ppt text");
        assert!(text.contains("Slide 1"));
        assert!(text.contains("内容"));
    }

    #[test]
    fn groups_ppt_text_by_slide_and_dedups_neighbor_lines() {
        let records = vec![
            PptRecord {
                rec_type: 0x03EE,
                rec_len: 0,
                depth: 0,
                data: Vec::new(),
            },
            PptRecord {
                rec_type: 0x0FA8,
                rec_len: 5,
                depth: 1,
                data: b"Title".to_vec(),
            },
            PptRecord {
                rec_type: 0x0FA8,
                rec_len: 5,
                depth: 1,
                data: b"Title".to_vec(),
            },
            PptRecord {
                rec_type: 0x03EE,
                rec_len: 0,
                depth: 0,
                data: Vec::new(),
            },
            PptRecord {
                rec_type: 0x0FA8,
                rec_len: 4,
                depth: 1,
                data: b"Body".to_vec(),
            },
        ];
        let slides = group_ppt_slide_text(&records);
        assert_eq!(slides.len(), 2);
        assert_eq!(slides[0], vec!["Title".to_string()]);
        assert_eq!(slides[1], vec!["Body".to_string()]);
    }

    #[test]
    fn extracts_multi_slide_ppt_sample() {
        // Slide 1 container with TextBytesAtom "Intro"
        let mut s1_text = Vec::new();
        s1_text.extend_from_slice(&0x0000u16.to_le_bytes());
        s1_text.extend_from_slice(&0x0FA8u16.to_le_bytes());
        s1_text.extend_from_slice(&5u32.to_le_bytes());
        s1_text.extend_from_slice(b"Intro");
        let mut slide1 = Vec::new();
        slide1.extend_from_slice(&0x000Fu16.to_le_bytes());
        slide1.extend_from_slice(&0x03EEu16.to_le_bytes());
        slide1.extend_from_slice(&(s1_text.len() as u32).to_le_bytes());
        slide1.extend_from_slice(&s1_text);

        // Slide 2 container with TextCharsAtom "结论"
        let mut s2_payload = Vec::new();
        for u in "结论".encode_utf16() {
            s2_payload.extend_from_slice(&u.to_le_bytes());
        }
        let mut s2_text = Vec::new();
        s2_text.extend_from_slice(&0x0000u16.to_le_bytes());
        s2_text.extend_from_slice(&0x0FA0u16.to_le_bytes());
        s2_text.extend_from_slice(&(s2_payload.len() as u32).to_le_bytes());
        s2_text.extend_from_slice(&s2_payload);
        let mut slide2 = Vec::new();
        slide2.extend_from_slice(&0x000Fu16.to_le_bytes());
        slide2.extend_from_slice(&0x03EEu16.to_le_bytes());
        slide2.extend_from_slice(&(s2_text.len() as u32).to_le_bytes());
        slide2.extend_from_slice(&s2_text);

        let mut ppt_stream = Vec::new();
        ppt_stream.extend_from_slice(&slide1);
        ppt_stream.extend_from_slice(&slide2);

        let streams = vec![OleStream {
            name: "PowerPoint Document".to_string(),
            data: ppt_stream,
        }];
        let text = extract_ppt_text_structured(&streams).expect("ppt text");
        assert!(text.contains("Slide 1\nIntro"));
        assert!(text.contains("Slide 2\n结论"));
    }

    #[test]
    fn marks_unsupported_and_corrupted_warnings() {
        let mut input = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1".to_vec();
        input.extend_from_slice(b"random");
        let out = extract_legacy_mso_text(&input).expect("ole parse");
        assert_eq!(out.kind, "ole-unknown");
        assert!(out.warnings.iter().any(|w| w == "Unsupported"));
        assert!(out.warnings.iter().any(|w| w == "Corrupted"));
    }

    #[test]
    fn marks_partial_extracted_for_capped_large_input() {
        let mut input = vec![0u8; (64 * 1024 * 1024) + 32];
        input[..8].copy_from_slice(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1");
        input[8..20].copy_from_slice(b"WordDocument");
        let out = extract_legacy_mso_text(&input).expect("ole parse");
        assert!(out.warnings.iter().any(|w| w == "PartialExtracted"));
    }

    #[test]
    fn asserts_empty_content_for_non_text_ole_payload() {
        let mut input = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1".to_vec();
        input.extend_from_slice(&[0u8; 256]);
        let out = extract_legacy_mso_text(&input).expect("ole parse");
        assert!(out.text.trim().is_empty());
    }

    #[test]
    fn asserts_corrupted_file_warning_for_truncated_ole() {
        let mut input = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1".to_vec();
        input.extend_from_slice(b"truncated");
        let out = extract_legacy_mso_text(&input).expect("ole parse");
        assert!(out.warnings.iter().any(|w| w == "Corrupted"));
    }

    #[test]
    fn decode_strategy_prefers_bom() {
        let bytes = b"\xEF\xBB\xBFhello";
        let out = decode_bytes_with_strategy(bytes, None);
        assert_eq!(out, "hello");
    }

    #[test]
    fn decode_strategy_falls_back_for_non_utf8_bytes() {
        let out = decode_bytes_with_strategy(&[0xC0, 0x80, 0x41], None);
        assert!(!out.is_empty());
    }

    #[test]
    fn mojibake_detector_hits_typical_pattern() {
        assert!(looks_like_mojibake("Ã¤Â½â"));
        assert!(!looks_like_mojibake("normal text"));
    }

    #[test]
    fn decode_strategy_second_passes_to_utf8_on_mojibake() {
        let out = decode_bytes_with_strategy("你好".as_bytes(), Some(encoding_rs::WINDOWS_1252));
        assert_eq!(out, "你好");
    }

    #[test]
    fn quality_score_penalizes_metadata_noise() {
        let low = score_text_quality("Root Entry\nSummaryInformation", "doc");
        let high = score_text_quality("项目总结\n这是正文段落", "doc");
        assert!(low < 45);
        assert!(high >= 45);
    }

    #[test]
    fn builds_blocks_for_xls_sections() {
        let text = "Sheet 1: A\nS1\nR1C1=1\nSheet 2: B\nS2\nR1C1=2";
        let blocks = build_text_blocks("xls", text);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].section, "Sheet 1: A");
        assert!(blocks[0].text.contains("R1C1=1"));
        assert_eq!(blocks[1].section, "Sheet 2: B");
    }

    #[test]
    fn builds_single_block_for_doc() {
        let blocks = build_text_blocks("doc", "line1\nline2");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].section, "doc");
        assert_eq!(blocks[0].text, "line1\nline2");
    }

    #[test]
    fn trims_doc_mojibake_tail_lines() {
        let lines = vec![
            "文章汽车没有准备国内对于".to_string(),
            "报告功能正在不过".to_string(),
            "通过决定解决信息参加".to_string(),
            "说明比较非常参加之后很多教育".to_string(),
            "耀(".to_string(),
            "瑯楨c".to_string(),
            "慃楬牢i".to_string(),
            "祰桴湯".to_string(),
            "瑯楨c".to_string(),
        ];
        let trimmed = trim_trailing_noise(lines, "doc");
        assert_eq!(trimmed.len(), 4);
        assert_eq!(trimmed.last().map(String::as_str), Some("说明比较非常参加之后很多教育"));
    }

    #[test]
    fn truncates_doc_tail_after_noise_anchor_or_streak() {
        let lines = vec![
            "正文第一段".to_string(),
            "正文第二段".to_string(),
            "耀(".to_string(),
            "瑯楨c".to_string(),
            "慃楬牢i".to_string(),
            "祰桴湯".to_string(),
            "0Table".to_string(),
            "Data".to_string(),
        ];
        let out = truncate_doc_tail_noise(lines, "doc");
        assert_eq!(out, vec!["正文第一段".to_string(), "正文第二段".to_string()]);
    }
}
