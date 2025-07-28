
import json
import re
from collections import Counter
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal, LTChar


def extract_spans(pdf_path):
    """
    Extracts text spans from a PDF, capturing font size, position, and boldness.

    Returns:
        spans: List of dicts with keys ['page', 'text', 'size', 'y0', 'bold']
    """
    spans = []
    for page_num, layout in enumerate(extract_pages(pdf_path), start=1):
        for element in layout:
            if not isinstance(element, LTTextBoxHorizontal):
                continue
            for line in element:
                if not isinstance(line, LTTextLineHorizontal):
                    continue
                chars = [c for c in line if isinstance(c, LTChar)]
                if not chars:
                    continue
                size = round(max(c.size for c in chars), 1)
                y0 = round(min(c.y0 for c in chars), 1)
                text = line.get_text().strip()
                if not text:
                    continue
                bold = any('Bold' in c.fontname for c in chars)
                spans.append({
                    'page': page_num,
                    'text': text,
                    'size': size,
                    'y0': y0,
                    'bold': bold
                })
    return spans


def classify_sizes(spans):
    """
    Classify font sizes into title, body, and heading levels.

    Returns:
      title_size: font size of the main title on page 1
      body_size: most common font size (body text)
      h_levels: dict mapping heading font sizes to 'H1', 'H2', ...
    """
    size_counts = Counter(s['size'] for s in spans)
    # Determine body size (most frequent)
    body_size = max(size_counts, key=size_counts.get)
    # Determine title size (largest on page 1)
    page1_sizes = [s['size'] for s in spans if s['page'] == 1]
    title_size = max(page1_sizes) if page1_sizes else body_size
    # Heading candidates: sizes > body_size on pages >1
    candidates = sorted(
        {s['size'] for s in spans if s['size'] > body_size and s['page'] > 1},
        reverse=True
    )
    h_levels = {size: f'H{i+1}' for i, size in enumerate(candidates)}
    return title_size, body_size, h_levels


def build_outline(spans, title_size, body_size, h_levels):
    """
    Build a high-level outline (titles only) of the document.

    Returns:
      title: The document title string
      outline: List of dicts with keys ['level', 'text', 'page']
    """
    # Extract title lines from page 1
    page1 = sorted([s for s in spans if s['page'] == 1], key=lambda s: -s['y0'])
    title_lines = [s['text'] for s in page1 if s['size'] == title_size]
    if not title_lines and page1:
        max_size = max(s['size'] for s in page1)
        title_lines = [s['text'] for s in page1 if s['size'] == max_size]
    title = "  ".join(title_lines).strip() + "  " if title_lines else ""
    # Collect headings
    raw = []
    for s in spans:
        if s['page'] == 1:
            continue
        txt = s['text'].strip()
        if not txt or not txt[0].isupper() or re.fullmatch(r"\d+", txt):
            continue
        lvl = h_levels.get(s['size'])
        if lvl:
            raw.append({'level': lvl, 'text': txt, 'page': s['page']})
    # Sort by page and y0
    raw.sort(key=lambda e: (e['page'], -next(s['y0'] for s in spans if s['text'].strip()==e['text'] and s['page']==e['page'])))
    # Merge split headings
    outline = []
    i = 0
    while i < len(raw):
        curr = raw[i]
        if i+1 < len(raw) and raw[i+1]['page']==curr['page'] and raw[i+1]['level']==curr['level']:
            nxt = raw[i+1]
            if curr['text'].endswith(':') or curr['text'][-1].islower():
                combined = f"{curr['text'].rstrip()} {nxt['text'].lstrip()}"
                outline.append({'level': curr['level'], 'text': combined.strip(), 'page': curr['page']})
                i += 2
                continue
        outline.append(curr)
        i += 1
    return title, outline


def build_sections(spans, title_size, body_size, h_levels):
    """
    Split spans into full sections with text bodies.

    Returns:
      sections: List of dicts {'title','level','page','texts'}
    """
    sections = []
    current = {'title': None, 'level': None, 'page': None, 'texts': []}
    for sp in sorted(spans, key=lambda x: (x['page'], -x['y0'])):
        size, txt, pg = sp['size'], sp['text'].strip(), sp['page']
        if size in h_levels:
            if current['title']:
                sections.append(current)
            current = {'title': txt, 'level': h_levels[size], 'page': pg, 'texts': []}
        else:
            if current['title']:
                current['texts'].append(txt)
    if current['title']:
        sections.append(current)
    return sections


if __name__ == '__main__':
    import sys
    if len(sys.argv)!=3:
        print("Usage: python extract_outline.py input.pdf output.json")
        sys.exit(1)
    spans=extract_spans(sys.argv[1])
    tsize, bsize, hlevels = classify_sizes(spans)
    title, outline = build_outline(spans, tsize, bsize, hlevels)
    result={'title':title,'outline':outline}
    with open(sys.argv[2],'w',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(f"Wrote outline to {sys.argv[2]}")

