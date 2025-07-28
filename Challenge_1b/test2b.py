import os
import json
import time
import argparse
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

# Import Round 1A utilities
from extract_outline import extract_spans, classify_sizes, build_sections


def load_encoder(model_path):
    """Load ONNX sentence-transformer model."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


def encode_texts(sess, texts, max_length=128):
    """
    Tokenize and encode texts into normalized embeddings.
    Returns array of shape (len(texts), hidden_size).
    """
    hidden_size = sess.get_outputs()[1].shape[-1]
    if not texts:
        return np.zeros((0, hidden_size))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2', local_files_only=True
    )
    batch = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors='np'
    )
    outputs = sess.run(
        None,
        {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
    )
    cls_emb = outputs[1]
    norms = np.linalg.norm(cls_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return cls_emb / norms


def rank_sections(sections, job_emb, encoder_sess, top_k=5):
    """
    Compute relevance scores and return top_k sections.
    """
    texts = [sec['title'] + ' ' + ' '.join(sec['texts']) for sec in sections]
    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not valid:
        return [], []
    idxs, docs = zip(*valid)
    embs = encode_texts(encoder_sess, list(docs))
    sims = cosine_similarity(embs, job_emb.reshape(1, -1)).flatten()
    weights = np.array([{'H1':1.0,'H2':0.9,'H3':0.8}.get(sections[i]['level'],0.7) for i in idxs])
    scores = sims * weights
    sel = np.argsort(-scores)[:top_k]
    ranked, chosen = [], []
    for rank, sel_idx in enumerate(sel, start=1):
        orig = idxs[sel_idx]
        sec = sections[orig]
        ranked.append({
            'document': sec['doc'],
            'page_number': sec['page'],
            'section_title': sec['title'],
            'importance_rank': rank
        })
        chosen.append(orig)
    return ranked, chosen


def analyze_subsections(sections, chosen_idxs, job_emb, encoder_sess):
    """
    Pick most relevant paragraph in each chosen section.
    """
    subs = []
    for idx in chosen_idxs:
        sec = sections[idx]
        paras = [p for p in '\n'.join(sec['texts']).split('\n') if p.strip()]
        if not paras:
            continue
        embs = encode_texts(encoder_sess, paras)
        sims = cosine_similarity(embs, job_emb.reshape(1, -1)).flatten()
        best = int(np.argmax(sims))
        subs.append({
            'document': sec['doc'],
            'page_number': sec['page'],
            'refined_text': paras[best].strip()
        })
    return subs


def main(input_dir, persona_path, job_path, output_path, model_path):
    start = time.time()
    # Load persona and job
    persona_json = json.load(open(persona_path, 'r', encoding='utf-8'))
    persona = persona_json.get('persona', '')
    proc_ts = persona_json.get('processing_timestamp', '')
    job_json = json.load(open(job_path, 'r', encoding='utf-8'))
    job_text = job_json.get('job_to_be_done') or job_json.get('task', '')

    # Encode job
    encoder = load_encoder(model_path)
    job_emb = encode_texts(encoder, [job_text])[0] if job_text.strip() else np.zeros((encoder.get_outputs()[1].shape[-1],))

    # Init result structure
    results = {
        'metadata': {
            'input_documents': [],
            'persona': persona,
            'job_to_be_done': job_text,
            'processing_timestamp': proc_ts
        },
        'extracted_sections': [],
        'subsection_analysis': []
    }

    # Process each PDF
    sections = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.pdf'):
            continue
        results['metadata']['input_documents'].append(fname)
        path = os.path.join(input_dir, fname)
        spans = extract_spans(path)
        tsize, bsize, h_levels = classify_sizes(spans)
        secs = build_sections(spans, tsize, bsize, h_levels)
        for s in secs:
            s['doc'] = fname
        sections.extend(secs)

    # Rank and analyze
    ranked, chosen = rank_sections(sections, job_emb, encoder)
    results['extracted_sections'] = ranked
    results['subsection_analysis'] = analyze_subsections(sections, chosen, job_emb, encoder)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"Processed {len(results['metadata']['input_documents'])} docs in {time.time()-start:.1f}s. Output -> {output_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Round 1B Document Intelligence')
    p.add_argument('--input', required=True)
    p.add_argument('--persona', required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='models/encoder.onnx')
    args = p.parse_args()
    main(args.input, args.persona, args.job, args.output, args.model)
