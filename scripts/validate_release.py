"""Validate file integrity and the published primary results; Python standard library only."""
from pathlib import Path
import csv,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for name,item in manifest['files'].items():
        p=ROOT/name;assert p.exists(),name;assert p.stat().st_size==item['bytes']and sha(p)==item['sha256'],name
    primary=rows(ROOT/'artifacts/coco_yolo/main_metrics.csv')
    raw=next(r for r in primary if r['method']=='Raw'and r['split']=='test');rf=next(r for r in primary if r['method']=='RF'and r['split']=='test')
    assert [int(raw['unknown_false_accept_objects']),int(rf['unknown_false_accept_objects'])]==[299,176]
    assert [int(raw['background_false_accept_count']),int(rf['background_false_accept_count'])]==[2875,2828]
    assert round(float(raw['precision_recall_unknown_balanced_score']),4)==.7729
    assert round(float(rf['precision_recall_unknown_balanced_score']),4)==.7789
    figures=json.loads((ROOT/'figures/manifest.json').read_text());assert len(figures['figures'])==9
    for fig in figures['figures']:assert sha(ROOT/fig['file'])==fig['extracted_sha256']
    for folder,count in [('manuscript_tables',7),('supplementary_tables',25)]:
        tables=json.loads((ROOT/'source_data'/folder/'manifest.json').read_text());assert len(tables['tables'])==count
        for t in tables['tables']:assert sha(ROOT/t['file'])==t['sha256']
    crowd=rows(ROOT/'source_data/round5_20261006/crowd_BG_summary.csv')
    assert [int(r['inside_crowd_box'])for r in crowd]==[1295,1307,1324,1318]
    ratings=rows(ROOT/'source_data/manual_review_20261006/ratings_anonymous.csv');assert len(ratings)==150 and all(r['reviewer']=='Author'and r['category']for r in ratings)
    print(json.dumps(dict(passed=True,verified_files=len(manifest['files']),figures=9,main_tables=7,supplementary_tables_and_notation=25,author_ratings=150)))
if __name__=='__main__':main()
