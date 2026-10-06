"""Extract manuscript figures and display tables without changing scientific content."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
from PIL import Image
import argparse,csv,hashlib,io,json,posixpath,re
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
sha=lambda b:hashlib.sha256(b).hexdigest()
def content(n):
    return ''.join(x.text or '' for x in n.iter()if x.tag in [f"{{{NS['w']}}}t",f"{{{NS['m']}}}t"])
def extract(source,out,supplement=False):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records=[];tables=[]
    with ZipFile(source)as z:
        root=E.fromstring(z.read('word/document.xml'));body=list(root.find('w:body',NS))
        rels={r.attrib['Id']:r.attrib['Target']for r in E.fromstring(z.read('word/_rels/document.xml.rels'))}
        if not supplement:
            for i,d in enumerate(root.findall('.//w:drawing',NS),1):
                blip=d.find('.//a:blip',NS);rid=blip.attrib[f"{{{NS['r']}}}embed"]
                part=posixpath.normpath(posixpath.join('word',rels[rid]));blob=z.read(part)
                im=Image.open(io.BytesIO(blob));fmt='jpg'if im.format=='JPEG'else im.format.lower()
                filename=f'figures/Fig{i:02d}.{fmt}';target=out/filename;target.parent.mkdir(parents=True,exist_ok=True)
                crop=d.find('.//a:srcRect',NS);rect=crop.attrib if crop is not None else {}
                crop_box=None;original=None
                if any(int(v)for v in rect.values()):
                    original=f'figures/embedded/Fig{i:02d}.{fmt}';p=out/original;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
                    w,h=im.size;crop_box=[round(w*int(rect.get('l',0))/100000),round(h*int(rect.get('t',0))/100000),round(w*(1-int(rect.get('r',0))/100000)),round(h*(1-int(rect.get('b',0))/100000))]
                    # Apply only the crop already recorded in Word; no scaling or retouching.
                    im=im.crop(crop_box);im.save(target)
                else:target.write_bytes(blob)
                caption=next(content(n)for n in body if re.match(rf'^Fig\.\s*{i}\.',content(n)))
                extent=d.find('.//wp:extent',NS)
                records.append(dict(figure=i,file=filename,caption=caption,embedded_part=part,embedded_sha256=sha(blob),extracted_sha256=sha(target.read_bytes()),pixels=list(im.size),word_extent_EMU=extent.attrib if extent is not None else {},word_crop_100000=rect,crop_pixels=crop_box,embedded_original=original))
            assert len(records)==9
            (out/'figures/manifest.json').write_text(json.dumps(dict(source_docx=source.name,source_sha256=sha(source.read_bytes()),figures=records),indent=2),encoding='utf-8')
        last=''
        for node in body:
            if node.tag==f"{{{NS['w']}}}p":
                if content(node):last=content(node)
                continue
            if node.tag!=f"{{{NS['w']}}}tbl":continue
            match=re.match(r'^Table\s+(S?\d+)',last)
            notation=supplement and last.startswith('Notation.')
            if not match and not notation:continue
            label=match[1]if match else'Notation'
            if '(continued)'in last:label+='b'
            if label=='S21':label='S21a'
            folder='supplementary_tables'if supplement else'manuscript_tables'
            file=f'source_data/{folder}/Table{label}.csv';dest=out/file;dest.parent.mkdir(parents=True,exist_ok=True)
            rows=[[content(c)for c in row.findall('w:tc',NS)]for row in node.findall('w:tr',NS)]
            with dest.open('w',encoding='utf-8-sig',newline='')as f:csv.writer(f).writerows(rows)
            tables.append(dict(table=label,caption=last,file=file,rows=len(rows)-1,columns=len(rows[0]),sha256=sha(dest.read_bytes())))
            last=''
        assert len(tables)==(25 if supplement else 7),len(tables)
        dest=out/'source_data'/('supplementary_tables'if supplement else'manuscript_tables')/'manifest.json'
        dest.write_text(json.dumps(dict(source_docx=source.name,source_sha256=sha(source.read_bytes()),display_values_not_unrounded_results=True,tables=tables),indent=2),encoding='utf-8')
    return records,tables
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manuscript',type=Path);p.add_argument('--supplement',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    extract(a.manuscript,a.out)
    if a.supplement:extract(a.supplement,a.out,True)
