from zipfile import ZipFile,ZIP_DEFLATED
from pathlib import Path
from lxml import etree

F=Path('manuscript/TCSST_IRO_FMG_revision_R11_final.docx'); W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'; M='http://schemas.openxmlformats.org/officeDocument/2006/math'; N={'w':W,'m':M}; q=lambda a,b:f'{{{a}}}{b}'
nums=['(1)','(2a)','(2b)','(3)','(4)','(5)','(6)','(7)','(8)','(9)','(10)','(11)','(12)','(13)','(14)','(15)','(16)']
def run(s):
 r=etree.Element(q(M,'r')); t=etree.SubElement(r,q(M,'t')); t.text=s; return r
def math(s):
 o=etree.Element(q(M,'oMath')); i=0
 while i<len(s):
  j=s.find('_',i)
  if j<0: o.append(run(s[i:])); break
  k=j-1
  while k>=i and (s[k].isalnum() or s[k] in 'Δαπτ'): k-=1
  if k==j-1: o.append(run(s[i:j+1])); i=j+1; continue
  if k>i: o.append(run(s[i:k+1]))
  ss=etree.SubElement(o,q(M,'sSub')); e=etree.SubElement(ss,q(M,'e')); e.append(run(s[k+1:j])); sub=etree.SubElement(ss,q(M,'sub')); sub.append(run(s[j+1:j+4])); i=j+4
  o.append(ss)
 return o
def text(p): return ''.join(p.xpath('.//w:t/text()',namespaces=N))
with ZipFile(F) as z:
 root=etree.fromstring(z.read('word/document.xml'))
 for p in root.xpath('.//w:p',namespaces=N):
  s=''.join(p.xpath('.//m:oMath//m:t/text()',namespaces=N))+text(p); n=next((x for x in nums if x in s),None)
  if n and (s.lstrip().startswith(n) or s.rstrip().endswith(n)):
   formula=s.split(n,1)[1].strip() if s.lstrip().startswith(n) else s.rsplit(n,1)[0].strip()
   for c in list(p):
    if c.tag in (q(W,'r'),q(M,'oMath')): p.remove(c)
   p.append(math(formula)); r=etree.SubElement(p,q(W,'r')); rp=etree.SubElement(r,q(W,'rPr')); hi=etree.SubElement(rp,q(W,'highlight')); hi.set(q(W,'val'),'yellow'); t=etree.SubElement(r,q(W,'t')); t.set('{http://www.w3.org/XML/1998/namespace}space','preserve'); t.text='\t'+n
 major={'Introduction':1,'Literature Review and Research Gap':2,'Methodology':3,'Results':4,'Discussion':5,'Limitations and Threats to Validity':6,'Future Work':7,'Conclusion':8}; h1=h2=h3=0
 for p in root.xpath('.//w:p',namespaces=N):
  st=''.join(p.xpath('./w:pPr/w:pStyle/@w:val',namespaces=N)); s=text(p).strip()
  if st in ('Heading1','Heading 1') and s in major:
   h1=major[s]; h2=h3=0
   for t in p.xpath('.//w:t',namespaces=N):
    if t.text==s: t.text=f'{h1}. {s}'
  elif st in ('Heading2','Heading 2') and h1:
   h2+=1
   for t in p.xpath('.//w:t',namespaces=N):
    if t.text==s: t.text=f'{h1}.{h2} {s}'
  elif st in ('Heading3','Heading 3') and h1 and h2:
   h3+=1
   for t in p.xpath('.//w:t',namespaces=N):
    if t.text==s: t.text=f'{h1}.{h2}.{h3} {s}'
 fixes={'Theory Extension: Masking Intervention Formulation':'5. Theory Extension: Masking Intervention Formulation','5. Discussion':'6. Discussion','6. Limitations and Threats to Validity':'7. Limitations and Threats to Validity','7. Future Work':'8. Future Work','8. Conclusion':'9. Conclusion'}
 for t in root.xpath('.//w:t',namespaces=N):
  if t.text in fixes: t.text=fixes[t.text]
 for p in root.xpath('.//w:p',namespaces=N):
  s=text(p)
  if s=='Theory Extension: Masking Intervention Formulation':
   for t in p.xpath('.//w:t',namespaces=N):
    if t.text and t.text.startswith('Theory Extension:'): t.text='5. '+t.text; break
  if s.startswith('5.1 Missingness-aware training comparator'):
   for t in p.xpath('.//w:t',namespaces=N):
    if t.text and t.text.startswith('5.1 '): t.text='6.1 '+t.text[4:]; break
 xml=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone='yes'); tmp=F.with_suffix('.tmp.docx')
 with ZipFile(tmp,'w',ZIP_DEFLATED) as out:
  for i in z.infolist(): out.writestr(i,xml if i.filename=='word/document.xml' else z.read(i.filename))
tmp.replace(F)
print('R24 equations and headings fixed')
