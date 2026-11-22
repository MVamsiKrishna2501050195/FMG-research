from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from lxml import etree

DOC=Path('manuscript/TCSST_IRO_FMG_revision_R11_final.docx')
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'; M='http://schemas.openxmlformats.org/officeDocument/2006/math'; N={'w':W,'m':M}; q=lambda n,t:f'{{{n}}}{t}'
def mr(s):
 r=etree.Element(q(M,'r')); t=etree.SubElement(r,q(M,'t')); t.text=s; return r
def sub(b,i):
 x=etree.Element(q(M,'sSub')); e=etree.SubElement(x,q(M,'e')); e.append(mr(b)); z=etree.SubElement(x,q(M,'sub')); z.append(mr(i)); return x
def sup(b,i):
 x=etree.Element(q(M,'sSup')); e=etree.SubElement(x,q(M,'e')); e.append(mr(b)); z=etree.SubElement(x,q(M,'sup')); z.append(mr(i)); return x
def o(*parts):
 x=etree.Element(q(M,'oMath'))
 for p in parts: x.append(mr(p) if isinstance(p,str) else p)
 return x

def eq2a(): return o('ΔR(ε) = R(',sub('g','0'),', clean) − R(',sub('g','0'),', masked)')
def eq6(): return o(sup('I','R'),' (δ) = 𝟙[ΔR(ε) ≥ δ],   ',sup('I','A'),' (δ) = 𝟙[ΔA(ε) ≥ δ],   U(δ) = ',sup('I','R'),' (δ)(1 − ',sup('I','A'),' (δ))')
def eq10(): return o(sub('C','A'),' (δ) = [',sub('N','R'),' (δ) − Σ U(δ)] / ',sub('N','R'),' (δ)')
def eq12(): return o(sub('τ','g'),' (q) = E[',sub('φ','h'),' (X) | G = g, Y = 1] − ',sub('E','M∼q(·|X,g)'),'[',sub('φ','h'),' (',sup('T','M'),' (X)) | G = g, Y = 1]')
builders={2:eq2a,7:eq6,11:eq10,13:eq12}
with ZipFile(DOC) as z:
 root=etree.fromstring(z.read('word/document.xml')); eqs=root.xpath('.//m:oMath',namespaces=N)
 for i,b in builders.items():
  old=eqs[i-1]; old.getparent().replace(old,b())
 xml=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone='yes'); tmp=DOC.with_suffix('.r26.docx')
 with ZipFile(tmp,'w',ZIP_DEFLATED) as out:
  for info in z.infolist(): out.writestr(info,xml if info.filename=='word/document.xml' else z.read(info.filename))
tmp.replace(DOC); print('Corrected Eq. 2a, 6, 10, and 12')
