from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from lxml import etree

DOC = Path('manuscript/TCSST_IRO_FMG_revision_R11_final.docx')
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
N={'w':W,'m':M}
q=lambda ns,name:f'{{{ns}}}{name}'

def mr(s):
    r=etree.Element(q(M,'r')); t=etree.SubElement(r,q(M,'t')); t.text=s; return r
def sub(base, index):
    x=etree.Element(q(M,'sSub')); e=etree.SubElement(x,q(M,'e')); e.append(mr(base)); z=etree.SubElement(x,q(M,'sub')); z.append(mr(index)); return x
def sup(base, index):
    x=etree.Element(q(M,'sSup')); e=etree.SubElement(x,q(M,'e')); e.append(mr(base)); z=etree.SubElement(x,q(M,'sup')); z.append(mr(index)); return x
def add(o,*parts):
    for part in parts: o.append(mr(part) if isinstance(part,str) else part)

def eq1():
    o=etree.Element(q(M,'oMath')); add(o,'Pr(at least one masked feature) = 1 − '); add(o,sup('(1 − ε)','p')); return o
def eq7():
    o=etree.Element(q(M,'oMath')); add(o,sub('FMG','all'),' = ',sub('ΔTPR','all'),' − ΔA = (1 − π)(',sub('ΔTPR','all'),' − ',sub('ΔTNR','all'),')'); return o
def eq8():
    o=etree.Element(q(M,'oMath')); add(o,'ΔA(π) = ',sub('ΔTNR','all'),' + π[',sub('ΔTPR','all'),' − ',sub('ΔTNR','all'),']'); return o
def eq9():
    o=etree.Element(q(M,'oMath')); add(o,sub('ΔTPR','all'),' = ',sub('α','0'),sub('ΔTPR','0'))
    # rebuild the remaining expression explicitly
    o=etree.Element(q(M,'oMath')); add(o,sub('ΔTPR','all'),' = ')
    add(o,sub('α','0')); add(o,sub('ΔTPR','0')); add(o,' + '); add(o,sub('α','1')); add(o,sub('ΔTPR','1')); add(o,',   '); add(o,sub('α','g')); add(o,' = Pr(G = g | Y = 1),   '); add(o,sub('α','0')); add(o,' + '); add(o,sub('α','1')); add(o,' = 1'); return o
def eq11():
    o=etree.Element(q(M,'oMath')); add(o,sub('φ','h'),' (x) = 𝟙[h(x) ≥ t]'); return o
def eq12():
    o=etree.Element(q(M,'oMath')); add(o,sub('τ','g'),' (q) = E'); x=etree.Element(q(M,'sSub')); e=etree.SubElement(x,q(M,'e')); e.append(mr('M')); z=etree.SubElement(x,q(M,'sub')); z.append(mr('∼q(·|X,g)')); o.append(x); add(o,'[',sub('φ','h'),' (X) | G = g, Y = 1] − E'); x=etree.Element(q(M,'sSub')); e=etree.SubElement(x,q(M,'e')); e.append(mr('M')); z=etree.SubElement(x,q(M,'sub')); z.append(mr('∼q(·|X,g)')); o.append(x); add(o,'[',sub('φ','h'),' (Tᴹ(X)) | G = g, Y = 1]'); return o
def eq13():
    o=etree.Element(q(M,'oMath')); add(o,'TV(P, Q) = '); add(o,sup('sup','B')); add(o,' |P(B) − Q(B)| = ½'); add(o,sub('‖P − Q‖','1')); return o

builders={1:eq1,8:eq7,9:eq8,10:eq9,12:eq11,13:eq12,14:eq13}
with ZipFile(DOC) as zin:
    root=etree.fromstring(zin.read('word/document.xml')); equations=root.xpath('.//m:oMath',namespaces=N)
    for idx,builder in builders.items():
        old=equations[idx-1]; parent=old.getparent(); parent.replace(old,builder())
    for p in root.xpath('.//w:p',namespaces=N):
        text=''.join(p.xpath('.//w:t/text()',namespaces=N))
        if text=='3.4.1 Algorithm 1. Focus-Group Masking Gap (FMG) Frozen-Classifier Audit':
            for t in p.xpath('.//w:t',namespaces=N):
                if t.text: t.text=t.text.replace('3.4.1 ','',1); break
        if text.startswith('[5] Kochs,'):
            for t in p.xpath('.//w:t',namespaces=N):
                if t.text and 'In CEUR Workshop Proceedings' in t.text:
                    t.text=t.text.replace('In CEUR Workshop Proceedings (Vol. 4147).','In Proceedings of the 3rd Workshop on Fairness and Bias in AI (AEQUITAS@ECAI 2025), CEUR Workshop Proceedings, Vol. 4147.')
    xml=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone='yes'); tmp=DOC.with_suffix('.r25.docx')
    with ZipFile(tmp,'w',ZIP_DEFLATED) as zout:
        for info in zin.infolist(): zout.writestr(info,xml if info.filename=='word/document.xml' else zin.read(info.filename))
tmp.replace(DOC); print('R25 equation structures, reference [5], and algorithm heading fixed')
