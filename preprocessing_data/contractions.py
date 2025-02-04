from cltk.tokenizers.lat.lat import LatinWordTokenizer
from string import punctuation
from cltk.alphabet.lat import normalize_lat
from cltk.stops.lat import STOPS
import pycld2 as cld2
import re


def tokenize_clean(text):
    global contractions
    source = text
    
    text = re.sub(r"[-_] ", "", text)
    # re.sub(r"\(.{0,6}[\.,].{0,6}[\.,].{0,6}\)", "", text)
    for i in contractions:
        text = re.sub(fr"[ \(]{i} ", f" {contractions[i]} ", text)
    text = re.sub(r"[0-9]*", "", text)
    # pattern = r"\b(?=[MDCLXVIΙ])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})([IΙ]X|[IΙ]V|V?[IΙ]{0,3})\b\.?"
    # text = re.sub(pattern, '', text)
    text = "".join(x for x in text if x.isascii())

    _, _, _, detected_language = cld2.detect(text,  returnVectors=True)
    # print(detected_language)
    cleaned_text = []
    for langs in detected_language:
        if langs[2] not in ["Unknown", "Latin"]:
            cleaned_text.append(text[langs[0]:langs[0]+langs[1]])

    text = normalize_lat(text, True, True, True, True)
    # re.sub(r'[^\w\s\.,:;\!\?]', "", text)
    tokenizer = LatinWordTokenizer()
    tokens = [re.sub(r'[^\w\s\.,:;\!\?]', '', x.lower()) for x in tokenizer.tokenize(text)]
    clean_text = " ".join(t for t in tokens)
    cleaned_text_sub = re.sub(r'\s+([.,:;\!\?])', r'\1', clean_text)

    return cleaned_text_sub
    # return text

contractions = {'A.D.': 'anno Domini',
                 'A.I.': 'ad interim',
                 'a.m.': 'ante meridiem',
                 'ca.': 'circa',
                 'c.': 'circa',
                 'Cap.': 'capitulus',
                 'cf.': 'confer',
                 'C. P.': 'ceteris paribus',
                 'C. V.': 'curriculum vitae',
                 'cwt.': 'centum weight',
                 'D.V.': 'Deo volente',
                 'D.G.': 'Dei gratia',
                 'ead.': 'eadem',
                 'etc.': 'et cetera',
                 'fac.': 'ex post facto',
                 'fl.': 'floruit',
                 'f.': 'floruit',
                 'ibid.': 'ibidem',
                 'id.': 'idem',
                 'J.D.': 'Juris Doctor',
                 'lb.': 'Juris Doctor',
                 'lbs.': 'libra',
                 'LL.B.': 'Legum Baccalaureus',
                 'M.A.': 'Magister Artium',
                 'M.O.': 'modus operandi',
                 'N.B.': 'nota bene',
                 'nem. con.': 'nemine contradicente',
                 'op. cit.': 'opere citato',
                 'P.A.': 'per annum',
                 'per cent.': 'per centum',
                 'Ph.D.': 'Philosophiae Doctor',
                 'p.m.': 'post meridiem',
                 'P.M.A.': 'post mortem auctoris',
                 'P.P.': 'post mortem auctoris',
                 'per pro.': 'per procurationem',
                 'P.R.N.': 'pro re nata',
                 'pro tem.': 'pro tempore',
                 'P.S.': 'post scriptum',
                 'P.P.S.': 'post post scriptum',
                 'Q.D.': 'quaque die',
                 'Q.E.D.': 'quod erat demonstrandum',
                 'q.v.': 'quod vide',
                 'qq.v.': 'quod vide',
                 're': 'in re',
                 'Reg.': 'regina',
                 'r.': 'regnavit',
                 'R.I.P.': 'requiescat in pace',
                 'S.A.': 'sensu amplo',
                 'sc.': 'sensu amplo',
                 'scil.': 'scilicet',
                 'S.O.S.': 'si opus sit',
                 'sic': 'sic erat scriptum',
                 'viz.': 'videlicet',
                 'vs.': 'videlicet',
                 'v.': 'versus',
                 'A.B.': 'Artium Baccalaureus',
                 'a.C.n.': 'ante Christum natum',
                 'A.M.D.G.': 'ad maiorem Dei gloriam',
                 'An. Sal.': 'Anno Salutis',
                 'a.u.': 'anno urbis',
                 'B.A.': 'Artium Baccalaureus',
                 'Ben': 'Benedictus',
                 'c': 'cum',
                 'CC.': 'Civis in plural',
                 'D.D.': 'Divinitatis Doctor',
                 'D.Lit.': 'Divinitatis Doctor',
                 'D.Litt.': 'Doctor Litterarum',
                 'D.M.': 'Doctor Medicinae',
                 'D.M.D.': 'Dentae Medicinae Doctor',
                 'D.Phil.': 'Doctor Philosophiæ',
                 'D.Sc.': 'Doctor Scientiae',
                 'DSP or D. S. P.': 'decessit sine prole',
                 'D.Th.': 'Doctor Theologiae',
                 'Ed.D.': 'Educationae Doctor',
                 'et seq.': 'Educationae Doctor',
                 'et seqq.': 'Educationae Doctor',
                 'et sequa.': 'et sequens',
                 'et ux.': 'et uxor',
                 'dwt.': 'denarius weight',
                 'F.D.': 'fidei defensor',
                 'FID.DEF.': 'fidei defensor',
                 'I.N.D.F.S.S.A.': 'In Nomine Domini Filii Spiritus Sancti Amen',
                 'in litt.': 'in litteris',
                 'inst.': 'instante mense',
                 'J.S.D.': 'Juridicae Scientiae Doctor',
                 'Lit.D.': 'Juridicae Scientiae Doctor',
                 'Litt.D.': 'Litterarum Doctor',
                 'Ll.D.': 'Legum Doctor',
                 'Ll.M.': 'Legum Magister',
                 'loq.': 'loquitur',
                 'M.D.': 'Medicinae Doctor',
                 'm.m.': 'mutatis mutandis',
                 'N.I.A.': 'In Nomine Iesus Amen',
                 'N.N.': 'nomen nescio',
                 'Nob.': 'nobis',
                 'O.D.': 'Optometriae Doctor',
                 'O.H.S.S.': 'Ossa hic sita sunt',
                 'O.S.': 'oculus sinister',
                 'O.U.': 'oculus uterque',
                 'per mil.': 'per mille',
                 'prox.': 'proximo mense',
                 'Q.D.B.V.': 'quod deus bene vertat',
                 'Q.E.C.': 'quod erat construendum',
                 'Q.E.F.': 'quod erat faciendum',
                 'Q.E.I.': 'quod erat inveniendum',
                 's': 'sine',
                 'S.': 'Sanctus',
                 'S.C.S': 'Sanctus',
                 'S.C.S.D.X': 'Sanctus Dominus Christus',
                 'S.D.X': 'Sanctus Dominus Christus',
                 'S.D.I.X': 'Salvator Dominus Iesus Christus',
                 'S.J.D.': 'Scientiae Juridicae Doctor',
                 'Sc.D.': 'Scientiae Doctor',
                 'sphalm.': 'sphalma typographicum',
                 'S.P.Q.R.': 'Senatus Populusque Romanus',
                 'sqq.': 'sequentia',
                 'S.S. Theol.': 'Sacrosanctae Theologiae',
                 'S.T.T.L.': 'sit tibi terra levis',
                 's.v.': 'sub verbo',
                 'S.V.B.E.E.V.': 'si vales bene est ego valeo',
                 'Th.D.': 'Theologiae Doctor',
                 'ult.': 'ultimo mense[1]',
                 'u.s.': 'ut supra',
                 'V.C.': 'vi coactus',
                 'V.I.': 'Venerate Iesum',
                 'v.i.': 'vide infra',
                 'v.s.': 'vide supra',
                 'C.': 'Caius',
                 'G.': 'Gaius',
               'Cn.': 'Cnaius',
               'Gn.': 'Gnaius',
                'Cor.': 'Cornelius',
                'Cic.': 'Cicero',
                'Joan.': 'Joannes',
                'Philip.': 'Philippus',
                'Psal.': 'Psalmus',
                 'Matth.': 'Matthaeus',
                 'Petr.': 'Petrus', 'Pet.': 'Petrus',
                 'Jac.': 'Jacobus',
                 'Tit.': 'Titus',
                 'Luc.': 'Lucius',
                 'Cass.': 'Cassius',
                 'Paul': 'Paulus',
                 'Mar': 'Marcus', 'Marc': 'Marcus',
                 'Greg.': 'Gregorius',
                 'Hist.': 'Historia',
                 'pag.': 'pagina',
                 'n.': ''
                }