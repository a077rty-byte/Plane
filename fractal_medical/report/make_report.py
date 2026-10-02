from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

d = Document()
for s in d.sections:
    s.left_margin = s.right_margin = Cm(2); s.top_margin = s.bottom_margin = Cm(2)
st = d.styles['Normal']; st.font.name = 'Arial'; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn('w:cs'), 'Arial')

def rtl(p):
    pPr = p._p.get_or_add_pPr(); b = OxmlElement('w:bidi'); pPr.append(b)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in p.runs:
        rPr = r._r.get_or_add_rPr(); e = OxmlElement('w:rtl'); rPr.append(e)

def para(t, bold=False, size=None):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = bold
    if size: r.font.size = Pt(size)
    rtl(p); return p

def head(t):
    p = para(t, True, 14); return p

def bullet(t):
    p = d.add_paragraph(style='List Bullet'); p.add_run(t); rtl(p)

def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement('w:shd')
    sh.set(qn('w:val'), 'clear'); sh.set(qn('w:fill'), color); tcPr.append(sh)

def table(title, header, rows):
    p = d.add_paragraph(); r = p.add_run(title); r.bold = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = d.add_table(rows=1, cols=len(header)); t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ''; rr = c.paragraphs[0].add_run(h); rr.bold = True
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER; shade(c, 'D9E2F3')
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''; rr = cells[i].paragraphs[0].add_run(str(v))
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if row[0] == 'Avg': rr.bold = True; shade(cells[i], 'F2F2F2')
    d.add_paragraph()

H = ['Sample MRI Image', 'τ = 10^-3', 'τ = 10^-4', 'τ = 10^-5', 'τ = 10^-6']
T1 = [[1,41.66,40.93,40.34,39.18],[2,42.41,41.86,40.88,39.70],[3,45.56,42.37,38.00,32.09],[4,42.57,41.83,40.57,38.78],['Avg',43.05,41.75,39.95,37.44]]
T2 = [[1,2.11,2.08,2.85,2.43],[2,2.64,2.69,3.40,3.25],[3,4.62,6.37,6.20,10.87],[4,2.93,3.14,3.94,3.56],['Avg',3.08,3.57,4.10,5.02]]
T3 = [[1,5.98,7.72,10.03,13.88],[2,4.48,5.55,7.36,9.93],[3,2.00,2.19,2.51,3.15],[4,3.49,4.10,5.22,7.41],['Avg',3.99,4.89,6.28,8.59]]
T4 = [['PSNR (dB)',32.58,41.12,40.57],['Compression Time (sec)',42.24,6.03,3.85],['Compression Ratio',17.66,5.35,5.22]]

para('نتائج تطبيق الترميز الكسوري المعتمد على الشبكة العصبية (SOM) لضغط الصور الطبية', True, 17)
para('تنفيذ برمجي (Python) لبحث: Neural based domain and range pool partitioning using Fractal Coding for nearly lossless Medical Image Compression — S. Bhavani & K. Thanushkodi, WSEAS Trans. on Signal Processing, 2013')

head('1. إعدادات التجربة')
bullet('البيانات: مجموعة صور رنين مغناطيسي للدماغ (MRI) تضم 7200 صورة؛ اختيرت صورة عشوائية واحدة من كل فئة (4 صور) بذرة عشوائية ثابتة (seed=1)، وجرى تحويل كل صورة إلى تدرج رمادي بحجم 512×512 كما في البحث.')
bullet('الجداول 1 و2 و3 محسوبة بالخوارزمية المقترحة II (الترميز الكسوري السريع مع SOM) عند أربع قيم للعتبة τ.')
bullet('الجدول 4 يقارن الخوارزميات الثلاث (القياسية، المقترحة I، المقترحة II) على الصورة الرابعة عند τ = 10^-5.')
bullet('المقاييس: PSNR (جودة الصورة المسترجعة، كلما زاد كان أفضل)، زمن الترميز بالثواني، ونسبة الضغط = الحجم الأصلي ÷ الحجم المضغوط.')

head('2. الجداول')
table('Table 1  PSNR at Different Levels of Compression (dB)', H, T1)
table('Table 2  Time Taken at Different Levels of Compression (sec)', H, T2)
table('Table 3  Compression Ratio at Different Thresholds', H, T3)
table('Table 4  PSNR achieved for Different Algorithms', ['Metric', 'Standard Fractal Encoding', 'Proposed Algorithm I (τ=1e-5)', 'Proposed Algorithm II (τ=1e-5)'], T4)

head('3. شرح محتوى النتائج')
para('أ) المخرجات الأولية (الأسطر التي تبدأ بـ image ... tau)', True)
para('كل سطر هو تشغيل واحد للخوارزمية II على صورة معينة بعتبة معينة، ويعرض: رقم الصورة، قيمة العتبة τ، قيمة PSNR، زمن الترميز، نسبة الضغط CR، وعدد كتل الـ domain (domains) التي حُفظت دون فقد (seeds). مثال: الصورة 4 عند τ=1e-3 أعطت PSNR=42.57 dB وزمن 2.93 ثانية ونسبة ضغط 3.49 و1183 كتلة domain. وعند τ=1e-6 انخفض عدد الكتل إلى 300.')
para('ب) الجدول 1 (PSNR)', True)
para('كلما صغرت العتبة τ قلّ عدد كتل الـ domain المحفوظة دون فقد، فيزداد الجزء المولَّد بالتقريب الكسوري ويقل PSNR. المتوسط ينخفض من 43.05 dB عند 10^-3 إلى 37.44 dB عند 10^-6. جميع القيم عند 10^-3 و10^-4 أعلى من 40 dB، وهذا ما يبرر وصف الطريقة بأنها شبه عديمة الفقد. الصورة 3 تفقد جودة أكثر من غيرها (45.56 إلى 32.09 dB)؛ أي أن حساسيتها للعتبة كبيرة.')
para('ج) الجدول 2 (زمن الترميز)', True)
para('الأزمنة بين 2 و11 ثانية، وتميل إلى الزيادة مع تصغير τ (المتوسط من 3.08 إلى 5.02 ثانية)، والسبب أن قلة كتل الـ domain تجعل عدداً أكبر من كتل الـ range يُقسَّم إلى كتل أصغر ويُعاد البحث. هذا يختلف عن البحث الأصلي الذي ذكر نقصان الزمن مع نقصان τ. الصورة 3 هي الأبطأ (10.87 ثانية عند 10^-6) لأن تفاصيلها تتطلب تقسيماً أكثر.')
para('د) الجدول 3 (نسبة الضغط)', True)
para('تزداد نسبة الضغط بوضوح مع تصغير τ لأن حجم الجزء المحفوظ دون فقد يقل: المتوسط من 3.99 إلى 8.59. أعلى نسبة 13.88 (الصورة 1 عند 10^-6) وأقل نسبة 2.00 (الصورة 3 عند 10^-3). فهناك مفاضلة واضحة: نسبة ضغط أعلى مقابل جودة أقل.')
para('هـ) الجدول 4 (مقارنة الخوارزميات الثلاث)', True)
bullet('الجودة: الخوارزمية القياسية الأقل (32.58 dB)، والمقترحتان أعلى بحوالي 8 إلى 8.5 dB (41.12 و40.57) لأن المناطق الغنية بالتفاصيل تُحفظ بدقة كاملة.')
bullet('الزمن: القياسية 42.24 ثانية، المقترحة I نحو 6.03 ثانية (أسرع بنحو 7 مرات)، والمقترحة II نحو 3.85 ثانية (أسرع بنحو 11 مرة). فتجميع الكتل بالـ SOM يقلل مساحة البحث.')
bullet('ثمن السرعة: انخفاض PSNR بين I وII بمقدار 0.55 dB فقط، ونسبة الضغط شبه متساوية (5.35 مقابل 5.22).')
bullet('نسبة الضغط: القياسية الأعلى هنا (17.66) بخلاف البحث الأصلي الذي ذكر أنها الأقل، لأن الطريقتين المقترحتين تخزنان كتل الـ domain دون فقد فتبقى نسبة الضغط أقل مقابل جودة أعلى.')

head('4. ملاحظات منهجية')
bullet('البحث الأصلي أجرى حساباته على صور MRI سريرية خاصة (غير متاحة)، لذلك الأرقام هنا لا تطابق أرقامه، لكن الاتجاه العام (سرعة أكبر وجودة أعلى للمقترحتين) متحقق.')
bullet('شرط فصل الـ domain عن الـ range كما هو مطبوع في البحث يتعارض مع الشكل 2 فيه؛ اعتُمد تحويل يوافق الشكل 2 (الخيار --rule figure).')
bullet('معاملات الضغط (نسبة الضغط) محسوبة من عدد بتات التحويلات (فهرس الـ domain والتماثل ومعاملا s و o) مضافاً إليها حجم كتل الـ seed بعد ضغط zlib، وهي تقدير برمجي لا ملف مضغوط فعلي.')
bullet('اختيار صورة واحدة عشوائية لكل فئة يعني أن المتوسطات مبنية على 4 صور فقط، ولا تصلح لاستنتاجات إحصائية عامة.')
d.save('/home/user/Plane/fractal_medical/report/Fractal_MRI_Results.docx')
