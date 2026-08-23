#!/usr/bin/env python3
"""
논문 PDF에서 Figure/Table을 캡션 기준으로 자동 추출한다.

사용법
    python3 extract_figs.py <pdf> <pages> <dpi> <outdir> <prefix>
    예)  python3 extract_figs.py paper.pdf 1-35 150 ./assets mixlm

원리
  1) pdftohtml -xml 로 각 페이지의 텍스트 조각을 좌표·폰트와 함께 얻어 '줄'로 묶는다.
  2) 캡션("Figure 3: ...", "Figure 7 | ...")을 찾고, 이어지는 줄을 캡션 블록으로 묶는다.
  3) '본문 줄'을 식별한다 — 본문 여백에 정렬되고, 조각이 촘촘하며(fill),
     본문 폰트 크기 이상인 줄. 그림 내부 라벨은 대개 더 작은 폰트라 걸러진다.
  4) 그림의 세로 범위 = (캡션 위쪽의 가장 가까운 본문 줄·다른 캡션·페이지번호의 끝)
     ~ (캡션의 시작).  Table 은 캡션이 위에 오므로 방향을 뒤집는다.
  5) 렌더 이미지에서 그 범위를 자르고 잉크 기준으로 정밀 트리밍한다.

벡터/래스터를 가리지 않는다 — 렌더 결과를 자르기 때문이다.
필요: poppler(pdftohtml, pdftoppm), Pillow
"""
import re, os, sys, subprocess, html, json
from collections import Counter

CAP_RE = re.compile(r'^(Figure|Table|Fig\.?)\s*(\d+)\s*[:.|–—-]\s')
CAP_ANY = re.compile(r'(?:(?<=\s)|^)(Figure|Table|Fig\.?)\s*(\d+)\s*[:.|–—-]\s')


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def parse_doc(pdf):
    """PDF 전체를 한 번에 파싱해 페이지별 텍스트 조각과 '문서 수준' 기준을 얻는다.

    본문 좌측 여백과 본문 폰트 크기는 문서의 성질이지 페이지의 성질이 아니다.
    표나 코드 박스가 지배하는 페이지에서 페이지 단위로 추정하면 그 열 위치가
    최빈값이 되어버리므로, 문서 전체에서 한 번만 추정한다.
    """
    xml = sh(['pdftohtml', '-xml', '-i', '-stdout', pdf])
    fsize = {m.group(1): int(m.group(2))
             for m in re.finditer(r'<fontspec id="(\d+)" size="(\d+)"', xml)}
    pages = {}
    for pm in re.finditer(
            r'<page number="(\d+)"[^>]*height="(\d+)" width="(\d+)"[^>]*>(.*?)</page>', xml, re.S):
        no, page_h, page_w, body = int(pm.group(1)), int(pm.group(2)), int(pm.group(3)), pm.group(4)
        els = []
        for m in re.finditer(
                r'<text top="(\d+)" left="(\d+)" width="(\d+)" height="(\d+)" font="(\d+)">(.*?)</text>',
                body, re.S):
            t = html.unescape(re.sub('<[^>]+>', '', m.group(6))).strip()
            if not t:
                continue
            els.append(dict(top=int(m.group(1)), left=int(m.group(2)), w=int(m.group(3)),
                            h=int(m.group(4)), size=fsize.get(m.group(5), 0), txt=t))
        if els:
            els.sort(key=lambda e: (e['top'], e['left']))
            pages[no] = dict(page_w=page_w, page_h=page_h, els=els)
    if not pages:
        return None

    allh = sorted(e['h'] for p in pages.values() for e in p['els'])
    med_h = allh[len(allh) // 2]
    wsz = Counter()
    for p in pages.values():
        for e in p['els']:
            wsz[e['size']] += e['w']
    doc_size = wsz.most_common(1)[0][0] if wsz else 0

    # 본문 여백: 문서 전체에서 '본문 크기 글자로 된 조밀한 줄'의 최빈 left
    lefts = Counter()
    for p in pages.values():
        for L in group_lines(p['els'], med_h):
            if L['fill'] >= 0.90 and L['size'] >= doc_size * 0.95:
                lefts[L['left']] += 1
    body_left = lefts.most_common(1)[0][0] if lefts else 0
    return dict(pages=pages, med_h=med_h, doc_size=doc_size, body_left=body_left)


def group_lines(els, med_h):
    """텍스트 조각을 '줄'로 묶는다."""
    lines = []
    for e in els:
        for L in lines:
            if abs(L['top'] - e['top']) <= med_h * 0.6:
                L['top'] = min(L['top'], e['top'])
                L['bottom'] = max(L['bottom'], e['top'] + e['h'])
                L['left'] = min(L['left'], e['left'])
                L['right'] = max(L['right'], e['left'] + e['w'])
                L['parts'].append((e['left'], e['left'] + e['w'], e['txt']))
                L['sizes'].append((e['w'], e['size']))
                break
        else:
            lines.append(dict(top=e['top'], bottom=e['top'] + e['h'], left=e['left'],
                              right=e['left'] + e['w'],
                              parts=[(e['left'], e['left'] + e['w'], e['txt'])],
                              sizes=[(e['w'], e['size'])]))
    for L in lines:
        L['parts'].sort()
        L['txt'] = re.sub(r'\s+', ' ', ' '.join(p[2] for p in L['parts'])).strip()
        L['w'] = L['right'] - L['left']
        L['h'] = L['bottom'] - L['top']
        L['size'] = max(sz for _, sz in L['sizes'])
        # fill = 조각들이 실제로 줄을 채우는 비율. 본문은 ~1.0,
        # 가로로 흩어진 그림 라벨이 한 줄로 병합된 경우는 낮다.
        L['fill'] = sum(r - l for l, r, _ in L['parts']) / max(1, L['w'])
    lines.sort(key=lambda L: L['top'])
    return lines


def parse_page(doc, page):
    """한 페이지의 줄·캡션·경계 후보를 반환. 기준은 문서 수준 값을 쓴다."""
    P = doc['pages'].get(page)
    if not P:
        return None
    page_w, page_h = P['page_w'], P['page_h']
    med_h, doc_size, body_left = doc['med_h'], doc['doc_size'], doc['body_left']
    lines = group_lines(P['els'], med_h)
    if not lines:
        return None
    max_w = max(L['w'] for L in lines)

    def caption_starts(L):
        """줄 안에서 캡션이 시작하는 지점들을 (x좌표, kind, num) 으로 돌려준다.

        나란히 배치된 그림(subfigure)은 두 캡션이 한 줄로 묶이므로 여럿이 나올 수 있다.
        """
        text, offs = '', []          # offs: (문자시작offset, 문자끝offset, x좌표)
        for pl, pr, t in L['parts']:
            if text:
                text += ' '
            offs.append((len(text), len(text) + len(t), pl))
            text += t
        hits = []
        for m in CAP_ANY.finditer(text):
            x = next((o[2] for o in offs if o[0] <= m.start() < o[1]), L['left'])
            hits.append((x, 'Table' if m.group(1) == 'Table' else 'Figure', m.group(2)))
        return text, sorted(hits)

    # --- 캡션 블록 ----------------------------------------------------------
    caps, used = [], set()
    for i, L in enumerate(lines):
        if i in used:
            continue
        _, hits = caption_starts(L)
        if not hits or not CAP_RE.match(L['txt']):
            continue
        block, bottom = [i], L['bottom']
        for j in range(i + 1, len(lines)):
            N = lines[j]
            if CAP_RE.match(N['txt']):            # 다음 캡션 시작
                break
            if N['top'] - bottom > med_h * 0.9:   # 줄 간격이 벌어짐
                break
            if abs(N['left'] - body_left) > 4:    # 본문 여백에서 벗어남
                break
            block.append(j)
            bottom = N['bottom']
            if N['w'] < 0.6 * max_w:              # 짧은 줄 = 문단의 마지막 줄
                break
        used.update(block)

        # 한 줄에 캡션이 여럿이면 x 로 열을 나눈다. 각 열의 그림은 세로 범위를 공유한다.
        for n, (x, kind, num) in enumerate(hits):
            # 각 캡션은 자기 열의 왼쪽 끝에서 시작하므로, 다음 캡션의 x 가 열 경계가 된다
            xl = 0 if n == 0 else x
            xr = page_w if n == len(hits) - 1 else hits[n + 1][0]
            txt = ' '.join(
                ' '.join(t for pl, pr, t in lines[k]['parts'] if xl <= pl < xr)
                for k in block)
            caps.append(dict(kind=kind, num=num, top=L['top'], bottom=bottom,
                             xl=xl, xr=xr, text=re.sub(r'\s+', ' ', txt).strip()))

    # --- 경계 후보 ----------------------------------------------------------
    indent = max(24, int(page_w * 0.04))          # 문단 첫 줄 들여쓰기 허용폭
    body = []
    for k, L in enumerate(lines):
        if k in used:
            continue
        if not (body_left - 4 <= L['left'] <= body_left + indent):
            continue
        # 본문 여백에 딱 붙은 줄은 기준을 조금 낮춘다 — 수식이 섞인 본문 줄은
        # 조각 사이 간격 때문에 fill 이 떨어지기 때문이다.
        fill_min = 0.85 if L['left'] <= body_left + 4 else 0.90
        if not (L['fill'] >= fill_min and L['size'] >= doc_size * 0.95 and L['h'] <= med_h * 2.0):
            continue
        if L['left'] > body_left + 4:
            # 들여쓴 줄은 '문단 첫 줄'일 때만 본문으로 인정한다.
            # 진짜 문단이면 다음 줄이 본문 여백으로 돌아온다. 표의 행은 그렇지 않다.
            nxt = lines[k + 1] if k + 1 < len(lines) else None
            if (nxt is None or abs(nxt['left'] - body_left) > 4
                    or nxt['top'] - L['bottom'] > med_h * 0.9):   # 같은 문단으로 이어져야 한다
                continue
        body.append(L)
    folios = [L for L in lines                    # 하단 여백의 페이지 번호
              if L['top'] > page_h * 0.9
              and re.fullmatch(r'[ivxlcdm\d]+', L['txt'].strip().lower())]

    return dict(page_w=page_w, page_h=page_h, lines=lines, caps=caps, body=body,
                folios=folios, body_left=body_left, max_w=max_w,
                med_h=med_h, doc_size=doc_size)


def render(pdf, page, dpi, outdir):
    tag = os.path.join(outdir, f'.r{page}')
    subprocess.run(['pdftoppm', '-png', '-r', str(dpi), '-f', str(page), '-l', str(page), pdf, tag],
                   check=True)
    hit = [f for f in os.listdir(outdir) if f.startswith(f'.r{page}-')]
    return os.path.join(outdir, hit[0])


def row_ink(img, thresh=243):
    """행별 잉크 픽셀 수 (4에서 조기 종료)."""
    g = img.convert('L')
    W, H = g.size
    px = g.load()
    out = []
    for y in range(H):
        c = 0
        for x in range(W):
            if px[x, y] < thresh:
                c += 1
                if c > 3:
                    break
        out.append(c)
    return out


def extract(doc, pdf, page, dpi, outdir, prefix, pad=8):
    from PIL import Image
    os.makedirs(outdir, exist_ok=True)
    info = parse_page(doc, page)
    if not info or not info['caps']:
        return []

    rpath = render(pdf, page, dpi, outdir)
    img = Image.open(rpath)
    W, H = img.size
    sy = H / info['page_h']
    rows = row_ink(img)
    lh = max(3, int(info['med_h'] * sy))

    def edge_up(y):
        """y 근처에서 시작해, 걸쳐 있는 글줄을 위로 빠져나온 뒤 첫 잉크를 찾는다."""
        y = min(max(y, 0), H)
        limit = y - int(lh * 1.5)
        while y > 0 and y > limit and rows[y - 1] > 1:
            y -= 1
        while y > 0 and rows[y - 1] <= 1:
            y -= 1
        return y

    def edge_down(y):
        y = min(max(y, 0), H - 1)
        limit = y + int(lh * 1.5)
        while y < H - 1 and y < limit and rows[y] > 1:
            y += 1
        while y < H - 1 and rows[y] <= 1:
            y += 1
        return y

    # 공백 구간 (대체 경로용)
    runs, st = [], None
    for y, v in enumerate(rows):
        if v <= 1:
            if st is None:
                st = y
        elif st is not None:
            runs.append((st, y))
            st = None
    if st is not None:
        runs.append((st, len(rows)))
    sep = lh * 2.2          # 본문/그림을 가르는 공백 크기

    bounds = info['body'] + info['folios']
    results = []
    for c in info['caps']:
        cap_top, cap_bot = int(c['top'] * sy), int(c['bottom'] * sy)
        if c['kind'] == 'Figure':
            ups = [L['bottom'] for L in bounds if L['bottom'] <= c['top']]
            ups += [o['bottom'] for o in info['caps'] if o is not c and o['bottom'] <= c['top']]
            y1 = edge_up(cap_top)
            if ups:
                y0 = edge_down(int(max(ups) * sy))
            else:
                # 경계로 쓸 본문 줄이 없다 (예: 초록이 양쪽 들여쓰기된 표제지).
                # 큰 가로 공백으로 대신 찾는다.
                y0 = next((e for s, e in reversed(runs) if e < y1 - lh and (e - s) >= sep), 0)
        else:
            downs = [L['top'] for L in bounds if L['top'] >= c['bottom']]
            downs += [o['top'] for o in info['caps'] if o is not c and o['top'] >= c['bottom']]
            y0 = edge_down(cap_bot)
            if downs:
                y1 = edge_up(int(min(downs) * sy))
            else:
                y1 = next((s for s, e in runs if s > y0 + lh and (e - s) >= sep), H)

        # 대체 경로 — 표/그림의 내용이 본문 산문과 구조적으로 구분되지 않는 경우
        # (예: 시스템 프롬프트를 담은 박스). 큰 가로 공백으로 경계를 다시 잡는다.
        if y1 - y0 < dpi * 0.25:
            if c['kind'] == 'Table':
                y0 = edge_down(cap_bot)
                y1 = next((s for s, e in runs if s > y0 + lh and (e - s) >= sep), H)
            else:
                y1 = edge_up(cap_top)
                y0 = next((e for s, e in reversed(runs) if e < y1 - lh and (e - s) >= sep), 0)

        if y1 - y0 < dpi * 0.25:
            results.append(dict(caption=c, ok=False, reason=f'영역 높이 {y1 - y0}px'))
            continue

        # 나란히 놓인 그림이면 해당 열만 잘라낸다
        cx0 = int(c.get('xl', 0) * W / info['page_w'])
        cx1 = int(c.get('xr', info['page_w']) * W / info['page_w'])
        band = img.crop((cx0, y0, cx1, y1))
        bbox = band.convert('L').point(lambda v: 0 if v > 243 else 255).getbbox()
        if not bbox:
            results.append(dict(caption=c, ok=False, reason='잉크 없음'))
            continue
        x0, by0, x1, by1 = bbox
        pt = pad if c['kind'] == 'Figure' else 1   # 캡션이 붙은 쪽 여백은 최소화
        pb = 1 if c['kind'] == 'Figure' else pad
        box = (max(0, cx0 + x0 - pad), max(0, y0 + by0 - pt),
               min(W, cx0 + x1 + pad), min(H, y0 + by1 + pb))
        fig = img.crop(box)
        name = f"{prefix}_{'fig' if c['kind'] == 'Figure' else 'table'}{c['num']}.png"
        fig.save(os.path.join(outdir, name))
        results.append(dict(caption=c, ok=True, file=name, size=fig.size))

    os.remove(rpath)
    return results


def main():
    pdf, pages, dpi, outdir, prefix = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    pl = []
    for part in pages.split(','):
        if '-' in part:
            a, b = part.split('-')
            pl += list(range(int(a), int(b) + 1))
        else:
            pl.append(int(part))

    doc = parse_doc(pdf)
    if not doc:
        print("  텍스트를 읽을 수 없는 PDF 입니다."); return
    print(f"  문서 기준: 본문여백={doc['body_left']} 본문크기={doc['doc_size']} 줄높이={doc['med_h']}\n")
    out, ok, fail = [], 0, 0
    for p in pl:
        for r in extract(doc, pdf, p, dpi, outdir, prefix):
            c = r['caption']
            if r['ok']:
                ok += 1
                print(f"  p{p:<3} {c['kind']} {c['num']:<3} → {r['file']:<26} {r['size'][0]}x{r['size'][1]}")
                print(f"        {c['text'][:100]}")
            else:
                fail += 1
                print(f"  p{p:<3} {c['kind']} {c['num']:<3} ✗ {r['reason']}")
            out.append(dict(page=p, kind=c['kind'], num=c['num'], caption=c['text'],
                            file=r.get('file'), ok=r['ok']))
    print(f"\n  성공 {ok} / 실패 {fail}")
    json.dump(out, open(os.path.join(outdir, 'figures.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
