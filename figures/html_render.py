"""Shared HTML -> PNG renderer for the example figures (qualitative verbalizations,
AO readouts, NLA edit example). Renders a styled card with headless chromium
(playwright) at 2x device scale and screenshots the card element.

Needs playwright's chromium (`playwright install chromium`). Chromium needs the host
libs libnss3/libnspr4; on a host without them, extract the .debs user-space
(dpkg-deb -x) and point env PW_LIBS at the extracted lib directory, which is passed to
the browser process via LD_LIBRARY_PATH. Chromium runs with --disable-gpu (software raster).

Style follows the post conventions: generations in monospace boxes, provenance pills,
judge digests italic, highlight marks gold (true value) / fab (fabricated or edited) /
genre (framing without content).
"""
import html as _html
import os

PW_LIBS = os.environ.get("PW_LIBS", "")

CSS = """
* { box-sizing: border-box; }
body { margin: 0; background: #ffffff; }
#card { width: 880px; padding: 22px 26px 16px; background: #ffffff;
        font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1a1a1a; }
h2 { font-size: 16.5px; margin: 0 0 3px; font-weight: 700; }
.sub { color: #555; font-size: 13px; margin: 0 0 14px; }
.pill { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .4px;
        padding: 2px 9px; border-radius: 999px; margin: 0 6px 4px 0; color: #fff;
        vertical-align: middle; }
.pill i { font-family: Georgia, "Times New Roman", serif; font-style: italic;
          font-weight: 700; letter-spacing: 0; }
.pill.z { background: #b64a44; } .pill.h { background: #3d6fb0; }
.pill.step { background: #6b6b70; } .pill.out { background: #3c8a5c; }
.pill.method { font-size: 13px; padding: 4px 14px; letter-spacing: .6px;
               margin-bottom: 10px; text-transform: uppercase; display: table; }
.pill.nla { background: #7b4bb7; } .pill.ao { background: #0f766e; }
.pill.digest { background: #ece8f5; color: #5b4a8a; font-style: normal; font-size: 10px;
               letter-spacing: .5px; text-transform: uppercase; vertical-align: 1px; }
.gen { font-family: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
       font-size: 12.5px; line-height: 1.6; white-space: pre-wrap;
       background: #f7f7f8; border: 1px solid #e5e5ea; border-radius: 8px;
       padding: 9px 12px; margin: 4px 0 12px; }
mark { border-radius: 4px; padding: 0 3px; }
mark.ans { background: #8fd19c; } mark.gold { background: #ddf1e0; }
mark.fab { background: #f6c1c1; }
mark.genre { background: #c9d9f2; } mark.pred { background: #f8d9a3; }
table.tbl { width: 100%; border-collapse: collapse; margin: 4px 0 16px; }
.tbl th { text-align: left; font-size: 11px; color: #555; font-weight: 700;
          letter-spacing: .3px; text-transform: uppercase; padding: 4px 9px;
          border-bottom: 1.5px solid #d9d9de; }
.tbl th .thq { display: block; text-transform: none; letter-spacing: 0;
               font-weight: 400; font-family: ui-monospace, "SF Mono", Menlo,
               Consolas, "Liberation Mono", monospace; font-size: 10.5px;
               color: #777; margin-top: 2px; }
.tbl td { vertical-align: top; padding: 8px 9px; border-bottom: 1px solid #ececf0;
          font-size: 12.5px; line-height: 1.55; }
.tbl td.feed { color: #333; width: 25%; }
.tbl td.mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas,
               "Liberation Mono", monospace; font-size: 12px; }
.arrow { font-size: 15px; color: #444; margin: -4px 0 8px 4px; }
.outcome { font-size: 13.5px; line-height: 1.65; margin: 2px 0 0; }
.digest { font-style: italic; color: #333; }
.legend { font-size: 12px; color: #444; margin-top: 12px; border-top: 1px solid #e5e5ea;
          padding-top: 8px; }
.legend mark { margin-right: 14px; }
b.q { font-weight: 700; }
.mv { font-family: Georgia, "Times New Roman", serif; font-style: italic;
      font-weight: 700; }
sub, sup { font-size: 9px; }
"""


def esc(t):
    return _html.escape(t, quote=False)


def mark_words(text, cats, merge=True):
    """text (single paragraph) + per-word categories (None or css class) -> HTML with
    <mark> spans, adjacent same-category words merged into one span."""
    words = text.split()
    assert len(words) == len(cats), (len(words), len(cats))
    out, i = [], 0
    while i < len(words):
        if cats[i] is None:
            out.append(esc(words[i]))
            i += 1
            continue
        j = i
        while merge and j + 1 < len(words) and cats[j + 1] == cats[i]:
            j += 1
        out.append(f'<mark class="{cats[i]}">{esc(" ".join(words[i:j + 1]))}</mark>')
        i = j + 1
    return " ".join(out)


def _launch(p):
    env = dict(os.environ)
    if PW_LIBS:
        env["LD_LIBRARY_PATH"] = PW_LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    return p.chromium.launch(env=env, args=["--disable-gpu"])  # GPU process init hangs set_content on hosts with a broken driver


def screenshot(url, out, width=2400, height=2600, scale=2):
    """Viewport screenshot of a page (headless chrome --window-size=W,H --force-device-scale-factor=S)."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=scale)
        page.goto(url)
        page.screenshot(path=out)
        browser.close()


def render_card(body_html, out, width=880):
    from playwright.sync_api import sync_playwright
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<style>{CSS}</style></head><body>"
           f"<div id='card'>{body_html}</div></body></html>")
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": width, "height": 600},
                                device_scale_factor=2)
        page.set_content(doc)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        page.locator("#card").screenshot(path=out)
        browser.close()
    print(f"wrote {out}")
