"""Build the project's static HTML documentation site.

Run from the project root with:

    .venv/bin/python -m scripts.docs.build_docs_site
"""

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text


def markdown_to_html(markdown):
    lines = markdown.splitlines()
    output = []
    paragraph = []
    list_type = None
    in_code = False
    code_lines = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            output.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph = []

    def close_list():
        nonlocal list_type
        if list_type:
            output.append(f"</{list_type}>")
            list_type = None

    index = 0
    while index < len(lines):
        line = lines[index]

        if line.startswith("```"):
            flush_paragraph()
            close_list()
            if in_code:
                output.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue

        if in_code:
            code_lines.append(line)
            index += 1
            continue

        table_match = line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|")
        if table_match:
            flush_paragraph()
            close_list()
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            if len(table_lines) >= 2:
                header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
                rows = []
                for row in table_lines[2:]:
                    rows.append([cell.strip() for cell in row.strip("|").split("|")])
                table = ["<table><thead><tr>"]
                table.extend(f"<th>{inline(cell)}</th>" for cell in header)
                table.append("</tr></thead><tbody>")
                for row in rows:
                    table.append("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>")
                table.append("</tbody></table>")
                output.append("".join(table))
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            output.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            index += 1
            continue

        unordered = re.match(r"^\s*[-*]\s+(.+)$", line)
        ordered = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if unordered or ordered:
            flush_paragraph()
            desired = "ul" if unordered else "ol"
            if list_type != desired:
                close_list()
                output.append(f"<{desired}>")
                list_type = desired
            item = unordered or ordered
            output.append(f"<li>{inline(item.group(1))}</li>")
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
            close_list()
        elif image_match := re.match(r"^!\[([^]]*)\]\(([^)]+)\)$", line.strip()):
            flush_paragraph()
            close_list()
            alt, source = image_match.groups()
            output.append(f'<p><img src="{html.escape(source, quote=True)}" alt="{html.escape(alt, quote=True)}" style="max-width:100%; height:auto;"></p>')
        elif list_type and output and output[-1].endswith("</li>"):
            # Markdown bullets are often wrapped across several source lines.
            # Keep continuation lines inside the preceding list item instead
            # of turning them into a visually detached paragraph.
            output[-1] = output[-1][:-5] + " " + inline(line.strip()) + "</li>"
        else:
            close_list()
            paragraph.append(line.strip())
        index += 1

    if in_code:
        output.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
    flush_paragraph()
    close_list()
    return "\n".join(output)


def sidebar(current):
    groups = [
        ("Overview", [
            ("Home", "index.html"),
            ("Timeline", "timeline.html"),
            ("Progress log", "progress_log.html"),
        ]),
        ("System", [
            ("MLPx world model", "mlpx.html"),
            ("Planner", "planner.html"),
            ("Policy", "policy.html"),
            ("Request DQN", "dqn_policy.html"),
            ("Rendering", "rendering.html"),
        ]),
        ("Data and experiments", [
            ("Intent-aware collection", "intent_aware_data_collection.html"),
            ("Training", "training.html"),
            ("Dataset dashboard", "dataset_dashboard.html"),
            ("Merge scenario", "merge_scenario.html"),
        ]),
    ]
    prefix = "../" if current.parent.name == "archive" else ""
    items = []
    for group, links in groups:
        items.append(f'<div class="nav-group"><div class="nav-label">{group}</div>')
        for label, target in links:
            target = prefix + target if not target.startswith("../") else target
            active = " class=\"active\"" if target == current.name else ""
            items.append(f'<a{active} href="{target}">{label}</a>')
        items.append("</div>")
    return "".join(items)


def page(title, body, current):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} · Echo Docs</title>
  <style>
    :root {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f6f7fb; }}
    * {{ box-sizing: border-box; }} body {{ margin: 0; }}
    .layout {{ display: flex; min-height: 100vh; }}
    aside {{ width: 250px; flex: 0 0 250px; background: #101828; color: white; padding: 28px 18px; position: sticky; top: 0; height: 100vh; }}
    .brand {{ font-weight: 800; font-size: 22px; margin: 0 10px 4px; }}
    .tagline {{ color: #aab5ca; font-size: 13px; margin: 0 10px 26px; }}
    nav a {{ display: block; color: #c9d2e5; text-decoration: none; padding: 10px; border-radius: 8px; margin: 3px 0; }}
    nav a:hover, nav a.active {{ background: #263b78; color: white; }}
    .nav-group {{ margin: 0 0 18px; }}
    .nav-label {{ color: #7180a0; text-transform: uppercase; letter-spacing: .08em; font-size: 10px; font-weight: 700; padding: 0 10px 5px; }}
    main {{ width: min(980px, calc(100% - 48px)); margin: 0 auto; padding: 44px 24px 70px; }}
    article {{ background: white; border: 1px solid #d9deea; border-radius: 14px; padding: 30px 36px; box-shadow: 0 2px 8px #1018280b; }}
    h1 {{ color: #243b7a; font-size: 36px; line-height: 1.15; margin-top: 0; }} h2 {{ color: #29458f; margin-top: 36px; }} h3 {{ color: #344054; }}
    p {{ line-height: 1.65; }} ul, ol {{ margin: 12px 0 20px; padding-left: 28px; }} li {{ line-height: 1.45; margin: 0 0 6px; }} a {{ color: #3457d5; }} code {{ background: #f0f2f7; color: #24324a; padding: 2px 5px; border-radius: 5px; }}
    pre {{ background: #101828; color: #f8fafc; padding: 16px; border-radius: 9px; overflow-x: auto; }} pre code {{ background: transparent; color: #f8fafc; padding: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 18px 0; }} th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #d9deea; }} th {{ color: #667085; }}
    @media (max-width: 720px) {{ .layout {{ display: block; }} aside {{ width: 100%; height: auto; position: static; }} nav a {{ display: inline-block; }} main {{ width: 100%; padding: 18px 10px; }} article {{ padding: 22px 18px; }} }}
  </style>
</head>
<body><div class="layout"><aside><div class="brand">Echo Docs</div><p class="tagline">Cooperative driving and intent-aware world models</p><nav>{sidebar(current)}</nav></aside><main><article>{body}</article></main></div></body>
</html>"""


def main():
    markdown_pages = [
        DOCS / "merge_scenario.md",
        DOCS / "progress_log.md",
        DOCS / "mlpx.md",
        DOCS / "planner.md",
        DOCS / "policy.md",
        DOCS / "dqn_policy.md",
        DOCS / "rendering.md",
        DOCS / "training.md",
    ]
    for source in markdown_pages:
        target = source.with_suffix(".html")
        body = markdown_to_html(source.read_text())
        target.write_text(page(source.stem.replace("_", " ").title(), body, target))

    archive_dir = DOCS / "archive"
    for source in sorted(archive_dir.glob("*.md")):
        target = source.with_suffix(".html")
        body = markdown_to_html(source.read_text())
        target.write_text(page(source.stem.replace("_", " ").title(), body, target))

    index_body = """<h1>Echo Documentation</h1>
<p>Echo studies when an autonomous vehicle should ask nearby vehicles for hidden intent information, and how that information changes safe driving actions.</p>
<h2>Start here</h2>
<div class="cards"><p><a href="mlpx.html"><strong>MLPx world model</strong></a><br>State, action, message inputs and next-state/reward predictions.</p><p><a href="planner.html"><strong>Planner</strong></a><br>How imagined transitions are used to compare ego actions.</p><p><a href="policy.html"><strong>Data-collection policy</strong></a><br>How the rule-based policy selects the ego driving action.</p><p><a href="dqn_policy.html"><strong>Request DQN</strong></a><br>How the learned policy decides which vehicle to ask.</p><p><a href="rendering.html"><strong>Rendering</strong></a><br>How to inspect the learned policy visually.</p><p><a href="training.html"><strong>Training</strong></a><br>Dataset preparation, training commands, metrics, and evaluation.</p><p><a href="intent_aware_data_collection.html"><strong>Intent-aware collection</strong></a><br>Current four-lane environment, visibility rules, intent previews, and dataset generation.</p><p><a href="dataset_dashboard.html"><strong>Dataset dashboard</strong></a><br>Charts and statistics for the generated intent-aware dataset.</p></div>
<h2>Project workflow</h2>
<ol><li>Create the environment with a controlled ego and aggressive/defensive background vehicles.</li><li>Filter visible neighbors and optionally request their next-action intent.</li><li>Use the rule or learned planner to choose an ego action.</li><li>Evaluate transitions and store state, messages, action, next state, reward, and episode flags.</li></ol>
<h2>Reference</h2><p><a href="progress_log.html">Progress log</a> contains the dated history. The <a href="timeline.html">timeline</a> provides a visual project history. Archived experiment notes are available under <a href="archive/earlier_experiments.html">archive</a>.</p>"""
    index = DOCS / "index.html"
    index.write_text(page("Home", index_body, index))

    print("Built HTML documentation site in docs/")


if __name__ == "__main__":
    main()
