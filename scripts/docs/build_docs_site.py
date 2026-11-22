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
            heading_text = heading.group(2)
            if re.match(r"^Experiment\s+\d+\b", heading_text):
                output.append(
                    f'<h{level} class="experiment-title">{inline(heading_text)}</h{level}>'
                )
            else:
                output.append(f"<h{level}>{inline(heading_text)}</h{level}>")
            index += 1
            continue

        subtitle = re.match(r"^\*([^*]+)\*$", line.strip())
        if subtitle and subtitle.group(1).startswith("Experiment "):
            flush_paragraph()
            close_list()
            output.append(f'<p class="subtitle">{inline(subtitle.group(1))}</p>')
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
        elif line.strip() in ("---", "***", "___"):
            flush_paragraph()
            close_list()
            output.append("<hr>")
        elif image_match := re.match(r"^!\[([^]]*)\]\(([^)]+)\)$", line.strip()):
            flush_paragraph()
            close_list()
            alt, source = image_match.groups()
            if source.lower().endswith(".mp4"):
                output.append(
                    f'<p><video controls preload="metadata" style="max-width:100%; height:auto; border-radius:10px;">'
                    f'<source src="{html.escape(source, quote=True)}" type="video/mp4">'
                    f'Your browser does not support embedded video.</video></p>'
                )
            else:
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
        ("Experiments", [
            ("Overview", "experiments.html"),
            ("Experiment 1: Baseline", "experiment_1.html"),
            ("Experiment 2: Scaled Data", "experiment_2.html"),
            ("Experiment 3: Spatial Slots", "experiment_3.html"),
            ("Experiment 4: Safety Guards", "experiment_4.html"),
            ("Experiment 5: Intent Scenarios", "experiment_5.html"),
            ("Experiment 6: Selective Requests", "experiment_6.html"),
        ]),
        ("Data and Benchmarks", [
            ("Intent-aware collection", "intent_aware_data_collection.html"),
            ("Training runbook", "training.html"),
            ("Dataset dashboard", "dataset_dashboard.html"),
            ("Intent scenarios", "intent_scenarios.html"),
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
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
      onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
  <style>
    :root {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f6f7fb; }}
    * {{ box-sizing: border-box; }} body {{ margin: 0; }}
    .layout {{ display: flex; min-height: 100vh; }}
    aside {{ width: 250px; flex: 0 0 250px; background: #101828; color: white; padding: 28px 18px; position: sticky; top: 0; height: 100vh; overflow-y: auto; }}
    .brand {{ font-weight: 800; font-size: 22px; margin: 0 10px 4px; }}
    .tagline {{ color: #aab5ca; font-size: 13px; margin: 0 10px 26px; }}
    nav a {{ display: block; color: #c9d2e5; text-decoration: none; padding: 10px; border-radius: 8px; margin: 3px 0; }}
    nav a:hover, nav a.active {{ background: #263b78; color: white; }}
    .nav-group {{ margin: 0 0 18px; }}
    .nav-label {{ color: #7180a0; text-transform: uppercase; letter-spacing: .08em; font-size: 10px; font-weight: 700; padding: 0 10px 5px; }}
    main {{ width: min(980px, calc(100% - 48px)); margin: 0 auto; padding: 44px 24px 70px; }}
    article {{ background: white; border: 1px solid #d9deea; border-radius: 14px; padding: 30px 36px; box-shadow: 0 2px 8px #1018280b; }}
    h1 {{ color: #243b7a; font-size: 36px; line-height: 1.15; margin-top: 0; }} h2 {{ color: #29458f; margin-top: 36px; }} h3 {{ color: #344054; }}
    .experiment-title {{ display: inline-block; color: #604700; background: linear-gradient(transparent 12%, #ffe66d 12%, #ffe66d 90%, transparent 90%); padding: 3px 9px; border-radius: 5px; transform: rotate(-1deg); }}
    p {{ line-height: 1.65; }} .subtitle {{ display: inline-block; color: #604700; background: linear-gradient(transparent 12%, #ffe66d 12%, #ffe66d 90%, transparent 90%); font-size: 16px; font-weight: 800; letter-spacing: .02em; margin: -8px 0 24px; padding: 3px 9px; border-radius: 5px; transform: rotate(-1deg); }} ul, ol {{ margin: 12px 0 20px; padding-left: 28px; }} li {{ line-height: 1.45; margin: 0 0 6px; }} a {{ color: #3457d5; }} code {{ background: #f0f2f7; color: #24324a; padding: 2px 5px; border-radius: 5px; }}
    pre {{ background: #101828; color: #f8fafc; padding: 16px; border-radius: 9px; overflow-x: auto; }} pre code {{ background: transparent; color: #f8fafc; padding: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 18px 0; }} th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #d9deea; }} th {{ color: #667085; }}
    @media (max-width: 720px) {{ .layout {{ display: block; }} aside {{ width: 100%; height: auto; position: static; }} nav a {{ display: inline-block; }} main {{ width: 100%; padding: 18px 10px; }} article {{ padding: 22px 18px; }} }}
  </style>
</head>
<body><div class="layout"><aside><div class="brand">Echo Docs</div><p class="tagline">Cooperative driving and intent-aware world models</p><nav>{sidebar(current)}</nav></aside><main><article>{body}</article></main></div></body>
</html>"""


def main():
    markdown_pages = [
        DOCS / "experiments.md",
        DOCS / "experiment_1.md",
        DOCS / "experiment_2.md",
        DOCS / "experiment_3.md",
        DOCS / "experiment_4.md",
        DOCS / "experiment_5.md",
        DOCS / "experiment_6.md",
        DOCS / "merge_scenario.md",
        DOCS / "progress_log.md",
        DOCS / "mlpx.md",
        DOCS / "planner.md",
        DOCS / "policy.md",
        DOCS / "dqn_policy.md",
        DOCS / "rendering.md",
        DOCS / "training.md",
        DOCS / "intent_scenarios.md",
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
<p>Echo studies <strong>communication-efficient cooperative driving</strong>: an autonomous ego vehicle learns <strong>when and whom to ask</strong> for hidden driver intent, balancing communication costs against driving safety and efficiency.</p>

<hr>

<h2>How Echo Works</h2>
<ol>
<li><strong>Observe Surroundings:</strong> The ego vehicle tracks up to 8 spatial neighbor slots within a 60-meter radius on a 4-lane highway.</li>
<li><strong>Select Communication:</strong> The <strong>Request DQN</strong> decides whether to query a neighbor for their hidden intent (target lane, acceleration) or ask nobody to save cost.</li>
<li><strong>Simulate Futures:</strong> The <strong>MLPx World Model</strong> simulates 3-step action trajectories (5³ = 125 candidate paths) conditioned on received messages.</li>
<li><strong>Execute Safely:</strong> The <strong>Model-Predictive Planner</strong> selects the highest-reward action, filtered by an executive safety guard (speed cap and lane-clearance checks).</li>
</ol>

<hr>

<h2>Key Takeaways</h2>
<ul>
<li><strong>Highway Cruising Doesn't Require Messages:</strong> Standard open-highway traffic is predictable from position and speed alone. Asking nobody achieves <strong>0% crashes</strong> while avoiding unnecessary messaging costs.</li>
<li><strong>Intent Messages Prevent Ambiguous Collisions:</strong> In sudden cut-in scenarios where driver intentions are hidden until the last second, intent messages <strong>cut crashes in half</strong> (from 12 down to 6).</li>
<li><strong>Safety Guards Prevent Catastrophic Failure:</strong> Pairing the neural planner with a 30 m/s speed cap and destination-lane clearance checks dropped crashes from <strong>97% to 0%</strong>.</li>
<li><strong>Selective Training Improves the Request DQN:</strong> On identical fresh seeds, DQN6 roughly halved DQN5's request count and improved adjusted return in ordinary and intent traffic. Asking nobody still wins after message cost.</li>
<li><strong>Best Current Learned Configuration:</strong> DQN6 with the smooth controller produces more completed lane changes with zero ordinary-road crashes. It is the safety/behavior recommendation, while ask-nobody remains the adjusted-return baseline to beat.</li>
</ul>

<hr>

<h2>Explore the Documentation</h2>
<div class="cards">
<p><a href="experiments.html"><strong>Experiments 1–6 Breakdown</strong></a><br>Chronological milestones, hypotheses, benchmark tables, and findings across all 6 experiments.</p>
<p><a href="planner.html"><strong>Planner & Safety Guards</strong></a><br>How 3-step model-predictive control works and how safety guards eliminate crashes.</p>
<p><a href="mlpx.html"><strong>MLPx World Model</strong></a><br>Neural architecture, 118-value input vectors, and 3-step rollout accuracy.</p>
<p><a href="dqn_policy.html"><strong>Request DQN Policy</strong></a><br>How the agent learns which vehicle to ask, action masking, and reward trade-offs.</p>
<p><a href="intent_scenarios.html"><strong>Intent Scenarios</strong></a><br>Challenge benchmarks where vehicles hide their true destination lanes.</p>
<p><a href="dataset_dashboard.html"><strong>Dataset Dashboard</strong></a><br>Interactive charts and distributions across 145,000+ collected transitions.</p>
<p><a href="rendering.html"><strong>Visual Rendering</strong></a><br>How to record and view policies driving live with real-time intent overlays.</p>
<p><a href="training.html"><strong>Training Runbook</strong></a><br>Step-by-step commands for data collection, model training, and evaluation.</p>
<p><a href="progress_log.html"><strong>Progress Log</strong></a><br>Dated technical milestones and chronological experiment history.</p>
</div>"""
    index = DOCS / "index.html"
    index.write_text(page("Home", index_body, index))

    print("Built HTML documentation site in docs/")


if __name__ == "__main__":
    main()
