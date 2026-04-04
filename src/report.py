from __future__ import annotations

import json
import html
from pathlib import Path
from typing import Any


def _escape(x: Any) -> str:
    if x is None:
        return ""
    return html.escape(str(x))


def _fmt_num(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _render_kv_table(d: dict[str, Any]) -> str:
    rows = []
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False, indent=2)
            v = f"<pre>{_escape(v)}</pre>"
        else:
            v = _escape(_fmt_num(v))
        rows.append(
            f"""
            <tr>
                <th>{_escape(k)}</th>
                <td>{v}</td>
            </tr>
            """
        )
    return f"""
    <table class="kv-table">
        {''.join(rows)}
    </table>
    """


def _render_list_of_dicts(items: list[dict[str, Any]], title: str) -> str:
    if not items:
        return f"<h4>{_escape(title)}</h4><p class='muted'>Нет данных</p>"

    all_keys = []
    seen = set()
    for item in items:
        for k in item.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    header = "".join(f"<th>{_escape(k)}</th>" for k in all_keys)
    body_rows = []

    for item in items:
        tds = []
        for k in all_keys:
            v = item.get(k, "")
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False, indent=2)
                v = f"<pre>{_escape(v)}</pre>"
            else:
                v = _escape(_fmt_num(v))
            tds.append(f"<td>{v}</td>")
        body_rows.append(f"<tr>{''.join(tds)}</tr>")

    return f"""
    <h4>{_escape(title)}</h4>
    <div class="table-wrap">
        <table>
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """


def _render_result_block(result: dict[str, Any]) -> str:
    parts = []

    for key, value in result.items():
        if isinstance(value, list):
            if value and all(isinstance(x, dict) for x in value):
                parts.append(_render_list_of_dicts(value, key))
            else:
                parts.append(
                    f"""
                    <h4>{_escape(key)}</h4>
                    <pre>{_escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>
                    """
                )
        elif isinstance(value, dict):
            # Если dict состоит из простых полей — показываем компактной таблицей
            simple = all(not isinstance(v, (dict, list)) for v in value.values())
            if simple:
                parts.append(f"<h4>{_escape(key)}</h4>{_render_kv_table(value)}")
            else:
                parts.append(
                    f"""
                    <h4>{_escape(key)}</h4>
                    <pre>{_escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>
                    """
                )
        else:
            parts.append(
                f"""
                <div class="inline-kv">
                    <span class="k">{_escape(key)}:</span>
                    <span class="v">{_escape(_fmt_num(value))}</span>
                </div>
                """
            )

    return "".join(parts)


def build_feature_cleaning_report_html(report: dict[str, Any]) -> str:
    target_column = report.get("target_column", "")
    initial_shape = report.get("initial_shape", [])
    final_shape = report.get("final_shape", [])
    steps = report.get("steps", [])

    step_cards = []
    for i, step in enumerate(steps, start=1):
        analyzer = step.get("analyzer", "")
        action = step.get("action", "")
        before_shape = step.get("before_shape", [])
        after_shape = step.get("after_shape", [])
        result = step.get("result", {})
        transform = step.get("transform", {})

        card = f"""
        <section class="card">
            <div class="card-header">
                <div>
                    <h2>Шаг {i}: {_escape(analyzer)}</h2>
                    <div class="sub">{_escape(action)}</div>
                </div>
                <div class="shape-box">
                    <div><strong>До:</strong> {_escape(before_shape)}</div>
                    <div><strong>После:</strong> {_escape(after_shape)}</div>
                </div>
            </div>

            <div class="two-col">
                <div>
                    <h3>Результат анализа</h3>
                    {_render_result_block(result)}
                </div>
                <div>
                    <h3>Преобразования</h3>
                    {_render_result_block(transform)}
                </div>
            </div>
        </section>
        """
        step_cards.append(card)

    html_doc = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8" />
        <title>Feature Cleaning Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 24px;
                color: #222;
                line-height: 1.45;
            }}
            h1, h2, h3, h4 {{
                margin-top: 0;
            }}
            .header {{
                margin-bottom: 24px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 10px;
                background: #f8f9fb;
            }}
            .summary {{
                display: flex;
                gap: 24px;
                flex-wrap: wrap;
                margin-top: 12px;
            }}
            .summary-box {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px 16px;
                background: white;
                min-width: 180px;
            }}
            .card {{
                margin-bottom: 24px;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 18px;
                page-break-inside: avoid;
            }}
            .card-header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 16px;
                margin-bottom: 16px;
                border-bottom: 1px solid #eee;
                padding-bottom: 12px;
            }}
            .sub {{
                color: #666;
                font-size: 14px;
            }}
            .shape-box {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px 12px;
                background: #fafafa;
                min-width: 180px;
            }}
            .two-col {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 6px 8px;
                vertical-align: top;
                text-align: left;
            }}
            th {{
                background: #f3f4f6;
            }}
            .kv-table th {{
                width: 220px;
            }}
            pre {{
                white-space: pre-wrap;
                word-break: break-word;
                margin: 0;
                font-size: 12px;
                background: #fafafa;
                border: 1px solid #eee;
                padding: 8px;
                border-radius: 6px;
            }}
            .muted {{
                color: #777;
            }}
            .inline-kv {{
                margin-bottom: 6px;
            }}
            .inline-kv .k {{
                font-weight: bold;
            }}
            .table-wrap {{
                overflow-x: auto;
            }}
            @media print {{
                body {{
                    margin: 10mm;
                }}
                .card {{
                    break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Feature Cleaning Report</h1>
            <div class="summary">
                <div class="summary-box"><strong>Target</strong><br>{_escape(target_column)}</div>
                <div class="summary-box"><strong>Initial shape</strong><br>{_escape(initial_shape)}</div>
                <div class="summary-box"><strong>Final shape</strong><br>{_escape(final_shape)}</div>
                <div class="summary-box"><strong>Steps</strong><br>{len(steps)}</div>
            </div>
        </div>

        {''.join(step_cards)}
    </body>
    </html>
    """
    return html_doc


def save_html_report(
    json_path: str | Path,
    html_path: str | Path = "feature_cleaning_report.html",
) -> Path:
    json_path = Path(json_path)
    html_path = Path(html_path)

    with json_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    html_text = build_feature_cleaning_report_html(report)

    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def save_pdf_report_from_html(
    html_path: str | Path,
    pdf_path: str | Path = "feature_cleaning_report.pdf",
) -> Path:
    html_path = Path(html_path)
    pdf_path = Path(pdf_path)

    try:
        from weasyprint import HTML
    except ImportError as e:
        raise ImportError(
            "Для генерации PDF установите weasyprint: pip install weasyprint"
        ) from e

    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    return pdf_path


if __name__ == "__main__":
    json_path = "feature_cleaning_report.json"

    html_path = save_html_report(
        json_path=json_path,
        html_path="feature_cleaning_report.html",
    )
    print(f"HTML report saved to: {html_path}")

    try:
        pdf_path = save_pdf_report_from_html(
            html_path=html_path,
            pdf_path="feature_cleaning_report.pdf",
        )
        print(f"PDF report saved to: {pdf_path}")
    except ImportError as e:
        print(e)
        print("PDF не создан. HTML уже сохранен.")