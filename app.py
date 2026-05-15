def make_report(summary, events, dwell_times, session_start):
    total = summary.get("total", 0)
    free  = summary.get("free", 0)
    occ   = summary.get("occ", 0)

    from reportlab.platypus import SimpleDocTemplate, Paragraph

    buf = io.BytesIO()

    doc = SimpleDocTemplate(buf)

    return buf.read(), "application/pdf", "parkvision_report.pdf"
