import os
import markdown
from xhtml2pdf import pisa

with open("RELATORIO.md", "r", encoding="utf-8") as f:
    texto_md = f.read()

html_corpo = markdown.markdown(texto_md, extensions=["tables", "fenced_code"])


def resolver_caminho(uri, rel):
    caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), uri)
    return caminho if os.path.exists(caminho) else uri

estilo = """
<style>
@page { size: A4; margin: 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; color: #1a1a1a; line-height: 1.4; }
h1 { font-size: 18pt; color: #0d3b66; border-bottom: 2px solid #0d3b66; padding-bottom: 4px; }
h2 { font-size: 13pt; color: #0d3b66; margin-top: 18px; border-bottom: 1px solid #cccccc; padding-bottom: 2px; }
h3 { font-size: 11pt; color: #333333; margin-top: 12px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9pt; }
th { background-color: #0d3b66; color: #ffffff; padding: 5px; text-align: left; }
td { border: 1px solid #cccccc; padding: 5px; }
tr:nth-child(even) { background-color: #f2f6fa; }
code { background-color: #eef1f4; padding: 1px 3px; font-family: Consolas, monospace; font-size: 9pt; }
hr { border: none; border-top: 1px solid #dddddd; margin: 14px 0; }
strong { color: #0d3b66; }
ul, ol { margin: 6px 0; }
img { margin: 8px 0; }
.legenda { font-size: 8.5pt; color: #666666; font-style: italic; text-align: center; }
</style>
"""

html_completo = f"<html><head><meta charset='utf-8'>{estilo}</head><body>{html_corpo}</body></html>"

saida = "RELATORIO_TECNICO_Seattle.pdf"
with open(saida, "w+b") as f:
    status = pisa.CreatePDF(html_completo, dest=f, encoding="utf-8", link_callback=resolver_caminho)

if status.err:
    print("Erro na geração do PDF.")
else:
    print(f"PDF gerado com sucesso: {saida}")
