python3 -c '
import csv
from collections import Counter
rows = list(csv.DictReader(open("/mnt/c/research/results/phantomlint_passthrough.csv", encoding="utf-8-sig")))
print("Total rows:", len(rows))
print("Exit codes:", Counter(r["exit_code"] for r in rows))
print("Hidden flagged:", sum(r["pl_hidden_flagged"] == "True" for r in rows))
print("Suspicious flagged:", sum(r["pl_suspicious_flagged"] == "True" for r in rows))
print("Canary caught:", sum(r["pl_canary_caught"] == "True" for r in rows))
'





# NLP mode
python /mnt/c/research/run_phantomlint.py --manifest /mnt/c/research/pdfs/manifest.csv --out /mnt/c/research/results/phantomlint_nlp.csv --analyze nlp --resume

# LLM-Guard mode
python /mnt/c/research/run_phantomlint.py --manifest /mnt/c/research/pdfs/manifest.csv --out /mnt/c/research/results/phantomlint_llmguard.csv --analyze llm-guard --resume
