"""Judge 1 of the blinded systematicity protocol: gemini-3.7-flash via OpenRouter.
Single big-context call (packet ~30k tok). Output: judge_gemini.md
"""
import json, os, pathlib, time
import httpx

D = pathlib.Path(__file__).parent
instruction = (D / "instruction.md").read_text()
packet = (D / "packet.json").read_text()
key = os.environ["OPENROUTER_API_KEY"]

body = {
    "model": "google/gemini-3.7-flash",
    "temperature": 0.3,
    "max_tokens": 8000,
    "messages": [
        {"role": "system", "content": "You are a careful, evidence-quoting analyst of model-generated text."},
        {"role": "user", "content": instruction + "\n\n## packet.json\n```json\n" + packet + "\n```"},
    ],
}
for attempt in range(5):
    r = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                   headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
    if r.status_code == 200:
        out = r.json()["choices"][0]["message"]["content"]
        (D / "judge_gemini.md").write_text(out)
        print(f"wrote judge_gemini.md ({len(out)} chars)")
        break
    print(f"attempt {attempt}: HTTP {r.status_code} {r.text[:300]}")
    time.sleep(2 ** attempt * 5)
else:
    raise SystemExit("gemini judge failed after 5 attempts")
