import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faqs import FAQS

_STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "not","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","this","that","these",
    "those","i","you","he","she","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","what","which","who","how","when",
    "where","why","with","from","by","as","if","so","up","out","about","into",
    "can","get","just","also","like","use","used","using",
}

def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return " ".join(tokens)

_questions_raw   = [f["q"] for f in FAQS]
_questions_clean = [_preprocess(q) for q in _questions_raw]

_vectorizer  = TfidfVectorizer()
_tfidf_matrix = _vectorizer.fit_transform(_questions_clean)

THRESHOLD = 0.12   

def get_answer(user_q: str) -> dict:
    clean = _preprocess(user_q)
    if not clean:
        return {"found": False, "answer": "Please type a question!", "confidence": 0, "matched": ""}
    vec  = _vectorizer.transform([clean])
    sims = cosine_similarity(vec, _tfidf_matrix).flatten()
    idx  = int(sims.argmax())
    score = float(sims[idx])
    if score < THRESHOLD:
        return {"found": False,
                "answer": "I'm sorry, I don't have an answer for that. Try rephrasing your question!",
                "confidence": round(score, 4), "matched": ""}
    return {"found": True,
            "answer": FAQS[idx]["a"],
            "confidence": round(score, 4),
            "matched": _questions_raw[idx]}

from flask import Flask, request, jsonify, Response
import threading, webbrowser, time

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FAQ Chatbot</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090f;--panel:#0d1018;--border:#1c2033;
  --accent:#00e5ff;--accent2:#7b61ff;
  --user:#7b61ff;--bot:#0d1018;
  --text:#dde3f0;--muted:#4a5270;
  --r:16px;
  --font-ui:'DM Mono',monospace;
  --font-head:'Syne',sans-serif;
}
body{font-family:var(--font-ui);background:var(--bg);color:var(--text);
  height:100dvh;display:flex;flex-direction:column;align-items:center;
  justify-content:center;overflow:hidden;}

/* noise grain overlay */
body::before{content:'';position:fixed;inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.04'/%3E%3C/svg%3E");
  pointer-events:none;z-index:0;opacity:.5;}

.shell{
  position:relative;z-index:1;
  width:min(720px,100vw);height:100dvh;
  display:flex;flex-direction:column;padding:14px;gap:10px;
}

/* header */
header{
  display:flex;align-items:center;gap:14px;
  background:var(--panel);border:1px solid var(--border);
  border-radius:var(--r);padding:14px 18px;
  animation:slideDown .4s ease both;
}
@keyframes slideDown{from{opacity:0;transform:translateY(-12px)}to{opacity:1;transform:translateY(0)}}

.logo{
  width:42px;height:42px;border-radius:50%;flex-shrink:0;
  background:linear-gradient(135deg,var(--accent2),var(--accent));
  display:grid;place-items:center;font-size:20px;
  box-shadow:0 0 18px color-mix(in srgb,var(--accent) 30%,transparent);
}
header h1{font-family:var(--font-head);font-size:1rem;letter-spacing:.02em;}
header p{font-size:.7rem;color:var(--muted);margin-top:2px;}
.dot{width:8px;height:8px;background:#22d17a;border-radius:50%;margin-left:auto;
  box-shadow:0 0 8px #22d17a;animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* chat area */
#chat{
  flex:1;overflow-y:auto;padding:14px;
  background:var(--panel);border:1px solid var(--border);
  border-radius:var(--r);display:flex;flex-direction:column;gap:14px;
  scroll-behavior:smooth;
}
#chat::-webkit-scrollbar{width:4px}
#chat::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}

.row{display:flex;gap:10px;animation:pop .22s ease both;}
@keyframes pop{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.row.user{flex-direction:row-reverse;}

.av{
  width:32px;height:32px;border-radius:50%;flex-shrink:0;
  display:grid;place-items:center;font-size:15px;align-self:flex-end;
}
.row.bot .av{background:var(--border);}
.row.user .av{background:linear-gradient(135deg,var(--accent2),var(--accent));}

.bubble{
  max-width:74%;padding:11px 15px;border-radius:var(--r);
  font-size:.84rem;line-height:1.65;white-space:pre-wrap;word-break:break-word;
}
.row.bot  .bubble{background:var(--bot);border:1px solid var(--border);border-bottom-left-radius:4px;}
.row.user .bubble{
  background:linear-gradient(135deg,var(--accent2),#5b41e0);
  color:#fff;border-bottom-right-radius:4px;
}
.meta{font-size:.65rem;color:var(--muted);margin-top:5px;line-height:1.4;}

/* suggestions */
#suggs{display:flex;gap:7px;overflow-x:auto;padding-bottom:2px;flex-shrink:0;}
#suggs::-webkit-scrollbar{display:none;}
.chip{
  flex-shrink:0;background:var(--panel);border:1px solid var(--border);
  color:var(--muted);padding:5px 13px;border-radius:99px;font-size:.73rem;
  cursor:pointer;font-family:var(--font-ui);
  transition:all .15s;white-space:nowrap;
}
.chip:hover{border-color:var(--accent);color:var(--accent);}

/* input bar */
.bar{
  display:flex;gap:10px;align-items:flex-end;
  background:var(--panel);border:1px solid var(--border);
  border-radius:var(--r);padding:10px 14px;
  transition:border-color .2s;
}
.bar:focus-within{border-color:var(--accent2);}
#inp{
  flex:1;background:transparent;border:none;outline:none;
  color:var(--text);font-size:.86rem;font-family:var(--font-ui);
  resize:none;max-height:110px;line-height:1.55;
}
#inp::placeholder{color:var(--muted);}
#btn{
  width:38px;height:38px;border-radius:50%;border:none;cursor:pointer;
  background:linear-gradient(135deg,var(--accent2),var(--accent));
  color:#fff;font-size:17px;display:grid;place-items:center;flex-shrink:0;
  transition:opacity .15s,transform .1s;
}
#btn:hover{opacity:.85;}
#btn:active{transform:scale(.91);}

/* typing dots */
.typing{display:flex;gap:5px;align-items:center;padding:6px 4px;}
.typing span{width:7px;height:7px;background:var(--muted);border-radius:50%;
  animation:bounce .8s infinite;}
.typing span:nth-child(2){animation-delay:.15s;}
.typing span:nth-child(3){animation-delay:.3s;}
@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-6px)}}
</style>
</head>
<body>
<div class="shell">

  <header>
    <div class="logo">🤖</div>
    <div>
      <h1>FAQ Chatbot</h1>
      <p>Python &amp; Tech — NLP powered · TF-IDF + Cosine Similarity</p>
    </div>
    <div class="dot"></div>
  </header>

  <div id="chat">
    <div class="row bot">
      <div class="av">🤖</div>
      <div>
        <div class="bubble">Hey! 👋 I'm your FAQ assistant. Ask me anything about Python, programming, or tech. You can also tap a suggestion below to get started!</div>
      </div>
    </div>
  </div>

  <div id="suggs"></div>

  <div class="bar">
    <textarea id="inp" rows="1" placeholder="Type your question…"></textarea>
    <button id="btn" title="Send">&#10148;</button>
  </div>

</div>
<script>
const chat=document.getElementById('chat'),
      inp =document.getElementById('inp'),
      btn =document.getElementById('btn'),
      suggs=document.getElementById('suggs');

const SAMPLE=[
  "What is Python?","What is pip?","What is Flask?",
  "Difference between list and tuple","How do I read a file?",
  "What is machine learning?","What is a virtual environment?",
  "How to handle exceptions?","What is a dictionary?","What is Git?"
];
SAMPLE.forEach(q=>{
  const c=document.createElement('button');
  c.className='chip';c.textContent=q;
  c.onclick=()=>send(q);
  suggs.appendChild(c);
});

function ts(){return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

function addMsg(text,role,meta=''){
  const row=document.createElement('div');
  row.className=`row ${role}`;
  row.innerHTML=`<div class="av">${role==='bot'?'🤖':'🧑'}</div>
    <div><div class="bubble">${esc(text)}</div>
    ${meta?`<div class="meta">${esc(meta)}</div>`:''}</div>`;
  chat.appendChild(row);
  chat.scrollTop=chat.scrollHeight;
}

function showTyping(){
  const r=document.createElement('div');
  r.className='row bot';r.id='typ';
  r.innerHTML=`<div class="av">🤖</div>
    <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chat.appendChild(r);chat.scrollTop=chat.scrollHeight;
}
function hideTyping(){document.getElementById('typ')?.remove();}

async function send(q){
  q=(q||inp.value).trim();
  if(!q)return;
  inp.value='';inp.style.height='auto';
  addMsg(q,'user',ts());
  showTyping();
  try{
    const r=await fetch('/ask',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q})});
    const d=await r.json();
    hideTyping();
    const meta=d.found
      ? `Matched: "${d.matched}" · Confidence: ${(d.confidence*100).toFixed(1)}% · ${ts()}`
      : ts();
    addMsg(d.answer,'bot',meta);
  }catch{
    hideTyping();
    addMsg('⚠️ Server error. Make sure chatbot.py is still running.','bot',ts());
  }
}

btn.onclick=()=>send();
inp.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}
});
inp.addEventListener('input',()=>{
  inp.style.height='auto';
  inp.style.height=inp.scrollHeight+'px';
});
</script>
</body>
</html>"""

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    q = data.get("question", "").strip()
    if not q:
        return jsonify({"error": "No question provided."}), 400
    return jsonify(get_answer(q))

if __name__ == "__main__":
    PORT = 5000
    url  = f"http://127.0.0.1:{PORT}"

    def _open_browser():
        time.sleep(1.2)          
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()

    print(f"\n{'='*52}")
    print(f"  🤖  FAQ Chatbot is starting …")
    print(f"  🌐  Opening browser at {url}")
    print(f"  ⏹   Press CTRL+C to stop")
    print(f"{'='*52}\n")

    app.run(port=PORT, debug=False)  