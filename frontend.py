#!/usr/bin/env python3
import asyncio, json, re, gradio as gr
from chatbot_dep import run_chat_turn
from stores_mem_and_cache.memory  import get_memory, _stm
from stores_mem_and_cache.cache   import _redis as redis_client, _local_cache   # reuse existing objects

# ────────── helpers ────────────────────────────────────────────────
def format_memory_md():
    mem = get_memory()
    stm = "\n".join(f"- **{t['role'].capitalize()}**: {t['content']}" for t in mem["stm"]) or "*empty*"
    ltm = "\n".join(f"- {m['content']}"               for m in mem["ltm"]) or "*empty*"
    return f"### Short‑Term\n\n{stm}\n\n---\n\n### Long‑Term\n\n{ltm}"

# REPLACE the current writer_sections() helper WITH:
def writer_sections(txt: str) -> dict[str, str]:
    """
    Robustly pull out THOUGHT / ACTION / EVIDENCE blocks.

    • Works whether or not <<END_EVIDENCE>> is present.
    • Strips a leading label inside the block (e.g. “EVIDENCE:”).
    """
    sections = {}
    spec = {"THOUGHT": "COT", "ACTION": "ACTION", "EVIDENCE": "EVIDENCE"}
    for lab, end in spec.items():
        m = re.search(rf"<<{lab}>>(.*?)(?:<<END_{end}>>|$)", txt, re.S)
        blk = m.group(1).strip() if m else ""
        blk = re.sub(rf"^{lab}\s*:\s*", "", blk, flags=re.I)  # dedup label
        sections[lab] = blk
    return sections

def _clear_memory():
    _stm.clear()

def _clear_cache():
    try:   redis_client.flushdb()
    except Exception: pass
    _local_cache.clear()


def compact_answer(txt:str):
    s=writer_sections(txt)
    return f"### THOUGHT\n{s['THOUGHT']}\n\n"\
           f"### ACTION\n{s['ACTION']}\n\n"\
           f"### EVIDENCE\n{s['EVIDENCE']}"

def detailed_logs(res: dict) -> str:
    """
    Show the *entire* raw JSON payload when we have a full pipeline run;
    otherwise show the fallback message from result["message"].
    """
    if res.get("type") == "pipeline":
        raw = json.dumps(res, indent=2, ensure_ascii=False)
        return f"```json\n{raw}\n```"
    else:
        return f"```\n{res.get('message','(no logs)')}\n```"
    
def full_cot_md(res:dict)->str:
    if res.get("type")!="pipeline":
        return res.get("message","")
    
    plan_cot = res["planner"]["cot_raw"]
    chunk_cots = "\n".join(f"- {c['cot_public']}" for c in res["chunks"])
    ver_cot  = res["verifier"]["reason"]

    # Writer CoT (robust)
    w_match = re.search(r'<<THOUGHT>>(.*?)<<END_COT>>', res["writer"], re.S)
    writer_cot = w_match.group(1).strip() if w_match else "(writer CoT not detected)"

    return (
        "### Planner CoT\n"
        f"{plan_cot}\n\n---\n\n"
        "### Per‑Chunk CoTs\n"
        f"{chunk_cots}\n\n---\n\n"
        "### Verifier CoT\n"
        f"{ver_cot}\n\n---\n\n"
        "### Writer CoT\n"
        f"{writer_cot}"
    )


# ────────── async wrapper ──────────────────────────────────────────
async def chat_backend(user_msg, chat_hist):
    logs_md = ""                                   # <‑‑ always defined
    cot_md  = ""

    loop=asyncio.get_event_loop()
    res = await loop.run_in_executor(None, run_chat_turn, user_msg)

     # ---- guarantee we have a dict ----
    # ── guarantee we work with a dict ─────────────────────────────
    # (When the fast‑path in run_chat_turn returns None.)
    if res is None:
        res = {"type": "error", "message": "No data returned from pipeline."}
    elif not isinstance(res, dict):
        res = {"type": "error", "message": f"Unexpected pipeline type: {type(res)}"}

    # decide visible message
    if res.get("type") == "pipeline":
        answer        = compact_answer(res["writer"])
        cot_md        = full_cot_md(res)
        logs_md       = detailed_logs(res)
    elif res.get("type") == "pipeline_cached":
        answer        = res["message"]                         # already compact
        cot_md        = "*(retrieved from cache – CoT not stored)*"
        logs_md       = "*(cached answer – raw logs unavailable)*"
    else:                                   # error / memory / off‑topic …
        answer        = res.get("message","")
        cot_md        = ""
        logs_md       = detailed_logs(res)

    chat_hist.append((user_msg, answer))
    mem_md  = format_memory_md()
    return chat_hist, mem_md, cot_md, logs_md, ""   # clear input box

# ────────── UI ─────────────────────────────────────────────────────
CSS = """
:root{
  --bg-user:   #0b57d0;
  --bg-bot:    #f1f3f4;
  --txt-user:  #fff;
  --txt-bot:   #202124;
  --mono: "SFMono-Regular",Consolas,Menlo,monospace;
}
.gr-chat-message.user   {background:var(--bg-user);color:var(--txt-user);}
.gr-chat-message.bot    {background:var(--bg-bot); color:var(--txt-bot);}
.gr-chat-message        {border-radius:8px;padding:8px 12px;margin:4px 0;}
.gr-prose pre, code     {font-family:var(--mono);}
#side-panels {max-height:calc(100dvh - 120px);overflow:auto;padding:0 8px;}
#side-panels           {display:flex;flex-direction:column;gap:6px}
.side-box              {max-height:260px;overflow:auto;}
.gr-accordion .label   {font-weight:600}
.gr-chatbot {border:1px solid #ddd}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("<h3 style='text-align:center'>🤖 MATLAB / Simulink Troubleshooter</h3>")

    with gr.Row():
        # chat & input
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600, label=None, value=[])  # full‑height GPT‑like
            with gr.Row():
                txt_in  = gr.Textbox(
                            placeholder="Ask a MATLAB / Simulink troubleshooting question…",
                            show_label=False, lines=1, autofocus=True, scale=4)
                send_btn   = gr.Button("Send",   variant="primary", scale=1)
                clear_btn  = gr.Button("⟲ Reset chat", variant="primary", scale=1)


        # side drawer (memory + logs) inside Tabs
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("🧠 Memory"):
                    mem_box = gr.Markdown(elem_classes="side-box")
                with gr.TabItem("🔍 Chain‑of‑Thought"):
                    cot_md  = gr.Markdown(elem_classes="side-box")
                with gr.TabItem("📜 Logs"):
                    log_md  = gr.Markdown(elem_classes="side-box")
                with gr.TabItem("🗑️ Controls"):
                    gr.Markdown("*Maintenance*")
                    clr_mem  = gr.Button("Clear Memory",   variant="destructive", size="sm")
                    clr_cache= gr.Button("Clear Cache",    variant="destructive", size="sm")

    # ---------- wiring ----------
    def _disable(): return gr.update(interactive=False)
    def _enable():  return gr.update(interactive=True)

    send_btn.click(_disable,  None, send_btn)
    send_btn.click(chat_backend,
                   [txt_in, chatbot],
                   [chatbot, mem_box, cot_md, log_md, txt_in]
                   ).then(_enable, None, send_btn)

    txt_in.submit(_disable, None, send_btn)\
          .then(chat_backend,
                [txt_in, chatbot],
                [chatbot, mem_box, cot_md, log_md, txt_in])\
          .then(_enable, None, send_btn)

    clear_btn.click(lambda: ([], "", "", "", ""), outputs=[chatbot, mem_box, cot_md, log_md, txt_in])

    def _do_clear_mem():
        _clear_memory()
        gr.Info("Memory cleared ✅")
        return format_memory_md()

    clr_mem.click(_do_clear_mem, None, mem_box)

    def _do_clear_cache():
        _clear_cache()
        return gr.Info("Cache cleared ✅")

    clr_cache.click(_do_clear_cache, None, None)

demo.launch(
    share=True
)