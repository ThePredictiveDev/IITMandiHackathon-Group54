#!/usr/bin/env python3
import asyncio, json, re, gradio as gr
from chatbot import run_chat_turn
from memory  import get_memory

# ────────── helpers ────────────────────────────────────────────────
def format_memory_md():
    mem = get_memory()
    stm = "\n".join(f"- **{t['role'].capitalize()}**: {t['content']}" for t in mem["stm"]) or "*empty*"
    ltm = "\n".join(f"- {m['content']}"               for m in mem["ltm"]) or "*empty*"
    return f"### Short‑Term\n\n{stm}\n\n---\n\n### Long‑Term\n\n{ltm}"

def writer_sections(txt:str):
    sec,pat={},{"THOUGHT":"COT","ACTION":"ACTION","EVIDENCE":"EVIDENCE"}
    for k,v in pat.items():
        m=re.search(rf"<<{k}>>(.*?)<<END_{v}>>",txt,re.S)
        sec[k]=m.group(1).strip() if m else ""
    return sec

def compact_answer(txt:str):
    s=writer_sections(txt)
    return f"#### THOUGHT\n{s['THOUGHT']}\n\n"\
           f"#### ACTION\n{s['ACTION']}\n\n"\
           f"#### EVIDENCE\n{s['EVIDENCE']}"

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
    loop=asyncio.get_event_loop()
    res = await loop.run_in_executor(None, run_chat_turn, user_msg)

    # decide visible message
    if res.get("type")=="pipeline":
        answer = compact_answer(res["writer"])
    else:
        answer = res["message"]

    chat_hist.append({"role": "user",      "content": user_msg})
    chat_hist.append({"role": "assistant", "content": answer})
    mem_md  = format_memory_md()
    cot_md  = full_cot_md(res)
    log_md  = detailed_logs(res)
    return chat_hist, mem_md, cot_md, log_md, ""   # clear input box

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
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("<h3 style='text-align:center;margin-bottom:4px'>🤖 MATLAB / Simulink Troubleshooter</h3>")

    with gr.Row():
        # -------- Left column: chat ----------
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                                        label="Conversation",
                                        height=460,
                                        avatar_images=(None, "🤖"),
                                        type="messages"          # suppress deprecation warning
                                    )
            txt_in  = gr.Textbox(
                placeholder="Ask a MATLAB / Simulink troubleshooting question…",
                show_label=False, lines=2, autofocus=True)
            send_btn = gr.Button("Send", variant="primary")
        # -------- Right column: insight panes -
        with gr.Column(scale=2, elem_id="side-panels"):
            with gr.Accordion("🧠 Memory", open=False):
                mem_box = gr.Markdown(elem_id="mem-md", show_label=False, elem_classes="side-box")
            with gr.Accordion("🔎 Full Chain‑of‑Thought", open=False):
                cot_md  = gr.Markdown(elem_id="cot-md",  show_label=False, elem_classes="side-box")
            with gr.Accordion("📜 Detailed Logs", open=False):
                log_md  = gr.Markdown(elem_id="log-md",  show_label=False, elem_classes="side-box")


    # wiring
    def _disable(): return gr.update(interactive=False)
    def _enable():  return gr.update(interactive=True)

    send_btn.click(_disable,  None, send_btn)
    send_btn.click(chat_backend,
                   [txt_in, chatbot],
                   [chatbot, mem_box, cot_md, log_md, txt_in]).then(
                   _enable, None, send_btn)

    txt_in.submit(_disable, None, send_btn)\
          .then(chat_backend,
                [txt_in, chatbot],
                [chatbot, mem_box, cot_md, log_md, txt_in])\
          .then(_enable, None, send_btn)

demo.launch(share=True)
