import os
import io
import re
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# Monkey-patch pandas so to_markdown() doesn’t need tabulate
pd.DataFrame.to_markdown = lambda self, **kwargs: self.to_string()

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType

try:
    import fitz  # PyMuPDF
    has_pymupdf = True
except ImportError:
    has_pymupdf = False
    print("Warning: PyMuPDF not installed; PDF processing disabled.")

# ── Load Azure settings ─────────────────────────────────────────────────────────
AZURE_OPENAI_KEY        = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION= os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

llm = AzureChatOpenAI(
    deployment_name    = AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint     = AZURE_OPENAI_ENDPOINT,
    openai_api_key     = AZURE_OPENAI_KEY,
    openai_api_version = AZURE_OPENAI_API_VERSION,
    temperature        = 0.1,
)

def extract_text(path: str) -> str:
    ext = path.lower().split('.')[-1]
    if ext == "pdf" and has_pymupdf:
        doc = fitz.open(path)
        txt = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return txt
    try:
        return open(path, "r", encoding="utf8", errors="ignore").read()
    except:
        return ""

def save_output(text: str) -> str:
    fn = f"output_{datetime.now():%Y%m%d_%H%M%S}.txt"
    with open(fn, "w", encoding="utf-8") as f:
        f.write(text)
    return fn

with gr.Blocks() as demo:
    gr.Markdown("## SMRT Knowledge Assistant")
    gr.Markdown("Upload CSV/PDF/TXT or ask general questions.")
    mode = gr.Radio(["Upload CSV","Upload PDF/TXT","General Question"],
                    label="Mode", value="General Question")
    upload = gr.File(label="Your File", file_count="multiple", visible=False)
    plot_cb = gr.Checkbox("This question requires a plot", visible=False)
    q_in = gr.Textbox(label="Your Question", lines=2)
    with gr.Row():
        submit = gr.Button("Submit")
        clear  = gr.Button("Clear")

    answer_md = gr.Markdown()
    plot_img  = gr.Image(type="pil", visible=False, label="Generated Plot")
    download  = gr.File(visible=False)
    notes     = gr.Textbox(interactive=False, label="Notes")

    gr.Markdown("---\n**Sample:**\n- CSV plot: “Histogram of age” (check plot box)\n"
                "- CSV query: “Top 5 causes?”\n- PDF/TXT: “Summarize this doc”\n"
                "- General: “Capital of Singapore?”")

    def on_mode(m):
        return (gr.update(visible=(m in ["Upload CSV","Upload PDF/TXT"])),
                gr.update(visible=(m=="Upload CSV")))
    mode.change(on_mode, mode, [upload, plot_cb])

    def clear_all():
        return "", gr.update(visible=False), "", "", ""
    clear.click(clear_all,
                outputs=[q_in, plot_img, answer_md, download, notes])

    def run(mode, files, want_plot, question):
        if mode=="Upload CSV":
            if not files:
                return "", gr.update(visible=False), "", "", "No file."
            df = pd.read_csv(files[0].name)

            # —— PLOT BRANCH —— 
            if want_plot:
                prompt = (
                    f"Generate Python code using matplotlib or pandas to answer: {question}\n\n"
                    f"Data preview:\n{df.head().to_string()}"
                )
                resp = llm.predict(prompt)
                m = re.search(r"```python\n(.*?)\n```", resp, re.DOTALL)
                if not m:
                    return resp, gr.update(visible=False), "", "", "No code."
                code = m.group(1)
                plt.figure()
                exec(code, {"pd":pd,"plt":plt,"df":df})
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                plt.close()
                img = Image.open(buf)
                return "Plot generated.", gr.update(value=img, visible=True), "", "", ""

            # —— AGENT BRANCH —— 
            agent = create_csv_agent(
                llm, files[0].name,
                verbose=False,
                allow_dangerous_code=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True
            )
            out = agent.run(question)
            path = save_output(out)
            return out, gr.update(visible=False), path, "", ""

        elif mode=="Upload PDF/TXT":
            if not files:
                return "", gr.update(visible=False), "", "", "No file."
            text = "\n\n".join(extract_text(f.name) for f in files)
            prompt = f"Answer based on:\n{text}\n\nQ: {question}"
            resp = llm.predict(prompt).strip()
            path = save_output(resp)
            return resp, gr.update(visible=False), path, "", ""

        else:  # General
            resp = llm.predict(question).strip()
            path = save_output(resp)
            return resp, gr.update(visible=False), path, "", ""

    submit.click(run,
                 inputs=[mode, upload, plot_cb, q_in],
                 outputs=[answer_md, plot_img, download, notes, notes])

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000, share=True)
