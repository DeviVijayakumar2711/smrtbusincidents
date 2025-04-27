import os
import io
import pandas as pd
import gradio as gr
import re
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType

# Monkey-patch pandas so to_markdown() doesn’t require tabulate
pd.DataFrame.to_markdown = lambda self, **kwargs: self.to_string()

try:
    import fitz  # PyMuPDF
    has_pymupdf = True
except ImportError:
    has_pymupdf = False
    print("Warning: PyMuPDF not installed. PDF processing will be disabled.")

# ── Load & Promote env vars ─────────────────────────────────────────────────
AZURE_OPENAI_KEY        = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION= os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# instantiate LLM
llm = AzureChatOpenAI(
    deployment_name    = AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint     = AZURE_OPENAI_ENDPOINT,
    openai_api_key     = AZURE_OPENAI_KEY,
    openai_api_version = AZURE_OPENAI_API_VERSION,
    temperature        = 0.1,
)

# ── Helpers ─────────────────────────────────────────────────────────────────
def extract_text(path: str) -> str:
    ext = path.lower().split('.')[-1]
    if ext == "pdf" and has_pymupdf:
        doc = fitz.open(path)
        txt = "\n\n".join(p.get_text() for p in doc)
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

# ── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("## **SMRT Knowledge Assistant**\nUpload CSV/PDF/TXT or ask general questions.\n---")

    mode_select     = gr.Radio(
        ["Upload CSV","Upload PDF/TXT","General Question"],
        label="Select Mode", value="General Question"
    )
    file_upload     = gr.File(file_count="multiple", visible=False, label="Upload File")
    is_plot_question= gr.Checkbox(label="This question requires a plot", visible=False)
    query_input     = gr.Textbox(label="Your Question", lines=2)

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn  = gr.Button("Clear",  variant="secondary")

    output_md       = gr.Textbox(label="Answer", interactive=False, lines=8)
    plot_output     = gr.Image(label="Generated Plot", visible=False, type="pil")
    download_output = gr.File(label="Download Output", visible=False)
    notes_bar       = gr.Textbox(label="Notes", interactive=False, lines=2)

    gr.Markdown("---\n### Sample Questions:\n"
                "- What are the top 5 accident causes?\n"
                "- Show age distribution of injured passengers\n"
                "- Which interchanges have highest incidents?\n"
                "- How many cases were conveyed to hospital?\n"
                "- What's the gender split in personal injuries?")

    def toggle_upload(m):
        if m=="Upload CSV":
            return gr.update(visible=True), gr.update(visible=True)
        if m=="Upload PDF/TXT":
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=False)

    mode_select.change(toggle_upload,
                       inputs=mode_select,
                       outputs=[file_upload, is_plot_question])

    def clear_all():
        return "", False, "", None, "", ""

    clear_btn.click(clear_all,
                    outputs=[query_input, is_plot_question, output_md, plot_output, download_output, notes_bar])

    def process_query(mode, files, do_plot, question):
        notes = ""
        # --- CSV + plotting path ---
        if mode=="Upload CSV" and files:
            for f in files:
                df = pd.read_csv(f.name)
                if do_plot:
                    # ask for python code
                    prompt = (
                        f"Generate Python matplotlib code to answer: {question}\n"
                        f"Data preview:\n{df.head().to_string()}"
                    )
                    resp = llm.predict(prompt)
                    m = re.search(r"```python\n(.*?)\n```", resp, re.DOTALL)
                    if not m:
                        return resp, None, None, "", "No plotting code found."
                    code = m.group(1)
                    # exec & capture
                    plt.figure()
                    exec(code, {"pd":pd,"plt":plt, "df":df})
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png"); buf.seek(0); plt.close()
                    img = Image.open(buf)
                    return "Plot generated.", img, None, "", ""
                else:
                    agent = create_csv_agent(
                        llm, f.name,
                        verbose=False,
                        allow_dangerous_code=True,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        handle_parsing_errors=True
                    )
                    ans = agent.run(question)
                    path = save_output(ans)
                    return ans, None, path, "", ""

            return "No CSV processed.", None, None, "", notes

        # --- PDF / TXT path ---
        if mode=="Upload PDF/TXT" and files:
            text = "\n\n".join(extract_text(f.name) for f in files)
            prompt = f"Answer using these docs:\n\n{text}\n\nQuestion: {question}"
            ans = llm.predict(prompt)
            path = save_output(ans)
            return ans, None, path, "", ""

        # --- General LLM path ---
        ans = llm.predict(question)
        path = save_output(ans)
        return ans, None, path, "", ""

    submit_btn.click(
        process_query,
        inputs=[mode_select, file_upload, is_plot_question, query_input],
        outputs=[output_md, plot_output, download_output, notes_bar, notes_bar]
    )

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000, share=True)
