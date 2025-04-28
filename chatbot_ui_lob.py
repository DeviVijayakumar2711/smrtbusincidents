import os
import io
import re
import base64
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image


# Now pull in exactly the same settings, regardless:
AZURE_OPENAI_KEY        = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION= os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# The rest of your code below is **identical** to your local version…
# ── Instantiate the LLM client ────────────────────────────────────────────────
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType

try:
    import fitz  # PyMuPDF
    has_pymupdf = True
except ImportError:
    has_pymupdf = False
    print("Warning: PyMuPDF not installed. PDF processing will be disabled.")

llm = AzureChatOpenAI(
    deployment_name    = AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint     = AZURE_OPENAI_ENDPOINT,
    openai_api_key     = AZURE_OPENAI_KEY,
    openai_api_version = AZURE_OPENAI_API_VERSION,
    temperature        = 0.1,
)

# ── Monkey-patch pandas so to_markdown() doesn’t need tabulate ────────────────
pd.DataFrame.to_markdown = lambda self, **kwargs: self.to_string()

# ── (all your helpers, UI layout, and handlers exactly as in your local code) ──
# ── In-memory stores ────────────────────────────────────────────────────────────
csv_agents    = {}  # filename → CSV agent
uploaded_texts= {}  # filename → content for PDF/TXT

# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_text(path: str) -> str:
    ext = path.lower().split('.')[-1]
    if ext == "pdf" and has_pymupdf:
        try:
            doc = fitz.open(path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            return f"Error extracting text from PDF {path}: {e}"
    try:
        with open(path, "r", encoding="utf8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {e}"

# ── Gradio UI and Handlers ───────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("# Intelligent Document Assistant")
    gr.Markdown("Upload your CSV, PDF, or TXT files and ask questions!")
    gr.Markdown("## Start Guide:")
    gr.Markdown("- Select mode, upload file if needed, then ask your question.")
    gr.Markdown("- For CSV plots, check **This question requires a plot**.")
    gr.Markdown("---")

    mode_select     = gr.Radio(["Upload CSV", "Upload PDF/TXT", "General Question"],
                               label="Select Mode", value="General Question")
    file_upload     = gr.File(label="Upload File", file_count="multiple", visible=False)
    is_plot_question= gr.Checkbox(label="This question requires a plot", visible=False)
    query_input     = gr.Textbox(label="Your Question", lines=2)
    submit_btn      = gr.Button("Submit")
    clear_btn       = gr.Button("Clear")

    output_md       = gr.Markdown(label="Answer")
    plot_output     = gr.Image(label="Generated Plot", visible=False, type="pil")
    notes_bar       = gr.Textbox(label="Notes", interactive=False)
    download_output = gr.File(label="Download Output", visible=False)

    def handle_mode_change(mode):
        if mode == "Upload CSV":
            return gr.update(visible=True), gr.update(visible=True)
        elif mode == "Upload PDF/TXT":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    mode_select.change(handle_mode_change,
                       inputs=mode_select,
                       outputs=[file_upload, is_plot_question])

    def save_output(text_output, plot_filepath=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = "plot_output" if plot_filepath else "output"
        fn = f"{base}_{ts}.txt"
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text_output)
        return fn

    def clear_outputs():
        return "", None, None, ""

    clear_btn.click(clear_outputs,
                    outputs=[query_input, output_md, plot_output, notes_bar])

    def process_query(mode, files, is_plot, question):
        notes = ""
        if mode == "Upload CSV":
            if not files:
                return "", gr.update(visible=False), gr.update(visible=False), "Please upload a CSV file."
            for f in files:
                try:
                    if is_plot:
                        # Ask LLM for matplotlib code
                        instr = ("Generate Python code using matplotlib or pandas to answer this question: "
                                 f"{question}. Return code in a ```python block```.") + \
                                f"\n\nHere is the data (first 5 rows):\n{pd.read_csv(f.name).head().to_string()}"
                        response = llm.predict(instr)
                        match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
                        if match:
                            code = match.group(1).strip()
                            # Execute and capture plot
                            plt.figure()
                            exec(code, {"pd": pd, "plt": plt})
                            buf = io.BytesIO()
                            plt.savefig(buf, format="png")
                            buf.seek(0)
                            img = Image.open(buf)
                            plt.close()
                            return ("Plot generated.", 
                                    gr.update(value=img, visible=True),
                                    gr.update(visible=False),
                                    notes)
                        else:
                            return ("No Python plotting code found.",
                                    gr.update(visible=False),
                                    gr.update(visible=False),
                                    "No plotting code.")
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
                        return (ans,
                                gr.update(visible=False),
                                gr.update(value=path, visible=True),
                                notes)
                except Exception as e:
                    return (f"Error processing CSV: {e}",
                            gr.update(visible=False),
                            gr.update(visible=False),
                            f"{e}")
            return ("No CSV processed.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    notes)

        elif mode == "Upload PDF/TXT":
            if not files:
                return ("Please upload a PDF or TXT file.",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        "No file.")
            combined = ""
            for f in files:
                txt = extract_text(f.name)
                combined += f"\n\n--- {os.path.basename(f.name)} ---\n\n{txt}"
            prompt = f"Answer based on these documents:\n\n{combined}\n\nQuestion: {question}"
            try:
                resp = llm.predict(prompt).strip()
                path = save_output(resp)
                return (resp,
                        gr.update(visible=False),
                        gr.update(value=path, visible=True),
                        notes)
            except Exception as e:
                return (f"LLM error PDF/TXT: {e}",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        f"{e}")

        elif mode == "General Question":
            try:
                resp = llm.predict(question).strip()
                path = save_output(resp)
                return (resp,
                        gr.update(visible=False),
                        gr.update(value=path, visible=True),
                        notes)
            except Exception as e:
                return (f"LLM error general: {e}",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        f"{e}")

        return ("Please select a valid mode.",
                gr.update(visible=False),
                gr.update(visible=False),
                "Invalid mode.")

    submit_btn.click(process_query,
                     inputs=[mode_select, file_upload, is_plot_question, query_input],
                     outputs=[output_md, plot_output, download_output, notes_bar])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000, share=True)


