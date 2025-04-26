import os
import io
import pandas as pd
import gradio as gr
from dotenv import load_dotenv  # Removed - not needed for Azure
import base64
import re
import matplotlib.pyplot as plt
from datetime import datetime

# Monkey-patch pandas so to_markdown() doesn’t require tabulate
pd.DataFrame.to_markdown = lambda self, **kwargs: self.to_string()

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType

try:
    import fitz  # PyMuPDF
    has_pymupdf = True
except ImportError:
    has_pymupdf = False
    print("Warning: PyMuPDF not installed. PDF processing will be disabled.")

# ── Load & Promote env vars ────────────────────────────────────────────────────
# load_dotenv() # Removed - Not used in Azure

AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")  # Added default
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")  # Added default

# Check if the keys are set
if not AZURE_OPENAI_KEY:
    raise EnvironmentError(
        "AZURE_OPENAI_KEY is not set.  Please configure it in Azure Application Settings.")
if not AZURE_OPENAI_ENDPOINT:
    raise EnvironmentError(
        "AZURE_OPENAI_ENDPOINT is not set. Please configure it in Azure Application Settings.")

# ── Instantiate clients ────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.1,
)

# ── In-memory stores ────────────────────────────────────────────────────────────
csv_agents = {}  # filename → CSV agent
uploaded_texts = {}  # filename → content for PDF/TXT


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
    # TXT or others: read raw bytes, decode safe
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
    gr.Markdown("- Select the appropriate mode for your file type or general questions.")
    gr.Markdown("- Upload your file if you selected 'Upload CSV' or 'Upload PDF/TXT'.")
    gr.Markdown(
        "- For CSV files, check 'This question requires a plot' if you want a visualization.")
    gr.Markdown("- Enter your question and click 'Submit'.")
    gr.Markdown("---")

    with gr.Row():
        mode_select = gr.Radio(
            ["Upload CSV", "Upload PDF/TXT", "General Question"], label="Select Mode",
            value="General Question")

    with gr.Row():
        file_upload = gr.File(label="Upload File", file_count="multiple", visible=False)
        is_plot_question = gr.Checkbox(label="This question requires a plot",
                                        visible=False)

    query_input = gr.Textbox(label="Your Question", lines=2)

    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")

    output_md = gr.Markdown(label="Answer")
    plot_output = gr.Image(label="Generated Plot", visible=False)
    notes_bar = gr.Textbox(label="Notes",
                            placeholder="Any important notes or observations will appear here.",
                            interactive=False)
    download_output = gr.File(label="Download Output", visible=False,
                                interactive=False)

    gr.Markdown("---")
    gr.Markdown("## Sample Questions:")
    gr.Markdown(
        "- **CSV:** 'Show me a histogram of the age column.' (Remember to check the plot box)")
    gr.Markdown("- **CSV:** 'What are the top 5 most frequent accident causes?'")
    gr.Markdown("- **PDF/TXT:** 'Summarize the key findings of this document.'")
    gr.Markdown("- **General:** 'What is the capital of Singapore?'")


    def handle_mode_change(mode):
        if mode == "Upload CSV":
            return gr.update(visible=True), gr.update(visible=True)
        elif mode == "Upload PDF/TXT":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=False)


    mode_select.change(handle_mode_change, inputs=mode_select,
                        outputs=[file_upload, is_plot_question])


    def save_output(text_output, plot_filepath=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = "output"
        if plot_filepath:
            filename_base = "plot_output"
        filepath = f"{filename_base}_{timestamp}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_output)
        return filepath


    def clear_outputs():
        return "", None, None, ""


    clear_btn.click(clear_outputs, outputs=[query_input, output_md, plot_output, notes_bar])


    def process_query(mode, files, is_plot, question):
        notes = ""
        plot_file_path = None
        if mode == "Upload CSV":
            if not files:
                return "", gr.update(visible=False), gr.update(
                    visible=False), "Please upload a CSV file."
            csv_agents.clear()  # Clear previous agents
            for f in files:
                try:
                    if is_plot:
                        plot_instruction = (
                            f"Generate Python code using matplotlib or pandas to answer this question: {question}. Ensure the code is within a ```python block.")
                        agent = AzureChatOpenAI(
                            temperature=0.1,
                            azure_endpoint=AZURE_OPENAI_ENDPOINT,
                            openai_api_key=AZURE_OPENAI_KEY,
                            openai_api_version=AZURE_OPENAI_API_VERSION,
                        )  # Explicitly pass credentials
                        response = agent.predict(
                            plot_instruction + f"\n\nHere is the data (first 5 rows):\n{pd.read_csv(f.name).head().to_string()}")
                        code_match = re.search(r"```python\n(.*?)\n```", response,
                                                re.DOTALL)
                        if code_match:
                            plotting_code = code_match.group(1).strip()
                            import matplotlib.pyplot as plt
                            import io
                            import base64
                            plt.figure()
                            try:
                                exec(plotting_code)
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                buf.seek(0)
                                img_base64 = base64.b64encode(buf.read()).decode(
                                    'utf-8')
                                plt.close()
                                return "Plot generated.", gr.update(
                                    value=f"data:image/png;base64,{img_base64}",
                                    visible=True), gr.update(visible=False), notes
                            except Exception as e:
                                plt.close()
                                return f"Error during plot execution: {e}", gr.update(
                                    visible=False), gr.update(
                                    visible=False), f"Error during plot execution: {e}"
                        else:
                            return "No Python plotting code found in the response.", gr.update(
                                visible=False), gr.update(visible=False), "No plotting code found."
                    else:
                        agent = create_csv_agent(
                            llm,
                            f.name,
                            verbose=False,
                            allow_dangerous_code=True,
                            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            handle_parsing_errors=True
                        )
                        output = agent.run(question)
                        filepath = save_output(output)
                        return output, gr.update(visible=False), gr.update(
                            value=filepath, visible=True), notes

                except Exception as e:
                    return f"Error processing CSV: {e}", gr.update(visible=False), gr.update(
                        visible=False), f"Error processing CSV: {e}"
            return "No CSV file processed.", gr.update(visible=False), gr.update(
                visible=False), notes

        elif mode == "Upload PDF/TXT":
            if not files:
                return "Please upload a PDF or TXT file.", gr.update(
                    visible=False), gr.update(visible=False), "Please upload a file."
            uploaded_texts.clear()  # Clear previous texts
            all_text = ""
            for f in files:
                content = extract_text(f.name)
                uploaded_texts[f.name] = content
                all_text += f"\n\n--- {f.name} ---\n\n{content}"
            prompt = (
                f"You are a helpful assistant. Answer the question based on the following documents:\n\n{all_text}\n\nQuestion: {question}")
            try:
                response = llm.predict(prompt)
                filepath = save_output(response.strip())
                return response.strip(), gr.update(visible=False), gr.update(
                    value=filepath, visible=True), notes
            except Exception as e:
                return f"LLM error for PDF/TXT: {e}", gr.update(visible=False), gr.update(
                    visible=False), f"LLM error for PDF/TXT: {e}"
            return "No PDF/TXT file processed.", gr.update(visible=False), gr.update(
                visible=False), notes

        elif mode == "General Question":
            try:
                response = llm.predict(question)
                filepath = save_output(response.strip())
                return response.strip(), gr.update(visible=False), gr.update(
                    value=filepath, visible=True), notes
            except Exception as e:
                return f"LLM error for general question: {e}", gr.update(
                    visible=False), gr.update(visible=False), f"LLM error for general question: {e}"
            return "", gr.update(visible=False), gr.update(visible=False), notes
        else:
            return "Please select a mode.", gr.update(visible=False), gr.update(
                visible=False), "Please select a mode."


    submit_btn.click(process_query,
                    inputs=[mode_select, file_upload, is_plot_question, query_input],
                    outputs=[output_md, plot_output, download_output, notes_bar])

if __name__ == "__main__":
    demo.launch()
