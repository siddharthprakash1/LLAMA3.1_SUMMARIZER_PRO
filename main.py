import pytube
import requests
import re
import gradio as gr
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
import tiktoken

# Function to retrieve the description of a YouTube video using regex to parse the HTML content
def get_youtube_description(url: str):
    full_html = requests.get(url).text
    y = re.search(r'shortDescription":"', full_html)
    desc = ""
    count = y.start() + 19  # adding the length of the 'shortDescription":"'
    while True:
        letter = full_html[count]
        if letter == "\"":
            if full_html[count - 1] == "\\":
                desc += letter
                count += 1
            else:
                break
        else:
            desc += letter
            count += 1
    return desc

# Function to get the YouTube video title and description using pytube and the custom get_youtube_description function
def get_youtube_info(url: str):
    yt = pytube.YouTube(url)
    title = yt.title if yt.title else "None"
    desc = get_youtube_description(url) if get_youtube_description(url) else "None"
    return title, desc

# Function to load the YouTube transcript using langchain's YoutubeLoader
def get_youtube_transcript_loader_langchain(url: str):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    return loader.load()

# Function to convert a list of documents into a single string
def wrap_docs_to_string(docs):
    return " ".join([doc.page_content for doc in docs]).strip()

# Function to create a text splitter for chunking the transcript into smaller pieces
def get_text_splitter(chunk_size: int, overlap_size: int):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=overlap_size)

# Function to get the YouTube transcript and count the number of tokens using tiktoken
def get_youtube_transcription(url: str):
    text = wrap_docs_to_string(get_youtube_transcript_loader_langchain(url))
    enc = tiktoken.encoding_for_model("gpt-4")
    count = len(enc.encode(text))
    return text, count

# Function to generate a summary of the YouTube transcription using langchain's summarize chain
def get_transcription_summary(url: str, temperature: float, chunk_size: int, overlap_size: int):
    docs = get_youtube_transcript_loader_langchain(url)
    text_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap_size)
    split_docs = text_splitter.split_documents(docs)
    llm = Ollama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=temperature,
    )
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    output = chain.run(split_docs)
    return output

# Create the Gradio interface with custom CSS for styling
with gr.Blocks(css=""" 
    #title, #desc, #trns_raw, #trns_sum, #tkncount, #temperature, #chunk, #overlap {
        transition: all 0.3s ease-in-out;
        background: radial-gradient(circle, rgba(0,36,72,1) 0%, rgba(3,169,244,1) 100%);
        color: white;
    }
    #title:focus, #desc:focus, #trns_raw:focus, #trns_sum:focus, #tkncount:focus, #temperature:focus, #chunk:focus, #overlap:focus {
        border-color: #1E88E5;
        box-shadow: 0 0 10px #1E88E5;
    }
    .gr-button {
        background-color: #43A047;
        color: white;
        border: none;
        transition: background-color 0.3s ease-in-out, transform 0.3s ease-in-out;
    }
    .gr-button:hover {
        background-color: #388E3C;
        transform: scale(1.05);
    }
    .gr-box {
        border-color: #1E88E5;
    }
    #clear-button {
        background-color: #E53935;
        color: white;
    }
    #clear-button:hover {
        background-color: #D32F2F;
    }
    .gr-textbox {
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        color: white;
    }
""") as demo:
    # Title of the application
    gr.Markdown("""
        <div style='text-align: center; font-size: 24px; font-weight: bold; color: #1E88E5;'>
            LlamaYT Insight Summarizer
        </div>
    """)
    # Section for input URL and buttons
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            url = gr.Textbox(label='YouTube URL', value="https://www.youtube.com/watch?v=6xY_LQzTyus")
        with gr.Column(scale=1):
            bttn_info_get = gr.Button('Retrieve Video Info', variant='primary')
            bttn_clear = gr.Button('Clear All Fields', variant='stop', elem_id="clear-button")

    # Section for displaying video title and description
    with gr.Row(variant='panel'):
        with gr.Column(scale=2):
            title = gr.Textbox(label='Video Title', lines=2, max_lines=10, show_copy_button=True, elem_id="title")
        with gr.Column(scale=3):
            desc = gr.Textbox(label='Video Description', max_lines=10, autoscroll=False, show_copy_button=True, elem_id="desc")

    # Connect the button to the function
    bttn_info_get.click(fn=get_youtube_info, inputs=url, outputs=[title, desc])

    # Section for generating transcription and summary
    with gr.Row(equal_height=True):
        with gr.Column():
            bttn_trns_get = gr.Button("Generate Transcription", variant='primary')
            tkncount = gr.Number(label='Estimated Token Count', interactive=False, elem_id="tkncount")
        with gr.Column(scale=3):
            bttn_summ_get = gr.Button("Generate Summary", variant='primary')
            with gr.Row():
                with gr.Column(scale=1):
                    temperature = gr.Number(label='Temperature', minimum=0.0, step=0.01, precision=2, elem_id="temperature")
                with gr.Column(scale=1):
                    chunk = gr.Number(label='Chunk Size', minimum=200, step=100, value=4000, elem_id="chunk")
                with gr.Column(scale=1):
                    overlap = gr.Number(label='Overlap Size', minimum=0, step=10, value=0, elem_id="overlap")

    # Section for displaying transcript and summary
    with gr.Row():
        with gr.Column():
            trns_raw = gr.Textbox(label='Transcript', show_copy_button=True, elem_id="trns_raw")
        with gr.Column():
            trns_sum = gr.Textbox(label="Summary", show_copy_button=True, elem_id="trns_sum")

    # Connect buttons to their respective functions
    bttn_trns_get.click(fn=get_youtube_transcription, inputs=url, outputs=[trns_raw, tkncount])
    bttn_summ_get.click(fn=get_transcription_summary, inputs=[url, temperature, chunk, overlap], outputs=trns_sum)
    bttn_clear.click(lambda: ('', '', '', '', 0), None, [url, title, desc, trns_raw, trns_sum, tkncount])

    # Add a section explaining the parameters
    gr.Markdown("""
        ## Parameter Explanations
        - **Token Count**: This is an estimate of the number of tokens (words or word pieces) in the transcription. It's useful for understanding the length and complexity of the content.
        - **Temperature**: Controls the randomness of the summary generation. Lower values (closer to 0) make the output more focused and deterministic, while higher values (closer to 1) make it more random and creative.
        - **Chunk Size**: The size of text chunks the video is split into for processing. Larger chunk sizes may capture more context but require more processing power.
        - **Overlap Size**: The number of tokens that overlap between consecutive chunks. A higher overlap can help maintain context between chunks, but it also increases processing time.
    """)

if __name__ == "__main__":
    demo.launch(share=True)
