import os

import base64
import gc
import random
import tempfile
import time
import uuid
from IPython.display import Markdown, display
import streamlit as st
from PIL import Image
import torch
import time
import numpy as np
from tqdm import tqdm
from pdf2image import convert_from_path
from rag_code import EmbedData, QdrantVDB_QB, Retriever, RAG
from pathlib import Path

collection_name = "Hot-Doc-AI"
torch.classes.__path__ = []


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    gc.collect()

session_id = st.session_state.id

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

if "symbols_list" not in st.session_state:
    st.session_state.symbols_list = None

st.set_page_config(
    layout = 'wide',
    page_title = 'Hot-Doc AI by AwanCerah'
)

st.markdown(
    """
    <style>
        [data-testid="stMainBlockContainer"] {
            margin-top: -5rem;
        }
        [data-testid="stAppViewContainer"] {
            background: rgb(226,229,230);
            background: linear-gradient(133deg, rgba(226,229,230,1) 20%, rgba(254,214,101,1) 100%);
        }
        footer {display: none}
        [data-testid="stHeader"] {display: none}
    </style>
    """, unsafe_allow_html = True
)

# Css File include
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

title_col, emp_col, banner_col = st.columns([1,0.1,5])

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='width: 100%; padding-top: 45px'>".format(
      img_to_bytes(img_path)
    )
    return img_html

with title_col:
    # st.markdown('<p class="dashboard_title">Hot-Doc</p>', unsafe_allow_html = True)
    # st.image("assets/logo_cimb.png", width=300)
    st.markdown(img_to_html('assets/logo_cimb.png'), unsafe_allow_html=True)


with banner_col:
    with st.container():
        st.html("""
                <div class="banner">
                <p class="xrp_text" style="opacity: 1">AwanCerah<br></p>
                <p class="price_details" style="opacity: 1">Hot-Doc AI</p></div>""".format(img_to_html('assets/logo_cimb.png')))

st.divider()
params_col, chart_col = st.columns([0.5,2])
with params_col:
    with st.columns(1)[0]:
        uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
        if uploaded_file:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    file_key = f"{session_id}-{uploaded_file.name}"
                    st.write("Indexing your document...")

                    if file_key not in st.session_state.get('file_cache', {}):

                        # Store Pdf with convert_from_path function
                        images = convert_from_path(file_path)

                        for i in range(len(images)):
                        
                            # Save pages as images in the pdf
                            images[i].save('./images/page'+ str(i) +'.jpg', 'JPEG')

                        # embed data
                        print("Start Emebedding data")    
                        embeddata = EmbedData()
                        print("Start Emebedding Images")
                        embeddata.embed(images)

                        # set up vector database
                        print("==== Start Vector DB ====")
                        qdrant_vdb = QdrantVDB_QB(collection_name=collection_name,
                                                vector_dim=1031)
                        print("==== Start Vector DB Define Client ====")                                              
                        qdrant_vdb.define_client()
                        print("==== Start Vector DB Create Collection ====")
                        qdrant_vdb.create_collection()
                        print("==== Start Vector DB Embedd Data ====")
                        qdrant_vdb.ingest_data(embeddata=embeddata)

                        # set up retriever
                        print("==== Start Retriever ====")
                        retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)

                        # set up rag
                        query_engine = RAG(retriever=retriever)

                        st.session_state.file_cache[file_key] = query_engine
                    else:
                        query_engine = st.session_state.file_cache[file_key]

                    # Inform the user that the file is processed and Display the PDF uploaded
                    st.success("Ready to Chat!")
                    display_pdf(uploaded_file)
                    print(query_engine)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()     
                torch.cuda.empty_cache()

with chart_col:
    with st.container():        
        # st.caption("""<img src="data:image/png;base64,{}" height="150" style="vertical-align: 0px;">""".format(base64.b64encode(open("assets/Qwen-logo-01.png", "rb").read()).decode()), unsafe_allow_html=True)
        
        st.button("Clear ↺", on_click=reset_chat)
        # Initialize chat history
        if "messages" not in st.session_state:
            reset_chat()
            torch.cuda.empty_cache()


        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        # Accept user input
        if prompt := st.chat_input("Ask about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                streaming_response = query_engine.query(prompt)
                        
                for chunk in streaming_response:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                    time.sleep(0.01)
                message_placeholder.markdown(full_response)
                print(full_response)
                print("Full Response Length:", len(full_response))

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

