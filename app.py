import streamlit as st

from PyPDF2 import PdfReader

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain.chains.question_answering import load_qa_chain

from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
os.environ['PINECONE_API_KEY']=os.getenv('PINECONE_API_KEY')


def get_pdf_text(pdf_docs):

    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)

        for pages in pdf_reader.pages:

            text+=pages.extract_text()

    return text 


def get_chunks(text):

    textsplitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=textsplitter.split_text(text)

    return chunks



def get_vector_store(chunks):

    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index ='pdfwrap'
    PineconeVectorStore.from_texts(chunks, embedding, index_name=index)
    

def conversational_chain():

    prompt = """Answer the question based on the context below. If the
    question cannot be answered using the information provided answer
    with Dont Provide the Wrong Answer\n

    Context:{context}\n
    Question:{question}\n


    Answer:"""

    model=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)

    input_prompt=PromptTemplate(template=prompt,input_variables=['context','question'])

    chain=load_qa_chain(model,chain_type="stuff", prompt=input_prompt)

    return chain



def user_input(user_question):

     embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

     index ='pdfwrap'
     docsearch = PineconeVectorStore.from_existing_index(embedding=embedding, index_name=index)
     docs=docsearch.similarity_search(user_question)


     chain=conversational_chain()

     response=chain({
         
         "input_documents":docs,"question":user_question
     },return_only_outputs=True)


     print(response)

     st.write("Reply: ", response["output_text"])

    

    

def main():

  st.set_page_config(page_title="Chat with PDF Using Gemini", page_icon=":gemini:")

  st.header("Chat with PDF Using Gemini :gemini:")

  user_question=st.text_input("Ask a Question from pdf files")


  if(user_question):
      
      user_input(user_question)

  with st.sidebar:
      st.title("Menu:")
      st.set_option('deprecation.showfileUploaderEncoding', False)
      pdf_docs=st.file_uploader("Upload Your PDF Files and CLick on SUbmit",type=["pdf"], accept_multiple_files=True)

      if st.button("Submit & Process"):
          
          with st.spinner("Processing...."):
              
              raw_text=get_pdf_text(pdf_docs)
              chunks=get_chunks(raw_text)

              get_vector_store(chunks)

              st.write("Done")


if __name__=='__main__':

    main()