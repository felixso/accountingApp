# Import required packages
# ======================================================================================================================
import keras_nlp
import streamlit as st
from langchain.llms.base import LLM
from langchain.schema import Document
from langchain.chains import RetrievalQA
from typing import Any, Iterator, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.outputs import GenerationChunk
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


# Define global vaiables
# ======================================================================================================================

# Define keras_nlp model to be used with Langchain
# .............................................................................
gpt2 = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")


# Define required class and functions
# ======================================================================================================================
class GPT2LLM(LLM):
    """
    A class which overrides LLM class from Langchain.
    This LLM class is useful when we want to define custom llm. This custom
    llm can be private/local API or private/local model as well.
    """
    num_output:int = 128

    @property
    def _llm_type(self) -> str:
        """
        Get the type of language model used by this chat model.
        Used for logging purposes only
        """
        return "GPT2 Large Model"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string.
        """
        print(f"{'*'*25} Prompt is \n {prompt}")

        response = gpt2.generate(prompt, max_length=self.num_output)

        return response
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        response = ""
        for token in gpt2.generate(prompt, max_length=self.num_output):
            response += token
            yield response


# Define global context for LLM and Embedings models
# ======================================================================================================================
llm = GPT2LLM()  # Using GPT2LLM as the language model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=100)


# Data Loading, Vector Embeddings Generation, Processing
# ======================================================================================================================
# Incase we want entire text file as single document we can use below code.
# documents = DirectoryLoader("/home/nayan/NLP/RAG/Scripts", glob="*.txt").load()

# Below code is to consider each sentence as separate document
# ..................................................................................................
# Read the text file and split into sentences
with open("sentences.txt", "r") as file:
    text = file.read()

# Assuming the sentences are separated by periods. Adjust as needed.
sentences = text.split('.')  # Split based on your preferred delimiter
print(type(sentences))  # It will be list data type

# Create Document objects for each sentence
documents = [Document(page_content=sentence) for sentence in sentences if sentence.strip()]

# Split the text into chunks (if needed)
split_docs = text_splitter.split_documents(documents)
# ..................................................................................................

# Generate vector embedings using the model defined by Settings.embed_model 
# Create the FAISS vector store and store embeddings
index = FAISS.from_documents(split_docs, embed_model)


# Load index and generate the asnwer
# ======================================================================================================================
# Set up the query engine using LangChain's RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.as_retriever()
)

# REF URL: https://docs.streamlit.io/develop/api-reference/widgets/st.text_area
txt = st.text_area("Question", "Ask questions to LLM!!!")

# Submission button
if st.button("Submit"):
    
    # Retrieve the answer for the submitted question
    response = qa_chain.invoke(txt)
    
    # Display the response
    st.write("Answer:", response["result"])