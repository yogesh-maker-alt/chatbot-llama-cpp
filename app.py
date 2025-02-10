import gradio as gr
from util import LlamaModel


with gr.Blocks() as demo:
    obj = LlamaModel("/home/neosoft/Downloads/Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf")

    gr.ChatInterface(
          type= "messages",
          fn=obj.inference,
          title="Chatbot using llama.cpp",
          description="This is a chatbot using llama.cpp",
          show_progress="full",
          theme="soft",


    )
            # button = gr.Button("Submit")
            # button.click(fn=obj.inference, inputs=input, outputs=gr.Textbox(label="Answer"))


demo.launch(share=True)