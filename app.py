import torch
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import base64

#st.set_page_config(layout="wide")
fig = plt.figure()
#st.set_page_config(layout="wide")

def add_bg_from_url(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url('white.jpeg') 


st.sidebar.header("Behind the scenes !")
st.sidebar.markdown('<div style="text-align: justify;">This emotion recongition module is a demonstration of our light-weight AI enabled Computer Vision Engine that identifies image pixels and classifies them into defined classes. Our read-to-deploy pipeline features: </div>', unsafe_allow_html=True)
st.sidebar.markdown("")
st.sidebar.subheader("- Minimal Training")
st.sidebar.subheader("- Accurate Results")
st.sidebar.subheader("- Edge compatible")

_num_beams = 4
_no_repeat_ngram_size = 3
_length_penalty = 2
_min_length = 30
_max_length = 200
_early_stopping = True





def main():
    
    with st.container():
        text = st.text_area('Text Input')
        
    def run_model(input_text):
        with st.spinner('Take a sip of your coffee, AI is extracting the summary..'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
            t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
            input_text = str(input_text).replace('\n', '')
            input_text = ' '.join(input_text.split())
            input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
            summary_task = torch.tensor([[21603, 10]]).to(device)
            input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
            summary_ids = t5_model.generate(input_tokenized,
                                        num_beams=_num_beams,
                                        no_repeat_ngram_size=_no_repeat_ngram_size,
                                        length_penalty=_length_penalty,
                                        min_length=_min_length,
                                        max_length=_max_length,
                                        early_stopping=_early_stopping)
            output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            st.write('Summary')
        st.success(output[0])

    if st.button('Submit'):
        run_model(text)

if __name__ == "__main__":
    with st.container():
        st.title('Abstractive Text Summarization using NLP')
        st.markdown("Powered by [Think In Bytes](https://www.thinkinbytes.in)")
    main()
    
    
    with st.container():    
        st.markdown("<h2 style='text-align: center; color: black;'>Neural language Processing - Applications</h2>", unsafe_allow_html=True)
        image = Image.open('screen2.png')
        st.image(image)
