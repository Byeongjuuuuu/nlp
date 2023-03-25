import streamlit as st
from pyparsing import empty

from transformers import(
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    XLMRobertaTokenizerFast,
    BertTokenizerFast,
)
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from manga_ocr import MangaOcr

import cv2
import numpy as np
import urllib.request
from bs4 import BeautifulSoup as bs
import requests

if 'src' not in st.session_state:
    st.session_state.src = ""
    
if 'img' not in st.session_state:
    st.session_state.img = None
    
if 'after' not in st.session_state:
    st.session_state.after = ""

if 'final' not in st.session_state:
    st.session_state.final = ""

encoder_model_name_1 = "cl-tohoku/bert-base-japanese-v2"
encoder_model_name_2 = "xlm-roberta-base"
decoder_model_name = "skt/kogpt2-base-v2"
if 'tokenizer' not in st.session_state:
    src_tokenizer_1 = BertTokenizerFast.from_pretrained(encoder_model_name_1)
    src_tokenizer_2 = XLMRobertaTokenizerFast.from_pretrained(encoder_model_name_2)
    trg_tokenizer = PreTrainedTokenizerFast.from_pretrained(decoder_model_name)
    st.session_state.tokenizer = src_tokenizer_1, src_tokenizer_2, trg_tokenizer
else:
    src_tokenizer_1, src_tokenizer_2, trg_tokenizer = st.session_state.tokenizer

@st.cache_data
def get_model_1(bos_token_id):
    model = EncoderDecoderModel.from_pretrained('/content/drive/MyDrive/구름/Project_3/dump2/best_model')
    model.config.decoder_start_token_id = bos_token_id
    model.eval()
    model.cuda()

    return model

@st.cache_data
def get_model_2(bos_token_id):
    model = EncoderDecoderModel.from_pretrained('/content/drive/MyDrive/구름/Project_3/찬우님모델')
    model.config.decoder_start_token_id = bos_token_id
    model.eval()
    model.cuda()

    return model

@st.cache_data
def get_ocr():
    ocr = MangaOcr()
    
    return ocr

def get_src(query):
    src = ""
    url = f"https://www.google.com/search?q={query}&rlz=1C5CHFA_enKR1025KR1027&sxsrf=AJOqlzXt2qww_s_QT6YIWsgprlaDwJt6Ew:1677046926223&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiNnrzSvqj9AhWcpVYBHayKDhsQ_AUoAXoECAEQAw&biw=1427&bih=709&dpr=2"

    headers = {"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.1"}
    req = requests.get(url, headers=headers)
    soup = bs(req.text, 'lxml')
    images = soup.find_all("img", attrs={'class':'rg_i Q4LuWd'})

    for img in images:
        try:
            src = img['data-src']
            break
        except KeyError:
            continue
    
    return src

def url_to_image(url):
  resp = urllib.request.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype='uint8')
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)

  return image

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(layout='wide')
col1, empty1 ,col2 = st.columns([1.0, 0.1, 1.0])

img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = '#0000FF'
view_img = st.sidebar.checkbox(label="번역 결과 이미지 보기", value=True)
stroke_width = st.sidebar.slider("펜의 두께: ", 1, 25, 3)
bg_color = "#eee"
drawing_mode = 'freedraw'

with col1:
    st.header("일-한 이미지 번역기")
    st.subheader("일-한 이미지 번역기에 오신 것을 환영합니다!")
    if img_file:
        img = Image.open(img_file)
        if not realtime_update:
            st.write("관심 영역을 설정한 후 더블 클릭해주세요.")
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color)
        if st.button("번역!", help="해당 일본어를 번역합니다."):
            model_1 = get_model_1(trg_tokenizer.bos_token_id)
            model_2 = get_model_2(trg_tokenizer.bos_token_id)
            ocr = get_ocr()
            src_text = ocr(cropped_img)
            st.session_state.img = cropped_img
            embeddings_1 = src_tokenizer_1(src_text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
            embeddings_2 = src_tokenizer_2(src_text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
            embeddings_1 = {k: v.cuda() for k, v in embeddings_1.items()}
            embeddings_2 = {k: v.cuda() for k, v in embeddings_2.items()}
            output_1 = model_1.generate(**embeddings_1)[0, 1:-1].cpu()
            output_2 = model_2.generate(**embeddings_2)[0, 1:-1].cpu()
            translated_1 = trg_tokenizer.decode(output_1)
            translated_2 = trg_tokenizer.decode(output_2)
            st.text_area("입력된 일본어", value=src_text, disabled=False)
            st.session_state.src = src_text
            st.text_area("모델 1", value=translated_1, disabled=True)
            st.text_area("모델 2", value=translated_2, disabled=True)
            if view_img:
                  src_1 = get_src(translated_1)
                  src_2 = get_src(translated_2)
                  st.image(url_to_image(src_1), channels="BGR")
                  st.caption(f"사진 출처: {src_1}")
                  st.image(url_to_image(src_2), channels="BGR")
                  st.caption(f"사진 출처: {src_2}")
            
with col2:
    if st.session_state.img is not None:
        st.image(st.session_state.img)
    
    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=150,
    drawing_mode=drawing_mode,
    point_display_radius=0,
    key="canvas",
    )
    
    prev = st.text_input('이전 글자', st.session_state.src)
    # 어떤 글자로 인식됐는지 여기서 보여주기
    
    if st.button(label='글자 추가하기', help='입력 일본어에 추가합니다.'):
        drawn = Image.fromarray(canvas_result.image_data)
        ocr = get_ocr()
        text = ocr(drawn)
        prev += text
        st.session_state.after = prev
    st.session_state.final = st.text_input(label='이후 글자', value=st.session_state.after)
    if st.button(label='번역!', help='입력된 일본어를 번역합니다.'):
        st.session_state.after = ""
        model_1 = get_model_1(trg_tokenizer.bos_token_id)
        model_2 = get_model_2(trg_tokenizer.bos_token_id)
        embeddings_1 = src_tokenizer_1(st.session_state.final, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
        embeddings_2 = src_tokenizer_2(st.session_state.final, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
        embeddings_1 = {k: v.cuda() for k, v in embeddings_1.items()}
        embeddings_2 = {k: v.cuda() for k, v in embeddings_2.items()}
        output_1 = model_1.generate(**embeddings_1)[0, 1:-1].cpu()
        output_2 = model_2.generate(**embeddings_2)[0, 1:-1].cpu()
        translated_1 = trg_tokenizer.decode(output_1)
        translated_2 = trg_tokenizer.decode(output_2)
        
        st.text_area("모델 1", value=translated_1, disabled=True)
        st.text_area("모델 2", value=translated_2, disabled=True)
        if view_img:
            src_1 = get_src(translated_1)
            src_2 = get_src(translated_2)
            st.image(url_to_image(src_1), channels="BGR")
            st.caption(f"사진 출처: {src_1}")
            st.image(url_to_image(src_2), channels="BGR")
            st.caption(f"사진 출처: {src_2}")

with empty1:
    empty()
