import streamlit as st 
st.set_page_config(
    layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title='Classify German News',  # String or None. Strings get appended with "• Streamlit". 
    page_icon='../../resources/favicon.png',  # String, anything supported by st.image, or None.
)
import pandas as pd
import numpy as np
import os
import base64
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
import streamlit_apps_config as config

root_path = config.project_path


style_config = config.STYLE_CONFIG
style_config = style_config.replace('height: 500px;', 'min-height: 200px; max-height: 500px; background-color: #F1F2F6; line-height: 2.0;')
st.markdown(style_config, unsafe_allow_html=True)

########## To Remove the Main Menu Hamburger ########

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

########## Side Bar ########

##-- Hrishabh Digaari: FOR SOLVING THE ISSUE OF INTERMITTENT IMAGES & LOGO-----------------------------------------------------
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img height="90%" width="90%" src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code

logo_html = get_img_with_href('../../resources/jsl-logo.png', 'https://www.johnsnowlabs.com/')
st.sidebar.markdown(logo_html, unsafe_allow_html=True)


## loading logo (older version without href)
# st.sidebar.image('../../resources/jsl-logo.png', use_column_width=True)

### loading description file
descriptions = pd.read_json('../../streamlit_apps_descriptions.json')
descriptions = descriptions[descriptions['app'] == 'CLASSIFICATION_DE_SENTIMENT']

### showing available models
model_names = descriptions['model_name'].values
st.sidebar.title("Choose the pretrained model to test")
selected_model = st.sidebar.selectbox("", list(model_names))#input_files_list)

## displaying selected model's output
app_title = descriptions[descriptions['model_name'] == selected_model]['title'].iloc[0]
app_description = descriptions[descriptions['model_name'] == selected_model]['description'].iloc[0]

### Creating Directory Structure

TEXT_FILE_PATH = os.path.join(os.getcwd(),'inputs/'+selected_model)
RESULT_FILE_PATH = os.path.join(os.getcwd(),'outputs/'+selected_model)

input_files = sorted(os.listdir(TEXT_FILE_PATH))
input_files_paths = [os.path.join(TEXT_FILE_PATH, fname) for fname in input_files]
result_files_paths = [os.path.join(RESULT_FILE_PATH, fname.split('.')[0]+'.json') for fname in input_files]

title_text_mapping_dict={}
title_json_mapping_dict={}
for index, file_path in enumerate(input_files_paths):
    lines = open(file_path, 'r', encoding='utf-8').readlines()
    title = lines[0]
    text = lines[1]
    title_text_mapping_dict[title] = text
    title_json_mapping_dict[title] = result_files_paths[index]


######## Main Page ################

st.title(app_title)
st.markdown("<h2>"+app_description+"</h2>", unsafe_allow_html=True)
st.subheader("")

#st.subheader(app_description)
# selection_text = "Pick a question from the below list"
# if selected_model=='classifierdl_use_spam2':
#     selection_text="Pick a message from the below list"
# elif selected_model=='classifierdl_use_fakenews2':
#     selection_text="Pick a news from the below list"
selection_text="Pick a news from the below list"
selected_title = st.selectbox(selection_text, list(title_text_mapping_dict.keys()))
selected_text = title_text_mapping_dict[selected_title]
# print(selected_title)
# print(selected_text)
#if selected_model == 'classifierdl_use_fakenews':
#if len(selected_text) 
st.subheader('Text to analyze')
st.markdown(config.HTML_WRAPPER.format(selected_text), unsafe_allow_html=True)
#text = st.text_area("Text to analyze. ", selected_text, height=min(500, len(selected_text)//4))
# text = st.text_area("Text to analyze. ", selected_text)
selected_file = title_json_mapping_dict[selected_title]
df = pd.read_json(selected_file)

class_ = df.iloc[0]['sentiment'][3]
score_ = round(float(df.iloc[0]['sentiment'][4][class_])*100, 2)

#classes_mapping_dict = {
#    "ABBR": "ABBREVIATIONS OF CONCEPTS",
#    "DESC": "DESCRIPTIONS AND ABSTRACT CONCEPTS",
#    "NUM": "NUMERIC VALUES (Post codes, Dates, Prices, Fractions, Speed, Temperature)",
#    "ENTY": "ENTITIES (Organisations, Animals, Food, Plants, Currency)",
#    "LOC": "LOCATION (Cities, Countries, Mountains)",
#    "HUM": "Human (group of people)"
#}
# parent = [" ABBR", " DESC", " NUM", " ENTY", " LOC", " HUM"]
# parent_expanded=["Abbreviation", "Description", "Numeric Values", "Entities", "Locations", "Human Beings" ]
# parent_meaning = ["ABBREVIATIONS OF CONCEPTS", "DESCRIPTIONS AND ABSTRACT CONCEPTS", "NUMERIC VALUES (Post codes, Dates, Prices, Fractions, Speed, Temperature)", "ENTITIES (Organisations, Animals, Food, Plants, Currency)", "LOCATION (Cities, Countries, Mountains)", "Human (group of people)"]
# enty = [" ENTY_animal"," ENTY_body"," ENTY_color"," ENTY_cremat"," ENTY_currency"," ENTY_dismed"," ENTY_event"," ENTY_food"," ENTY_instru"," ENTY_lang"," ENTY_letter"," ENTY_other"," ENTY_plant"," ENTY_product"," ENTY_religion"," ENTY_sport"," ENTY_substance"," ENTY_symbol"," ENTY_techmeth"," ENTY_termeq"," ENTY_veh"," ENTY_word"]
# enty_meaning = ["animals","organs of body","colors","inventions, books and other creative pieces","currency names","diseases and medicine","events","food","musical instrument","languages","letters like a-z","other entities","plants","products","religions","sports","elements and substances","symbols and signs","techniques and methods","equivalent terms","vehicles","words with a special property"]
# desc = [" DESC_def"," DESC_desc"," DESC_manner"," DESC_reason"]
# desc_meaning = ["definition of sth.","description of sth.","manner of an action","reasons"]
# hum = [" HUM_gr"," HUM_ind"," HUM_title"," HUM_desc"]
# hum_meaning = ["a group or organization of persons","an individual","title of a person","description of a person"]
# loc = [" LOC_city"," LOC_country"," LOC_mount"," LOC_other"," LOC_state"]
# loc_meaning = ["cities","countries","mountains","other locations","states"]
# num = [" NUM_code"," NUM_count"," NUM_date"," NUM_dist"," NUM_money"," NUM_ord"," NUM_other"," NUM_period"," NUM_perc"," NUM_speed"," NUM_temp"," NUM_volsize"," NUM_weight"]
# num_meaning = ["postcodes or other codes","number of sth.","dates","linear measures","prices","ranks","other numbers","the lasting time of sth.","fractions","speed","temperature","size, area and volume","weight"]
# abbr = [' ABBR_abb', ' ABBR_exp']
# abbr_meaning = ['abbreviation', 'expression abbreviated']

# final_keys = parent+enty+desc+hum+loc+num+abbr
# final_vals = parent_meaning + enty_meaning + desc_meaning + hum_meaning + num_meaning + abbr_meaning

# classes_mapping_dict = dict(zip(final_keys, final_vals))
# classes_mapping_dict['spam'] = 'A Spam Message &#x1F5D1;'
# classes_mapping_dict['ham'] = 'Not a Spam Message &#x1F4E9;'

# if selected_model == 'classifierdl_use_spam':
#     result = 'This message has been classified as : **'
# elif selected_model == 'classifierdl_use_fakenews':
#     result = 'This news has been classified as : **'
# else:
#     result = 'This sentence has been classified as : **'

# try:
#     temp_dict_expanded=dict(zip(parent, parent_expanded))
#     class_explanation = classes_mapping_dict[class_].title()
#     if "_" in class_:
#         splts = class_.split('_')
#         class_ = temp_dict_expanded[splts[0]]+' ('+splts[1]+')'

#     result += class_ + " - " + class_explanation
# except:
#     result += class_.upper()# + " - " + class_.title()
# result += '**'

#------RESULT-----#

result = f"This sentence has been classified as : **{class_.title()}**"
st.markdown(result)

st.markdown("Classification Confidence: **{}%**".format(score_))

st.title("")

try_link="""<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_TR_NEWS.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Try it yourself:')
st.sidebar.markdown(try_link, unsafe_allow_html=True)