import time
import json
from bs4 import BeautifulSoup
import tabula
import numpy as np
import pandas as pd

'''
The source file is 6.37GB in size and it did not fit into the laptop memory. 
Thus, the file is loaded as buffers to preserv memory and process line by line
'''
extracted_data = []
with open(r"D:\dev.jsonl",'r',buffering=1000, encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            #print(data)
            doc_title = data['document_title']
            question_text = data['question_text']
            yes_no_answer = data['annotations'][0]['yes_no_answer']
            long_answer_dict = data['annotations'][0]['long_answer']
            long_answer_start, long_answer_end = long_answer_dict['start_byte'], long_answer_dict['end_byte']
            long_answer_text = ""
            if long_answer_start != -1 and long_answer_end != -1:
                long_answer_byte = bytes(data['document_html'], 'utf-8')[long_answer_start : long_answer_end]
                long_answer_text = BeautifulSoup(long_answer_byte, "lxml").text
            short_answer_list = data['annotations'][0]['short_answers']
            temp_data = [doc_title, question_text.replace("\\r\\n", "").replace("\n", ""), yes_no_answer]
            if len(short_answer_list) > 1:
                print("Short answrs found: {} - {}".format(len(short_answer_list), question_text))
            for i, short_answer_dict in enumerate(short_answer_list):
                short_answer_start, short_answer_end = short_answer_dict['start_byte'], short_answer_dict['end_byte']
                short_answer_byte = bytes(data['document_html'], 'utf-8')[short_answer_start : short_answer_end]
                short_answer_text = BeautifulSoup(short_answer_byte, "lxml").text
                temp_data.append(short_answer_text.replace("\\r\\n", "").replace("\n", ""))
            extracted_data.append(temp_data)
        except Exception as ex:
            print(ex)
            pass
        #time.sleep(1)
df = pd.DataFrame(extracted_data)
columns = ['document_title', 'question_text', 'yes_no_answer']
for i in range(1, len(df.columns)-2):
    columns.append('short_answer{}'.format(i))
df.columns = columns
df.to_csv("clean_data.csv")