import json
import re
import os
import pandas as pd
import datetime
from dateutil import parser
import warnings
import random
import csv
def splitdict(string):
    i = 0
    dict_list = []
    for index in range(len(string) - 2):
        if string[index] == "}" and string[index+1] == "{" and string[index+2].isalpha() == False:
            sub_string = string[i:index+1]
            i = index + 1
            try:
                dict1 = json.loads(sub_string)
                dict_list.append(dict1)
            except Exception as e:
                dict1 = []
                #print(sub_string)
                print("ERROR!")
                print(e)
                pass
    sub_string = string[i:]
    dict2 = json.loads(sub_string)
    dict_list.append(dict2)
    return dict_list


def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    
    return tweet

def Extraction_comparison_of_keyword(keyword, device_num, article_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    warnings.filterwarnings("ignore")
    from sentence_transformers import SentenceTransformer, util
    comparison_model = SentenceTransformer('all-MiniLM-L12-v2')
    corpus_main = ""
    output_main = ""
    keyword_dir = os.path.join(corpus_main, keyword)
    for each_day in os.listdir(keyword_dir):
        day_dir = os.path.join(keyword_dir, each_day)
        for each_topic in os.listdir(day_dir):
            topic_dir = os.path.join(day_dir, each_topic)
            output_list = []
            output_file = output_main + keyword + "/" + each_day + "/" + each_topic + "/" + "output.csv"
            for each_file in os.listdir(topic_dir):
                if each_file == "output.txt":
                    file_dir = os.path.join(topic_dir, each_file)
                    with open(file_dir, 'r', encoding = 'utf-8') as f:
                        summary_string = f.read()
                        for each_article in article_list:
                            article_full_text_embedding = comparison_model.encode(each_article[2][0].lower(), convert_to_tensor = True)
                            summary_embedding = comparison_model.encode(summary_string.lower(), convert_to_tensor = True)
                            similarity = util.pytorch_cos_sim(article_full_text_embedding, summary_embedding).item()
                            out_article = each_article.copy()
                            out_article.append(similarity)
                            output_list.append(out_article)
            out_df = pd.DataFrame(output_list)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            out_df.to_csv(output_file, index = False)
def Dup_removal(keyword):
    warnings.filterwarnings("ignore")
    corpus_main = ""
    keyword_dir = os.path.join(corpus_main, keyword)
    for each_day in os.listdir(keyword_dir):
        print("Finished " + keyword + " date " + each_day) 
        day_dir = os.path.join(keyword_dir, each_day)
        for each_topic in os.listdir(day_dir):
            topic_dir = os.path.join(day_dir, each_topic)
            input_file = topic_dir + "/" + "output.csv"
            output_file = topic_dir + "/" + "output_no_dup.csv"
            with open(input_file, 'r', encoding = 'utf-8') as f:
                df = pd.read_csv(input_file)
                out_df = df.drop_duplicates(subset=["2"])
                out_df.to_csv(output_file, index = False)
def Comparison_analysis(keyword):
    warnings.filterwarnings("ignore")
    corpus_main = ""
    output_dir = ""
    keyword_dir = os.path.join(corpus_main, keyword)
    output_list = []
    for each_day in os.listdir(keyword_dir):
        #print("Finished " + keyword + " date " + each_day) 
        day_dir = os.path.join(keyword_dir, each_day)
        for each_topic in os.listdir(day_dir):
            topic_dir = os.path.join(day_dir, each_topic)
            input_file = topic_dir + "/" + "output_no_dup.csv"
            with open(input_file, 'r', encoding = 'utf-8') as f:
                df = pd.read_csv(input_file)
                overall_similarity = list(df["6"])
                thresholds = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]
                count_list = []
                for threshold in thresholds:
                    count = sum(1 for number in overall_similarity if number > threshold)
                    count_list.append(count)
            result = [keyword, each_day, each_topic, count_list]
            output_list.append(result)
    output_dir = "/" + keyword + "/" #your own output directory here
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Keyword', 'Date', "Topic", "Count_list"])
        for line in output_list:
            csv_writer.writerow(line)    
def Comparison_analysis_max(keyword):
    warnings.filterwarnings("ignore")
    corpus_main = ""
    output_dir = ""
    keyword_dir = os.path.join(corpus_main, keyword)
    output_list = []
    for each_day in os.listdir(keyword_dir):
        #print("Finished " + keyword + " date " + each_day) 
        day_dir = os.path.join(keyword_dir, each_day)
        for each_topic in os.listdir(day_dir):
            topic_dir = os.path.join(day_dir, each_topic)
            input_file = topic_dir + "/" + "output_no_dup.csv"   
            with open(input_file, 'r', encoding = 'utf-8') as f:
                df = pd.read_csv(input_file)
                overall_similarity = list(df["6"])
                max_num = max(overall_similarity)
                result = [keyword, each_day, each_topic, max_num]
                output_list.append(result)
    output_dir = "" + keyword + ""
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Keyword', 'Date', "Topic", "Max_similarity"])
        for line in output_list:
            csv_writer.writerow(line)
            
def Extraction_summary_of_keyword(keyword, device_num):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    from summarizer.sbert import SBertSummarizer
    from summarizer import Summarizer
    warnings.filterwarnings("ignore")
    #proxy = {'https': ''}
    corpus_main = ""
    #output_main = ""
    custom_config = AutoConfig.from_pretrained('sentence-transformers/paraphrase-albert-small-v2', proxies = proxy)
    custom_config.output_hidden_states=True
    custom_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-albert-small-v2', proxies = proxy)
    custom_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-albert-small-v2', config=custom_config, proxies = proxy)
    model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
    corpus_main = ""
    output_main = ""
    corpus = ""
    keyword_dir = os.path.join(corpus_main, keyword)
    for each_day in os.listdir(keyword_dir):
        day_dir = os.path.join(keyword_dir, each_day)
        for each_topic in os.listdir(day_dir):
            topic_dir = os.path.join(day_dir, each_topic)
            output_file = output_main + keyword + "/" + each_day + "/" + each_topic + "/" + "output.txt"
            corpus = ""
            for each_file in os.listdir(topic_dir):
                file_dir = os.path.join(topic_dir, each_file)
                with open(file_dir, 'r', encoding = 'utf-8') as f:
                    try:
                        listdict = splitdict(f.read())
                        for x in listdict:
                            if 'retweeted_status' in x:
                                full_text = cleaner(x['retweeted_status']['full_text'])
                            else:
                                full_text = cleaner(x['full_text'])
                            #print(full_text)
                            #print(x['full_text'])
                            corpus = corpus + "\n" + full_text
                    except Exception as e:
                        print(e)
                if len(corpus) > 1000000:
                    corpus = corpus[0:999999]
            summarized_tweets = model(corpus, num_sentences = 20)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summarized_tweets)