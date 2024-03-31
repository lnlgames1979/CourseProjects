import streamlit as st
import pandas as pd

import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib # để lưu và đọc các mô hình đã lưu
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex as re

from gensim import corpora, models, similarities
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from sklearn.cluster import KMeans
import plotly.express as px


chPositiveSymbol= 'positive' # positive or plus
chNegativeSymbol= 'negative' # negative or minus
chNeutralSymbol= 'neutral' # neutral or none
chSpamSymbol = 'spam' # spam
input1Col= 'Comment_std'
labelCol= 'cmt_Label'

rColName= 'Recency'
fColName= 'Frequency'
mColName= 'Monetary'
clusterCol= 'Grade'

SVIP= 'Siêu VIP'
KHTX= 'KH thường xuyên'
KHBig= 'Khách sộp'
KHMid= 'KH tầm trung'
KHVL= 'KH vãng lai'
KHVD= 'Khách vô danh'
KHTH= 'Khách trả hàng'
def RFMGrade(df):
    if df['CustomerID'] < 0: # khách không có mã KH
        return KHVD
    elif df[mColName] <= 0:
        return KHTH
    elif (df.R >= 2)and(df.F >= 2)and(df.M == 4): # khách thường xuyên chi tiêu lớn
        return SVIP
    elif (df.R >= 3)and(df.F >= 3): # khách thường xuyên giao dịch
        return KHTX
    elif (df.M == 4): # khách chi tiêu lớn
        return KHBig
    elif (df.R > 1)and(df.F > 1)and(df.M > 1): # khách bình thường
        return KHMid
    else:
        return KHVL

def newRFMGrade(df, Rmin, Rmax, Fmin, Fmax, Mmin, Mmax, sName):
    if isinstance(df, str):
        return 'Error: Invalid input'  # Trả về một giá trị mặc định hoặc thông báo lỗi nếu df là một chuỗi
    if df['CustomerID'] < 0: # khách không có mã KH
        return KHVD
        
    countCond= len(sName)
    for i in range(countCond):
        if(Rmin[i] <= df.R <= Rmax[i])and(Fmin[i] <= df.F <= Fmax[i])and(Mmin[i] <= df.M <= Mmax[i]):
            return sName[i]
    return 'Nhóm khác'
    
def LoadDicts(path):
    file = open(path + r'vietnamese-stopwords.txt', 'r', encoding="utf8")
    stopwords_lst = file.read().split('\n')
    file.close()
    return stopwords_lst

def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    negative_lst= ['không', 'chẳng', 'chả']
    # không, chẳng, chả...
    for w in negative_lst:
        if w in text_lst:
            while i <= len(text_lst) - 1:
                word = text_lst[i]
                #print(word)
                #print(i)
                if word == w:
                    next_idx = i+1
                    if next_idx <= len(text_lst) -1:
                        word = word +'_'+ text_lst[next_idx]
                    i= next_idx + 1
                else:
                    i = i+1
                new_text = new_text + word + ' '
        else:
            new_text = text
    return new_text.strip()

def process_postag_thesea(text):
    new_document = ''  # Chuỗi mới chứa văn bản đã xử lý
    for sentence in sent_tokenize(text):  # Duyệt qua mỗi câu trong văn bản
        sentence = sentence.replace('.', '')  # Loại bỏ dấu chấm
        # Loại bỏ các từ không phải là danh từ, danh từ riêng, tính từ, phó từ, v.v. khỏi mỗi câu
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        sentence = ' '.join(word[0] if word[1].upper() in lst_word_type else ''
                            for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '  # Ghép câu đã xử lý vào văn bản mới
    new_document = re.sub(r'\s+', ' ', new_document).strip()  # Loại bỏ khoảng trắng dư thừa
    return new_document  # Trả về văn bản đã xử lý

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = re.sub(r'\s+', ' ', document).strip()
    return document

class DoAnMonHoc:
    def VN_preprocess(self, txt): # project 1
        if self.stopwords_lst is None:
            self.stopwords_lst= LoadDicts(self.paths[self.iProject])
        cmt= process_postag_thesea(txt)
        cmt= remove_stopword(cmt, self.stopwords_lst)
        cmt= 'trung tính' if len(cmt) == 0 else cmt
        return cmt
        
    def preprocess_text(self, s): # project 2
        if self.stopwords_lst is None:
            self.stopwords_lst= LoadDicts(self.paths[self.iProject])
        words = word_tokenize(s)  # Chỉ cần gọi word_tokenize(text) mà không cần thêm format='text'
        words = [re.sub(r'[0-9]+', '', word) for word in words]  # Điều chỉnh cú pháp của re.sub()
        words_re = [word.lower() for word in words if word not in ['', ' ', ',', '.', '...', '-', ':', ';', '?', '%', '_%', '(', ')', '+', '/', 'g', 'ml']]
        words_re = [word for word in words_re if word not in self.stopwords_lst]  # Điều chỉnh vị trí của điều kiện
        return words_re

    def prj1(self):
        model_path = self.paths[self.iProject] + 'tfidf.joblib'
        tfidf= joblib.load(model_path)
        
        model_path = self.paths[self.iProject] + 'svc_model.joblib'
        predictModel= joblib.load(model_path)
        
        self.comment= st.text_input("Express your feeling:", "tuyệt vời")
        if st.button("Submit"):
            self.cmtLabel= self.VN_preprocess(self.comment)
            self.cmtLabel= tfidf.transform([self.cmtLabel])
            self.cmtLabel= predictModel.predict(self.cmtLabel)
            st.write(f"Your status: {self.comment} -> {self.cmtLabel}")
        # -----------------------------------------------------------------------------------------------------
        st.write('-'*50)
        # Đọc file csv tương ứng với nút nhấn được chọn
        self.info_df = pd.read_csv(self.paths[self.iProject] + self.infoFiles[self.iProject], index_col= 0)
        # st.write("Selected Info:")
        # st.write(self.info_df)
        
        self.data_df= pd.read_csv(self.paths[self.iProject] + self.dataFiles[self.iProject], index_col= 0)
        # st.write("Selected Data:")
        # st.write(self.data_df)
        
        # Chọn một mục từ DataFrame
        selected_item = st.selectbox("Select a restaurant:", self.info_df['ID'].astype(str) + " - " + self.info_df['Restaurant'] + " - " + self.info_df['Address'])
        
        # Xuất ra thông tin của mục được chọn
        if selected_item:
            selID = int(selected_item.split(" - ")[0])
            selRow = self.info_df[self.info_df['ID'] == selID]
            
            st.write("Restaurant ID:", selRow['ID'].iloc[0])
            st.write("Name:", selRow['Restaurant'].iloc[0])
            st.write("Address:", selRow['Address'].iloc[0])
            st.write("Price:", selRow['Price'].iloc[0])
            st.write("Time:", selRow['Time'].iloc[0])
            st.write(f"Rating: {round(selRow['cmt_Rate'].iloc[0], 2)}/10")
            st.write(f"Positive: {selRow['positive'].iloc[0]}/{selRow['total'].iloc[0]}")
            st.write(f"Negative: {selRow['negative'].iloc[0]}/{selRow['total'].iloc[0]}")

            st.write('-'*50)
            
            self.DrawWordCloud(restID= selID, labelText= chPositiveSymbol)
            if self.wordcloud is not None:
                st.write('Positive words')
                st.image(self.wordcloud)
            else:
                st.write("No WordCloud available.")
        
            st.write('-'*50)
            
            self.DrawWordCloud(restID= selID, labelText= chNegativeSymbol)
            if self.wordcloud is not None:
                st.write('Negative words')
                st.image(self.wordcloud)
            else:
                st.write("No WordCloud available.")

    def prj2(self):
        self.info_df = pd.read_csv(self.paths[self.iProject] + self.infoFiles[self.iProject], index_col= 0)
        st.write("Loaded data:")
        st.dataframe(self.info_df)
        st.write(f'There has {len(self.info_df)} items.')
        
        st.header('Content base - apply GenSim') #-------------------------------------------------------------
        dictionary = corpora.Dictionary.load(self.paths[self.iProject] + 'Dict_gs.dict')
        tfidf_model = models.TfidfModel.load(self.paths[self.iProject] + self.modelFiles[self.iProject][0])
        # Load corpus
        tfidf_corpus = np.load(self.paths[self.iProject] + 'tfidf_corpus.npy', allow_pickle=True)
        # Load ma trận tương tự cosine
        index = similarities.SparseMatrixSimilarity.load(self.paths[self.iProject] + 'index.index')
        
        self.searchText= st.text_input("Enter a text to search:", "áo thun việt nam")
        n= st.slider("Number of similar:", min_value=1, max_value=20, value=5)
        if st.button("Submit"):
            s_preprocessed= self.preprocess_text(self.searchText)
            s_bow = dictionary.doc2bow(s_preprocessed)
            s_tfidf = tfidf_model[s_bow]
            # Tìm n phần tử có nội dung gần với câu s nhất
            sims = index[s_tfidf]
            
            n_most_similar_indices = sims.argsort()[::-1][:n]
            st.write(f'Similar item indexes: {n_most_similar_indices}')
            
            st.dataframe(self.info_df.iloc[n_most_similar_indices])
            
        st.write('-'*50)
        st.header('Content base - apply Cosine similarity') #-------------------------------------------------------------
        tf, tfidf_matrix = joblib.load(self.paths[self.iProject] + self.modelFiles[self.iProject][1])
        self.searchText = st.text_input("Enter a text to search:", "áo thun việt nam", key="search_text_input")
        m = st.slider("Number of similar:", min_value=1, max_value=20, value=5, key="similar_slider")
        if st.button("Submit", key="submit_button"):
            sc_preprocessed= [' '.join(self.preprocess_text(self.searchText))] # list các chuỗi đã xử lý
            
            tf_ws_texts = tf.transform(sc_preprocessed)
            # Tính ma trận tương tự cosine giữa các câu
            cosine_similarities = cosine_similarity(tf_ws_texts, tfidf_matrix)
            similar_indices = cosine_similarities[0].argsort()[::-1][:m]  # Lấy m câu tương tự, bỏ qua câu gốc
            st.dataframe(self.info_df.iloc[similar_indices])
        
        st.write('-'*50)
        st.header('User base (Collaborative Filtering) - apply Surprise SVD') #-------------------------------------
        with open(self.paths[self.iProject] + self.modelFiles[self.iProject][2], 'rb') as file:
            svd_alg = pickle.load(file)
        self.data_df= pd.read_csv(self.paths[self.iProject] + self.dataFiles[self.iProject], header= 0, delimiter= '\t')
        self.data_df= self.data_df[:10_000] # giảm bớt theo số lượng dữ liệu đã train
        # st.dataframe(self.data_df)
        userIdCol= 1
        userID_df= pd.DataFrame(self.data_df[self.data_df.columns[userIdCol]].unique())
        # st.dataframe(userID_df)
        st.write(f'There are {len(userID_df)} users.')
        userID = st.selectbox('Select a userID:', userID_df[0])
        n= st.slider("Number of product:", min_value=1, max_value=10, value=5)
        if userID != "":
            userID= int(userID)
            products_df = self.data_df[["product_id"]]
            products_df= products_df.drop_duplicates() # xóa bỏ các sản phẩm trùng lắp
            products_df['EstimateScore'] = products_df['product_id'].apply(lambda x: svd_alg.predict(userID, x).est) # est: get EstimateScore
            # Sắp xếp rating dự đoán giảm dần
            products_df = products_df.sort_values(by=['EstimateScore'], ascending=False)[:n]
            st.write(f'Các ID sản phẩm được đề xuất cho userID {userID}')
            st.dataframe(products_df)
            
        average_ratings = self.data_df.groupby('product_id')['rating'].mean().reset_index()
        # products_df[products_df.EstimateScore >= 5].head(n)
        average_ratings= average_ratings.sort_values(by='rating', ascending=False)[:n]
        st.write(f'Các ID sản phẩm được đề xuất cho khách mới')
        st.dataframe(average_ratings)

    def prj3(self):
        self.data_df= pd.read_csv(self.paths[self.iProject] + self.dataFiles[self.iProject], index_col= 0)
        st.write("Loaded data:")
        st.write(self.data_df)
        st.write(f'There has {len(self.data_df)} items.')
        kmean_df= self.data_df[[rColName, fColName, mColName]]
        self.DrawElbow(kmean_df)
        
        if self.KMeanElbowImg is not None:
            st.header('K-means elbow diagram')
            st.image(self.KMeanElbowImg)
            
            cluster_k= [str(k + 1) for k in range(20)]
            k= st.slider("Number of cluster:", min_value=1, max_value=20, value=5)
            
            kmodel= KMeans(n_clusters= k, random_state= 42)
            kmodel.fit(kmean_df)
            kmean_df[clusterCol]= kmodel.labels_
            kmean_agg= kmean_df.groupby(clusterCol).agg({rColName: 'mean', fColName: 'mean', mColName: ['mean', 'count']})
            kmean_agg.columns= kmean_agg.columns.droplevel() # xóa bỏ các cấp độ đặt tên cột sau khi group và tính toán theo các hàm agg
            kmean_agg.columns= ['R_mean', 'F_mean', 'M_mean', 'Count'] # đặt lại tên các cột
            kmean_agg['Percent']= round(kmean_agg['Count']/kmean_agg['Count'].sum()*100, 2)
            kmean_agg.reset_index(inplace= True)
            self.DrawScatter(kmean_agg)
        else:
            st.write("No record available.")

        st.write('-'*5)
        
        st.header('Clustering by R F M')
        
        iCluster= st.slider("Number of cluster:", min_value=1, max_value=20, value=6)
        Rmin= []
        Rmax= []
        Fmin= []
        Fmax= []
        Mmin= []
        Mmax= []
        sClusterName= []
        
        for i in range(iCluster):
            with st.expander(f"Cluster {i+1}"):
                Rmin.append(st.slider(f"R{i} min:", min_value=0, max_value=4, value=2, key=f"Rmin_{i}"))   
                Rmax.append(st.slider(f"R{i} max:", min_value=0, max_value=4, value=2, key=f"Rmax_{i}"))   
        
                Fmin.append(st.slider(f"F{i} min:", min_value=0, max_value=4, value=2, key=f"Fmin_{i}"))   
                Fmax.append(st.slider(f"F{i} max:", min_value=0, max_value=4, value=2, key=f"Fmax_{i}"))   
        
                Mmin.append(st.slider(f"M{i} min:", min_value=0, max_value=4, value=2, key=f"Mmin_{i}"))   
                Mmax.append(st.slider(f"M{i} max:", min_value=0, max_value=4, value=2, key=f"Mmax_{i}"))  
                
                sClusterName.append(st.text_input(f"Cluster name {i+1}:"))
        
        st.write('-'*20)
        
        # self.data_df[clusterCol]= self.data_df.apply(RFMGrade, axis= 1)
        self.data_df[clusterCol]= self.data_df.apply(lambda r: newRFMGrade(r, Rmin, Rmax, Fmin, Fmax, Mmin, Mmax, sClusterName), axis= 1)
        
        rfm_agg= self.data_df.groupby(clusterCol).agg({rColName: 'mean', fColName: 'mean', mColName: ['mean', 'count']})
        rfm_agg.columns= rfm_agg.columns.droplevel() # xóa bỏ các cấp độ đặt tên cột sau khi group và tính toán theo các hàm agg
        rfm_agg.columns= ['R_mean', 'F_mean', 'M_mean', 'Count'] # đặt lại tên các cột
        rfm_agg['Percent']= round(rfm_agg['Count']/rfm_agg['Count'].sum()*100, 2)
        rfm_agg.reset_index(inplace= True)
        self.DrawScatter(rfm_agg)
        
        st.write('-'*20)
        st.write('RFM default')
        self.data_df[clusterCol]= self.data_df.apply(RFMGrade, axis= 1)
        
        rfm_agg= self.data_df.groupby(clusterCol).agg({rColName: 'mean', fColName: 'mean', mColName: ['mean', 'count']})
        rfm_agg.columns= rfm_agg.columns.droplevel() # xóa bỏ các cấp độ đặt tên cột sau khi group và tính toán theo các hàm agg
        rfm_agg.columns= ['R_mean', 'F_mean', 'M_mean', 'Count'] # đặt lại tên các cột
        rfm_agg['Percent']= round(rfm_agg['Count']/rfm_agg['Count'].sum()*100, 2)
        rfm_agg.reset_index(inplace= True)
        self.DrawScatter(rfm_agg)
        
    def __init__(self):
        # self.paths = ['Project 01 - Sentiment Analysis/prj1_data/', 'Project 02 - Recommendation System/prj2_data/', 'Project 03 - Customer Segmentation/prj3_data/']
        self.paths = ['data/', 'data/', 'data/']
        
        # Dữ liệu mẫu
        self.btnCaptions = ["1. Sentiment snalysis", "2. Recommendation system", "3. Customer segmentation"]
        self.mnuItems = ["Introduction", "Experiment", "Model information"]
        self.infoFiles = ["restaurant_overview.csv", "cleaned_P2Data.csv", "cleaned_P3Data.csv"]
        self.dataFiles = ["cleaned_P1data.csv", "Products_ThoiTrangNam_rating_raw.csv", "cleaned_P3Data.csv"]
        self.modelFiles= [['svc_model.joblib', 'nb_model.joblib', 'rf_model.joblib', 'lr_model.joblib', 'gb_model.joblib'], 
                          ['tfidf_gs.model', 'Cosine.pkl', 'svd_model.pkl'], 
                          []]
        self.modelNames= [['SVC model', 'Naive Bayes model', 'Random Forest model', 'Logistic Regression model', 'Gradient Booting model'], 
                          ['GenSim', 'Cosine similarity', 'Surprise'], 
                          ['RFM', 'K-means', 'Hierarchical clustering']]
        
        # Khởi tạo biến lưu trữ chỉ số được chọn
        self.iProject = 0
        self.iMenuItem = 0

        self.wordcloud= None
        self.stopwords_lst= None
        self.intro= [
            """
**Sentiment Analysis - Understanding User Opinions**
---
Sentiment analysis, also known as opinion mining, is a natural language processing technique used to determine the sentiment or emotional tone behind a piece of text. The main goal of sentiment analysis is to classify the text as positive, negative, or neutral based on the opinions expressed within it.

The process of sentiment analysis involves several steps:
1. **Text Preprocessing:** This step involves removing noise from the text, such as special characters, stopwords, and converting text to lowercase.
  
2. **Tokenization:** The text is divided into individual words or tokens to analyze each word's sentiment.

3. **Part-of-Speech Tagging:** Each word is assigned a part-of-speech tag to identify its grammatical category.

4. **Sentiment Scoring:** Words in the text are assigned sentiment scores based on their emotional connotation. These scores are aggregated to determine the overall sentiment of the text.

5. **Classification:** Finally, the sentiment of the text is classified as positive, negative, or neutral using machine learning algorithms such as Support Vector Machines (SVM), Naive Bayes, or Logistic Regression.

Sentiment analysis has various applications, including social media monitoring, brand reputation management, market research, and customer feedback analysis.
""", 
            """
**Recommendation System - Personalized Content Discovery**
---
Recommendation system is an information technology designed to assist users in finding and exploring products, services, or content they may be interested in. The goal of recommendation systems is to provide users with personalized suggestions based on their previous behavior or data from similar user groups.

There are two main types of recommendation systems:
1. **Content-based Recommendation System:** It relies on item descriptions and user preferences to suggest similar items. For example, if a user enjoys watching action movies, the system may recommend other action movies.
  
2. **Collaborative Filtering Recommendation System:** It relies on user behavior and ratings to identify interactions between users and items. The system will suggest items highly rated by users with similar preferences.

Recommendation systems have become an integral part of our daily lives, improving the online shopping experience, discovering digital media content, and providing personalized recommendations in music, video, and e-book applications.
""", 
            """
**Customer Segmentation - Understanding Customer Groups**
---
Customer segmentation is the process of dividing customers into smaller groups based on common characteristics such as purchasing behavior, age, gender, etc. The goal of customer segmentation is to gain a better understanding of the needs, preferences, and characteristics of each customer group in order to develop effective business strategies.

Methods of customer segmentation include:
1. **Demographic Segmentation:** Classifying customers based on factors such as age, gender, income, etc.
  
2. **Behavioral Segmentation:** Segmenting customers based on purchasing behavior, interaction with products or services, etc.

3. **Value-based Segmentation:** Segmenting customers based on the value they bring to the business, including purchase frequency, average spending, etc.

Customer segmentation helps businesses gain insights into their consumers, enabling them to develop marketing strategies, services, and products tailored to each customer segment.
"""
        ]
        self.modelIntro= [
            [
                "Support Vector Classifier (SVC) is a supervised learning algorithm used for classification tasks, including sentiment analysis. SVC aims to find the hyperplane that best separates different classes in the feature space. It works by mapping data to a high-dimensional space and finding the hyperplane that optimally separates the classes.", 
                "Naive Bayes is a probabilistic classifier based on Bayes' theorem with an assumption of independence between features. Despite its simplicity, Naive Bayes often performs well in sentiment analysis tasks, especially when dealing with large datasets.",
                "Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mode of the classes for classification tasks. It's widely used in sentiment analysis due to its ability to handle large datasets, feature importance estimation, and resistance to overfitting.",
                "Logistic Regression is a linear model used for binary classification tasks, which can also be applied to sentiment analysis where the goal is to predict sentiment as positive or negative. It estimates probabilities using the logistic function and makes predictions based on these probabilities.",
                "Gradient Boosting is an ensemble learning technique that builds a strong model by sequentially adding weak learners (typically decision trees). It aims to minimize the loss function by optimizing the gradient direction. Gradient Boosting models, like XGBoost and LightGBM, are effective for sentiment analysis tasks, often providing high accuracy and robustness.",
            ], 
            [
                "GenSim is an open-source library for unsupervised topic modeling and natural language processing, using modern statistical machine learning. It's designed to handle large text collections, using efficient algorithms to discover semantic structure in text data.",
                "Cosine similarity is a metric used to measure the similarity between two vectors by computing the cosine of the angle between them. It's commonly used in information retrieval, document similarity analysis, and recommendation systems. In the context of natural language processing, cosine similarity is often applied to vectorized representations of text documents, such as TF-IDF vectors or word embeddings.", 
                "Surprise is a Python library designed for building recommendation systems. It provides various algorithms for collaborative filtering, including Singular Value Decomposition (SVD), k-Nearest Neighbors (k-NN), and Non-negative Matrix Factorization (NMF)"
            ], 
            [
                "RFM (Recency, Frequency, Monetary) analysis is a marketing technique used to analyze customer behavior by categorizing them into segments based on their transaction history. Recency represents the last purchase date, Frequency is the number of purchases made by the customer, and Monetary represents the total amount spent.",
                "K-means clustering is a popular unsupervised learning algorithm used to partition data into K clusters based on similarity. It aims to minimize the within-cluster sum of squares and assign each data point to the nearest cluster centroid.",
                "Hierarchical clustering is another method of clustering data into groups. It creates a tree of clusters where each node represents a cluster composed of the merged clusters of its children. The linkage criteria (e.g., single, complete, average) determine how the distance between clusters is calculated."
            ]
        ]
        self.KMeanElbowImg= None
        # Thiết kế giao diện
        st.title("Course projects")
        
    def DrawWordCloud(self, restID, labelText= ""):
        wc_stopword= ['quán', 'món']
    
        draw_df= self.data_df[self.data_df['IDRestaurant'] == restID]
        # st.write(draw_df)
        if labelText != "":
            draw_df= draw_df[draw_df[labelCol] == labelText]
        if len(draw_df) > 0:
            theLine= self.info_df[self.info_df['ID'] == restID]
            st.write(theLine)
            # Generate a word cloud image
            wordcloud = WordCloud(width = 800, height = 800,
                            background_color ='white',
                            stopwords = wc_stopword,
                            max_words= 50,
                            min_font_size = 10).generate(' '.join(draw_df[input1Col]))
    
            # Plot the WordCloud image
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
    
            # plt.show()
            # Lưu hình ảnh vào biến self.wordcloud
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            self.wordcloud = buffer.read()
            plt.close()
        else:
            st.write('No record is valid.')
            
    def DrawElbow(self, kmean_df):
        sse= {}
        for k in range(1, 20): # tính toán cho mỗi giá trị k
            kmeans= KMeans(n_clusters= k, random_state= 42)
            kmeans.fit(kmean_df)
            sse[k]= kmeans.inertia_
        if len(sse) > 0:
            # vẽ đồ thị elbow cho các giá trị cua k
            plt.title('The elbow method')
            plt.xlabel('k')
            plt.ylabel('SSE')
            plt.plot(list(sse.keys()), list(sse.values()), marker= 'o')            
            plt.tight_layout(pad = 0)
        
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            self.KMeanElbowImg = buffer.read()
            plt.close()
        else:
            st.write('No record is valid.')

    def DrawScatter(self, kmean_agg):
        fig= px.scatter(kmean_agg, x= 'R_mean', y= 'M_mean', size= 'F_mean', color= clusterCol, hover_name= clusterCol, size_max= 100)
        st.plotly_chart(fig)

    def run(self):
        # Sử dụng st.columns để tạo ra 3 cột để chứa 3 nút nhấn
        # col1, col2, col3 = st.columns(3)
        
        # btnCount= len(self.btnCaptions)
        # self.btns= st.columns(btnCount)
        # for i in range(btnCount):
        #     if self.btns[i].button(self.btnCaptions[i]):
        #         self.iProject= i
        prj = st.radio("Select a project", self.btnCaptions, format_func=lambda x: x)

        # Convert selected value to index
        if prj is not None:
            self.iProject = self.btnCaptions.index(prj)
        # self.iProject= 2

        # Thêm menu
        mnuItem = st.sidebar.selectbox(f"{self.btnCaptions[self.iProject]} content:", self.mnuItems)
        if mnuItem != '':
            self.iMenuItem= self.mnuItems.index(mnuItem)
      
        if (self.iMenuItem == 0): # intro
            st.write(self.intro[self.iProject])
        # ------------------------------------------------------------------------------------------------------------
        elif (self.iMenuItem == 1): # experiment
            if (self.iProject == 0): # sentiment analysis
                self.prj1()
            # ------------------------------------------------------------------------------------------------------------
            elif (self.iProject == 1): # recommendation system
                self.prj2()
            # # ------------------------------------------------------------------------------------------------------------
            elif (self.iProject == 2): # customer segmentation
                self.prj3()
            # ------------------------------------------------------------------------------------------------------------
        elif (self.iMenuItem == 2): # model information
            st.subheader("Model Information")
            model_submenu = st.sidebar.selectbox(f"Model for [{self.btnCaptions[self.iProject]}]", self.modelNames[self.iProject])
            for modelName in self.modelNames[self.iProject]:
                if model_submenu == modelName:
                    st.write(f"Description of {modelName}")
                    idx = self.modelNames[self.iProject].index(model_submenu)
                    st.write(self.modelIntro[self.iProject][idx])
            pass

if __name__ == "__main__":
    app = DoAnMonHoc()
    app.run()
