import numpy as np
import pandas as pd
import unicodedata,re,math,random
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#加载数据集
train_df = pd.read_csv('train.tsv', sep = '\t')
test_df = pd.read_csv('test.tsv',sep='\t')

#show data information
print(train_df.info())

#描述性统计分析
print(train_df.describe())

#可视化，查看Sentiment的得分分布
uniqueLabel=set(train_df['Sentiment'])
x=[]
y=[]
for i in uniqueLabel:
    x.append(i)
    y.append(train_df['Sentiment'][train_df['Sentiment']==i].size)
plt.figure(111)
plt.bar(x,y)
plt.xlabel('type of review  ')
plt.ylabel('count')
plt.title('Movie Review')
plt.show()

#数据预处理（Normalize data)

# 第一步，定义对数据规范化的函数
def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words
    
#数据全部小写化
def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

#去除标点符号
def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words
    
#去掉数据集里面的数字
def remove_numbers(words):
    new_words = []
    for word in words:
        new_word = re.sub("\d+", "", word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

#去除停用词
def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english') and len(word)>1:
            new_words.append(word)
    return new_words

#取词干
def stem_words(words):
    stemmer = nltk.LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

#词性还原
def lemmatize_verbs(words):
    lemmatizer = nltk.WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_numbers(words)
    words = remove_stopwords(words)
#     words = stem_words(words)
#     words = lemmatize(words)
    return words

#切分Phrase
#第二步，tokenize Phrase
textList = train_df['Phrase'].apply(nltk.word_tokenize)
print(textList.head())

# 第三步 - 应用之前定义的函数
textList = textList.apply(normalize) 
print(textList.head())

#词向量化
#统计所有Phrase中出现的unique单词，并统计词频，并按从多到少排序
word_counts = {}
for row in textList:
    for word in row:
        if word in word_counts:
            word_counts[word] += 1
        else:word_counts[word]=1
sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1],reverse=True))
print(sorted_word_counts)

#选取词频>400的单词作为词汇表
corpus = []
for word in sorted_word_counts:
    if sorted_word_counts[word]>400:
        corpus.append(word)
print(corpus)

#词向量化
def wordvec(corpus,textList):
    returnVec = []
    for i in  range(len(textList)):                
        returni = [0]*len(corpus)
        for word in textList.iloc[i]:
            if word in corpus:
                returni[corpus.index(word)] = 1
        returnVec.append(returni)
    return returnVec
wordvec = wordvec(corpus,textList)
wordvec = np.array(wordvec)
print(wordvec)

#Softmax分类算法
def Hypothesis(theta,dataset):    
    score=np.dot(theta,dataset.T)    
    a=np.max(score,axis=0)    
    exp_score=np.exp(score-a)    
    sum_score=np.sum(exp_score,axis=0)    
    relative_probability=exp_score/sum_score    
    return relative_probability 

#计算损失函数#theta为参数矩阵k*(n+1)
def Cost_function(theta,dataset,labels,lamda):    
    m,n=dataset.shape    
    new_code=One_hot_encode(labels)    
    log_probability = np.log(Hypothesis(theta,dataset))    
    cost = -1/m * np.sum(np.multiply(log_probability,new_code)) + lamda * np.sum(theta)/2   
    return cost

#对标签进行独热编码#new_code为 k*m  k为标签数 m为样本数
def One_hot_encode(labels):    
    classes = max(labels) + 1 ##类别数为最大数加1
    one_hot_label = np.zeros((len(labels),classes))##生成全0矩阵
    one_hot_label[np.arange(0,len(labels)),labels] = 1##相应标签位置置1
    return one_hot_label.T
    
#进行梯度检验
def Gradient_checking(gradient,theta,EPSILON,eps,dataset,labels,lamda):    
    theta_vector= theta.ravel()  #将参数矩阵向量化    
    num=len(theta_vector)    
    vector=np.zeros(num)    
    for i in range(num):        
        vector[i]=1        
        theta_plus= theta_vector + EPSILON * vector  #将已求得参数进行微调求近似梯度        
        theta_minus = theta_vector - EPSILON * vector        
        approxiamte_gradient=(Cost_function(theta_plus.reshape(theta.shape),dataset,labels,lamda)-                            
                              Cost_function(theta_minus.reshape(theta.shape),dataset,labels,lamda))/(2*EPSILON)        
        vector[i]=0        
        a = abs(approxiamte_gradient-gradient[i])        
        if a > eps:            
            return False    
        if np.linalg.norm(approxiamte_gradient-gradient,ord=2)/(np.linalg.norm(approxiamte_gradient,ord=2))> eps:        
            return False    
        return True

#使用Batch Gradient Descent优化损失函数#迭代终止条件：  1：达到最大迭代次数   2：前后两次梯度变化小于一个极小值   3：迭代前后损失函数值变化极小
#dataset为原始数据集：m*n     labels:标签   lamda：正则项系数   learning_rate：学习率   max_iter：最大迭代次数
#eps1：损失函数变化量的阈值  eps2：梯度变化量阈值
def SoftmaxRegression(dataset,labels,lamda,learning_rate,max_iter,eps1,eps2,EPS):    
    loss_record=[]    
    m,n = dataset.shape    
    k = len(np.unique(labels))    
    new_code = One_hot_encode(labels)    
    iter = 0    
    new_cost = 0    
    cost = 0    
    dataset=np.column_stack((dataset,np.ones(m)))    
    theta = np.random.random((k,n+1))    
    gradient = np.zeros(n)    
    while iter < max_iter:        
        new_theta = theta.copy()        
        temp = new_code - Hypothesis(new_theta,dataset)        
        for j in range(k):            
            sum = np.zeros(n+1)            
            for i in range(m):                
                a=dataset[i,:]                
                sum += a * temp[j,i]            
                j_gradient=-1/m * sum + lamda * new_theta[j,:] #计算属于第j类相对概率的梯度向量            
                new_theta[j,:] = new_theta[j,:] - learning_rate * j_gradient        
                iter += 1        
#                 print("第"+str(iter)+"轮迭代的参数：")        
                #print(new_theta)        
                new_cost = Cost_function(new_theta,dataset,labels,lamda)        
                loss_record.append(new_cost)        
                #print(new_theta)        
#                 print('损失函数变化量；' ,loss_record)       
                if abs(new_cost-cost) < eps1:            
                    break        
                    theta = new_theta    
                    return theta,loss_record

def Classification(theta,dataset):    
    X=dataset.copy()    
    X=np.column_stack((X,np.ones(X.shape[0])))    
    relative_probability=Hypothesis(theta,X)    
    return np.argmax(relative_probability,axis=0)

#训练集化为训练集和验证集
x_train,x_test,y_train,y_test= train_test_split(wordvec,train_df['Sentiment'],test_size=0.3,random_state=3)
print(x_train.shape)
print(y_train.shape)

#测试，查看准确率
theta,loss_record=SoftmaxRegression(x_train,y_train,lamda=0.1,learning_rate=1e-4,max_iter=1000,eps1=1e-6,eps2=1e-4,EPS=1e-6)
predict=Classification(theta,x_train)
acc = (predict==y_train).mean()  #训练集上精度
print(acc)
acc


    

