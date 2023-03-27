import chardet
from time import sleep
from selenium import webdriver
import sys
import io
import time
import nltk
from snownlp import sentiment
from snownlp.sentiment import Sentiment
from selenium import webdriver
import time
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from snownlp import SnowNLP
import jieba.analyse
from collections import Counter
import jieba
from nltk.sentiment import SentimentIntensityAnalyzer  # 英文用这个
import Levenshtein
import matplotlib.font_manager as fm
from pydub import AudioSegment
from pydub.playback import play
import gensim
from gensim import corpora
from pprint import pprint
from gensim.corpora import Dictionary


def tiktokreptile(urls, n):
    driver = webdriver.Chrome()
    driver.get('https://www.douyin.com/video/7193986868055264567')
    sleep(2)
    # delete = driver.find_element_by_xpath('/html/body/div[3]/div/div/div/div[2]')
    # delete.click()#2022.3.7.19：59没有弹出需要取消的窗口
    # print(a.text)
    sleep(2.5)
    for i in urls:
        driver.get(i)
        sleep(1.5)
        num_of_scrolls = n
        for i in range(num_of_scrolls):
            # 获取页面当前的高度
            last_height = driver.execute_script(
                "return document.body.scrollHeight")
            # 使用 JavaScript 将页面滑动到底部
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            # 等待页面加载完成
            time.sleep(2.5)
            # 获取页面滑动后的高度
            while i == 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90:
                sleep(3.8)
            new_height = driver.execute_script(
                "return document.body.scrollHeight")
            # 如果滑动到页面底部，则退出循环
            if new_height == last_height:
                break
                # 更新页面高度
            last_height = new_height
        time.sleep(2)
        a = driver.find_element_by_xpath(
            '/html/body/div[2]/div[1]/div[2]/div[2]/div/div[1]/div[5]/div/div/div[4]')
        print(a.text)
        sleep(1.1)


def tiktoktextfiltration():
    with open('count.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(len(text))
    text1 = ""
    pattern0 = r'展开\d+条回复\n'
    text = re.sub(pattern0, '', text)
    pattern1 = r'加载中\n'
    text = re.sub(pattern1, '', text)
    pattern2 = r'作者赞过\n'
    text = re.sub(pattern2, '', text)
    # for line in text.split("\n"):
    #     if "20021210" not in line:  # 将 "o" 替换为你想要删除的字符
    #         text1 += line + '\n'
    # for line in text.split("\n"):
    #     if "20021209" not in line:  # 将 "o" 替换为你想要删除的字符
    #         text1 += line + '\n'
    # for line in text.split("\n"):
    #     if "20021208" not in line:  # 将 "o" 替换为你想要删除的字符
    #         text1 += line + '\n'
    # text=text1
    # print(text)
    pattern = r'\n(.+)\n(.+)\n(.+)\n回复\n(.+)\n'
    # 使用 re.sub() 函数将匹配到的字符串1替换为空字符串
    clean_text = re.sub(pattern, '\n\n\n回复\n', text)
    pattern3 = r'作者\n'
    clean_text = re.sub(pattern3, '', clean_text)
    pattern4 = r'回复\n'
    clean_text = re.sub(pattern4, '', clean_text)
    pattern5 = r'分享\n'
    clean_text = re.sub(pattern5, '', clean_text)
    pattern6 = r'\d+天前(.+)\n'
    clean_text = re.sub(pattern6, '', clean_text)
    # print(clean_text)
    words = jieba.cut(clean_text)
    with open(r'C:\Users\wr\Downloads\cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read().splitlines()
    m = ['']
    stop_words.extend(m)
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    text = '\n'.join(non_empty_lines)
    print(text)
    return text


def emotionanalysis(text):
    keywords = jieba.analyse.extract_tags(text, topK=20, withWeight=True)
    # keywords加权输出
    print(keywords)
    # 情感分析score
    scores = []
    for sentence in text.split('\n'):
        s = SnowNLP(sentence)
        scores.append(s.sentiments)
        print(sentence, s.sentiments)
    # 平均score
    avg_score = sum(scores) / len(scores)
    print('Average score:', avg_score)


def wordcloud1(text):
    words = text
    with open(r'C:\Users\wr\Downloads\cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read().splitlines()
    m = ['b站', '概念', '概', '念', '念版', '确实', 'b 站', '软件', '听歌', 'b', '站', '概念版', '下载', '不用', '我用',
         '概 念 版', '评论', '', '赞同', '喜欢', '回答', '收藏', '说', '听', '没有', '一个', 'sk', '分享', '想', '更多',
         '会', "歌", '更', '现在', '更多', '可能', '真的', '收起', '条', '数', '推荐', '问题', '觉得', '关注', '不能',
         '很多', '已经', '应该', '感觉', '选择', '年', '知道', '支持', '区', '做', '直接', '朋友',
         '国内', '编辑', '发现', '警员', '东西', '找到', '找', '', '', 'Apple', 'Music', '添加', '发布', '几个',
         '是因为', '中', '只']
    # stop_words.append(m)
    stop_words.extend(m)
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    wc = WordCloud(background_color='white', width=800,
                   height=600, font_path='msyh.ttc')
    wc.generate(text)
    # 显示词云
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    urls = ['https://www.douyin.com/video/7044067893213957412',
            'https://www.douyin.com/video/6916808640670272771',
            'https://www.douyin.com/video/7016321893363993860', 'https://www.douyin.com/video/6959858691352841510',
            'https://www.douyin.com/video/6922665352836943119', 'https://www.douyin.com/video/6946111246789807374',]
    u = ['https://www.douyin.com/video/7172441725456141606']
    a = ['https://www.douyin.com/video/6922665352836943119']
    c = ['https://www.douyin.com/video/7179551433220508965']
    less = ['https://www.douyin.com/video/7081548430325058850', 'https://www.douyin.com/video/7162022643296685349',
            'https://www.douyin.com/video/7085179710958865664', 'https://www.douyin.com/video/7071205886227713316',
            'https://www.douyin.com/video/6988859513361255687']
    d = ['https://www.douyin.com/video/7155389438221241607', 'https://www.douyin.com/video/7113034748003355940',
         'https://www.douyin.com/video/7014434496082693406', 'https://www.douyin.com/video/7054763734711029000',
         'https://www.douyin.com/video/7091113234744298783', 'https://www.douyin.com/video/7010677434718653726',
         'https://www.douyin.com/video/7078495609103207713', 'https://www.douyin.com/video/7091535125422427406',
         'https://www.douyin.com/video/7117566964486180126', 'https://www.douyin.com/video/7145602993906273576',
         'https://www.douyin.com/video/6973987181018860832', 'https://www.douyin.com/video/7033711551680974092',
         'https://www.douyin.com/video/7042573797555260680', 'https://www.douyin.com/video/6989181024613748006', 'https://www.douyin.com/video/7196674047940283686', 'https://www.douyin.com/video/7055478797923126543',
         'https://www.douyin.com/video/7103466141808397581', 'https://www.douyin.com/video/7139462439283772676',
         'https://www.douyin.com/video/7190050806270283060', 'https://www.douyin.com/video/7185134332653194554',
         'https://www.douyin.com/video/6980524135256575262']
    # emotionanalysis(tiktoktextfiltration())
    # tiktoktextfiltration()

    # tiktokreptile(a,88)
    # tiktokreptile(a,50)
    # tiktokreptile(u+c,38)
    # 重新爬取

    # tiktokreptile(less,15)
    # tiktokreptile(d,48)

    # tiktokreptile(c,88)
    # a0=tiktoktextfiltration()
    # emotionanalysis(a0)
    # wordcloud1(a0)
# # 载入声音文件
#     sound = AudioSegment.from_wav("recorded_audio.wav")
# # 播放声音
#     play(sound）
    # iktoktextfiltration()
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    with open(r'bilicomments.txt', 'r', encoding='utf-8') as f:
        words1 = f.read().split()
    with open(r'tiktokcomments.txt', 'r', encoding='utf-8') as f:
        words = f.read().split()  # 依据空格来读
    words = words+words1
    word_counts = Counter(words)
    # print(word_counts)

    # x=[]
    # y=[]
    # for word, count in word_counts.most_common(30):
    #     x.append(count)
    #     y.append(word)
    #     #print(word, count)
    # sorted_index = sorted(range(len(x)), key=lambda k: x[k], reverse=False)
    # x= [x[i] for i in sorted_index]
    # y= [y[i] for i in sorted_index]
    # plt.barh(y, x)
    # # 设置标题和坐标轴标签
    # plt.title('文本分析—词频统计排名')
    # plt.xlabel('词频')
    # plt.ylabel('词语')
    # # 显示图形#柱状图
    # plt.show()

    with open(r'tiktokcomments.txt', 'r', encoding='utf-8') as f:
        tokens = f.read().splitlines()
    with open(r'bilicomments.txt', 'r', encoding='utf-8') as f:
        tokens1 = f.read().splitlines()
    with open(r'zhihu2.txt', 'r', encoding='utf-8') as f:
        tokens2 = f.read().splitlines()
    tokens = tokens+tokens1+tokens2
    tokens = [x.lower() for x in tokens]

    nltk.download('vader_lexicon')

    # 0.02613 0.13848 0.83539
    wyykeywords = ['网易云音乐', '网易', '云音乐', '云村']
    applekeywords = ['apple music', 'am', '苹果音乐']
    qqkeywords = ['qq音乐', 'q音', 'qq']
    kugoukeywords = ['酷狗', '酷狗音乐']
    kuwokeywords = ['酷我', '酷我音乐']
    spotifykeywords = ['spotify']
    qishuikeywords = ['汽水', '汽水音乐']
    keywords0 = ['版权', '曲库', '歌多', '歌少']
    allskey = wyykeywords+qqkeywords+applekeywords + \
        kugoukeywords+kuwokeywords+spotifykeywords+qishuikeywords
    keywords1 = ['广告']
    keywords2 = ['评论', '评论区', '热评', '氛围']
    keywords3 = ['推送', '日推', '推荐', '个性推荐']
    keywords4 = ['音质']
    keywords5 = ['周杰伦']
    keywords6 = ['免费', '不要钱']
    keywords7 = ['界面', '设计']
    keywords8 = ['会员', 'vip', '付费', '要钱']
    keywords9 = ['歌单']
    keywords10 = ['社交', '圈子']
    keyword = ['广告', '版权', '曲库', '推送', '日推', '推荐', '评论', '音质', '周杰伦', '界面', '设计', '会员',
               'vip', '免费', '要钱', '不要钱', '歌单', '付费', '评论区', '热评', '氛围', '社交', '圈子', '商城', '视频', '个性推荐']
    kugougainian = ['酷狗概念版', '概念版', '推荐', '概念']
    text = '我喜欢用网易云  个音乐听歌'

    def widelyfind(keywords, tokens):
        matched_tokens = []
        for token in tokens:
            for keyword in keywords:
                words = jieba.lcut(keyword)
                match = True
                for word in words:
                    if word not in token:
                        distance = Levenshtein.distance(word, token)
                        if distance > len(word) // 2:
                            match = False
                            break
                if match:
                    matched_tokens.append(token)
        print(len(matched_tokens))
        # print(matched_tokens)
        return matched_tokens
    # a=widelyfind(wyykeywords,tokens)
    # print(a)
    # dele=widelyfind()
    # b=widelyfind(keywords,a)

    allkeywords = keyword

    texts = widelyfind(allskey, tokens)
    results = []
    po = []
    ne = []
    for text in texts:
        s = SnowNLP(text)
        polarity = s.sentiments
        if polarity >= 0.6:
            po.append({'text': text, 'polarity': 'pos'})
        elif polarity < 0.6:
            ne.append({'text': text, 'polarity': 'neg'})
        # print(results)
    with open('possentiment.txt', 'w', encoding='utf-8') as f:
        for d in po:
            line = d['text'] + '\t' + str(d['polarity']) + '\n'
            f.write(line)
    with open('negsentiment.txt', 'w', encoding='utf-8') as f:
        for d in ne:
            line = d['text'] + '\t' + str(d['polarity']) + '\n'
            f.write(line)

    sentiment.train('possentiment.txt', 'negsentiment.txt')
    sentiment.save('sentiment.marshal')

    # 对测试文本进行情感分析
    sentiment = Sentiment()
    sentiment.load('sentiment.marshal')
    # 对文本进行情感分析
    text = '网易云音乐没版权'
    result = sentiment.classify(text)
    print(result)

    # sentiment.train('sentiment.txt')
    # # 保存训练结果
    # sentiment.save('sentiment.marshal')
    # # 对测试文本进行情感分析
    # text = '这家餐厅的服务很糟糕，菜品也很一般。'
    # s = Sentiment()
    # s.sentiment(text)

    # keywords0=['版权','曲库','歌多','歌少']
    # allskey=wyykeywords+qqkeywords+applekeywords+kugoukeywords+kuwokeywords+spotifykeywords+qishuikeywords
    # keywords1=['广告']
    # keywords2=['评论','评论区','热评','氛围']
    # keywords3=['推送','日推','推荐','个性推荐']
    # keywords4=['音质']
    # keywords5=['周杰伦']
    # keywords6=['免费','不要钱']
    # keywords8=['会员','vip','付费','要钱']
    # keywords7=['界面','设计']
    # keywords9=['歌单']
    # keywords10=['社交','圈子']
    # q2=widelyfind(kugoukeywords,tokens)
    # q1=widelyfind(kugougainian,tokens)
    # # result = [x - y for x, y in zip(q2, q1)]
    # result = [x for x in q1 if x not in q2]
    q = widelyfind(kugoukeywords, tokens)
    c = [widelyfind(keywords0, q), widelyfind(keywords1, q), widelyfind(keywords2, q), widelyfind(keywords3, q),
         widelyfind(keywords4, q), widelyfind(keywords5, q), widelyfind(
             keywords6, q), widelyfind(keywords8, q),
         widelyfind(keywords7, q), widelyfind(keywords9, q), widelyfind(keywords10, q)]
    data = []
    for j in c:
        sentence = j
        j = 0
        m = 0
        p = 0
        n = 0
        fulls = 0
        for i in sentence:
            s = SnowNLP(i)
            sentiment = s.sentiments
            fulls = fulls + sentiment
            j = j + 1
            if sentiment > 0.6:
                p += 1
            elif sentiment < 0.4:
                n += 1
            else:
                m += 1
        # print(sentiment)
        # averagescore = fulls / j
        # print(averagescore)
        all = p + n + m
        positive_percent = p / all
        negative_percent = n / all
        # 绘制饼图
        neutral_percent = m / all
        labels = ['正面', '负面', '中立']
        positive_percent = "{:.5f}".format(positive_percent)
        negative_percent = "{:.5f}".format(negative_percent)
        neutral_percent = "{:.5f}".format(neutral_percent)
        neutral_percent = float(neutral_percent)
        positive_percent = float(positive_percent)
        negative_percent = float(negative_percent)
        a = (positive_percent, negative_percent, neutral_percent)
        data.append(a)
    print(data)
    # print(positive_percent)
    # print(negative_percent)
    # print(neutral_percent)
    # sizes = [positive_percent, negative_percent, neutral_percent]

    colors = [(255, 192, 203), (238, 216, 174), (32, 178, 170)]
    # 将颜色列表中的每个元组转换为0到1之间的浮点数值
    colors = [(r / 255.0, g / 255.0, b / 255.0) for (r, g, b) in colors]
    #

    import matplotlib.pyplot as plt

    colors = [(238, 216, 174), (32, 178, 170), (255, 192, 203)]
    # 将颜色列表中的每个元组转换为0到1之间的浮点数值
    colors = [(r / 255.0, g / 255.0, b / 255.0) for (r, g, b) in colors]

    # data = [(0.39184,0.51813,0.09003), (0.44988,0.46659,0.08353), (0.71242,0.20588,0.08170),
    #         (0.71242,0.20588,0.08170), (0.54361,0.32975,0.12664), (0.39340,0.44842,0.15818),
    #         (0.67263,0.24022,0.08715), (0.40685,0.47336,0.11980), (0.71910,0.21208,0.06882),
    #         (0.59463,0.31746,0.08791), (0.56735,0.32245,0.11020)]
    labels = ['版权', '广告', '氛围', '歌曲推荐', '音质',
              '周杰伦', '免费', '付费', '界面', '歌单', '社交']
    plt.figure(figsize=(10, 5))
    for i in range(len(data)):
        plt.bar(i, data[i][0], width=0.25, color=colors[2])
        plt.bar(i, data[i][1], bottom=data[i][0], width=0.25, color=colors[0])
        plt.bar(i, data[i][2], bottom=data[i][0] +
                data[i][1], width=0.25, color=colors[1])
    plt.xticks(range(len(labels)), labels)

    plt.xlabel('关键词')
    plt.ylabel('情感比例')
    plt.text(len(data), max([item for sublist in data for item in sublist]) / 3, '积极', ha='center', color=colors[2],
             fontsize=12)
    plt.text(len(data), max([item for sublist in data for item in sublist]) / 3 * 2, '消极', ha='center',
             color=colors[0], fontsize=12)
    plt.text(len(data), max([item for sublist in data for item in sublist]), '中性', ha='center', color=colors[1],
             fontsize=12)
    plt.title('用户对酷狗音乐主要关注点的情感分析')
    plt.show()

    # # 颜色设置
    # # 情感分析饼状图：
    # plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f')
    # plt.axis('equal')
    # plt.title('以免费为关键词的情感分析')
    # plt.show()

    # #matched_tokens=widelyfind(keywords, tokens)
    #
    # my_string = ' '.join(tokens)
    # words=my_string.split()
    # # stop_words = ['网易','・','qq', '我用','音乐', '酷狗','云','听','听歌','喜欢','真的','酷我','说','b','站','腾讯','狗','版','想','概念','酷','年','里','区',
    # #               '咪咕','苹果','q','用酷','apple music', 'apple','・', 'music', 'spotify', '虾米', '天天']
    # # words = [word for word in words if word not in stop_words]
    # # words = ' '.join(words)
    # # words = words.split()
    # word_counts = Counter(words)
    # x=[]
    # y=[]
    # for word, count in word_counts.most_common(30):#人工去词
    #     x.append(count)
    #     y.append(word)
    #     print(word, count)
    # sorted_index = sorted(range(len(x)), key=lambda k: x[k], reverse=False)
    # x= [x[i] for i in sorted_index]
    # y= [y[i] for i in sorted_index]
    # plt.barh(y, x)
    # # 设置标题和坐标轴标签
    # plt.title('文本分析—词频统计排名')
    # plt.xlabel('词频')
    # plt.ylabel('词语')
    # # 显示图形#柱状图
    # plt.show()
