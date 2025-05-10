以下是课件从第14页开始的逐句中英对照翻译：

---

**Page 14**  
**# Token Vectorization**  
**词向量化**  
- **Extract the word vector (or word embedding) of every token**  
  提取每个词元的词向量（或词嵌入）。  
- **A word vector is a multi-dimensional mathematical representation of a word**  
  词向量是单词的多维数学表示。  
- **Semantically similar words have vectors located close to each other (cosine distance or similarity is usually adopted)**  
  语义相似的词其向量在空间中距离相近（通常用余弦距离/相似度衡量）。  
- **Word vectors are computed by training a neural network to predict a word given its surrounding context (such as the words that appear before or after it in a sentence), and then using the weights of the neural network's hidden layers as the word's embedding**  
  词向量通过训练神经网络预测上下文单词生成，其隐藏层权重作为词嵌入。  
- **Usually, it is better to use pretrained word vectors (transfer learning) such as Word2Vec, FastText, Glove, ...**  
  通常建议使用预训练词向量（如Word2Vec、FastText、GloVe等）。  
- **Generally, word vectors are high dimensional (usually \(\mathbb{R}^{100}\) or \(\mathbb{R}^{300}\))**  
  词向量通常是高维的（如100维或300维）。  

---

**Page 15**  
**How NLP pipeline steps work?**  
**NLP流程步骤如何工作？**  
- **Usually, the NLP steps seen before are implemented by using some form of Neural Network**  
  前述NLP步骤通常通过神经网络实现。  
- **We will use them out-of-the-box by exploiting a Python's library called Spacy**  
  我们将直接使用Python库Spacy调用这些功能。  

---

**Page 16**  
**Outline**  
**大纲**  
- **NLP and NLP pipelines**  
  自然语言处理及其流程  
- **The Spacy library**  
  Spacy库  
- **Pretrained LLM**  
  预训练大型语言模型  

---

**Page 17**  
**# Install Spacy and NLTK**  
**安装Spacy和NLTK**  
- **If you have created a Conda environment for our scripts, activate it with the following command:**  
  如果已创建Conda环境，请用以下命令激活：  
  ```bash
  conda activate webintelligence
  ```  
- **Install Spacy and NLTK:**  
  安装Spacy和NLTK：  
  ```bash
  pip install spacy nltk
  ```  
- **Download a prebuilt NLP English pipeline for Spacy**  
  下载Spacy的英文预训练模型：  
  ```bash
  python -m spacy download en_core_web_md
  ```  
- **There are also:**  
  其他可选模型：  
  - **en_core_web_sm (but it misses some processing steps)**  
    轻量版（缺少部分功能）  
  - **en_core_web_lg (but it is quite slow)**  
    完整版（速度较慢）  

---

**Page 18**  
**The pipeline of the Spacy model en_core_web_md**  
**Spacy模型en_core_web_md的流程组成**  
```python
In [1]: import spacy  
In [2]: nlp = spacy.load('en_core_web_md')  
In [3]: nlp.pipe_names  
Out[3]: ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']  
```  
**解释：**  
- **tok2vec**: 词元向量化  
- **tagger**: 词性标注  
- **parser**: 依存句法分析  
- **attribute_ruler**: 属性规则器  
- **lemmatizer**: 词形还原  
- **ner**: 命名实体识别  

---

**Page 19**  
**# Sentence Segmentation with Spacy**  
**使用Spacy进行句子分割**  
```python
# 定义待分析文本
txt = 'Alan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician...'  

# 加载Spacy模型
nlp = spacy.load('en_core_web_md')  

# 处理文本
doc = nlp(txt)  

# 打印所有句子
for i, sent in enumerate(doc.sents):  
    print(f"Sentence #{i+1}:")  
    print(sent)  
```  
**输出示例：**  
```
Sentence #1:  
Alan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician...  

Sentence #2:  
Turing was highly influential in the development...  
```  

---

**Page 20**  
**# Tokenization in Spacy**  
**Spacy中的分词**  
```python
txt = 'Hong Kong is a beautiful city!'  
doc = nlp(txt)  
for token in doc:  
    print(token.text)  
```  
**输出：**  
```
Hong  
Kong  
is  
a  
beautiful  
city  
!  
```  

---

**Page 21**  
**Spacy container objects: Doc, Token, Span**  
**Spacy容器对象：Doc（文档）、Token（词元）、Span（片段）**  

---

**Page 22**  
**# Token basic properties in Spacy**  
**Spacy中词元的基本属性**  
- **is_alpha**: True if the token is a proper word  
  是否为字母组成的单词  
- **is_stop**: True if the token is a stopword (e.g., "the", "is")  
  是否为停用词  
- **shape_**: Orthographic features (e.g., "XXXx" for "Hong")  
  词形特征（字母替换为x/X，数字替换为d）  

**示例代码：**  
```python
for token in doc:  
    print(token.text, token.is_alpha, token.is_stop, token.shape_)  
```  
**输出：**  
```
Hong True False Xxxx  
Kong True False Xxxx  
is True True xx  
a True True x  
beautiful True False xxxx  
city True False xxxx  
! False False !  
```  

---

**Page 23**  
**# Lemma vs Stem of a Token**  
**词元与词干的对比**  
```python
from nltk.stem import PorterStemmer  
stemmer = PorterStemmer()  

for token in doc:  
    print(token.text, token.lemma_, stemmer.stem(token.text))  
```  
**输出：**  
```
Hong Hong hong  
Kong Kong kong  
is be is  
a a a  
beautiful beautiful beauti  
city city citi  
! ! !  
```  
**说明：**  
- Spacy提供词形还原（lemma），但需用NLTK进行词干提取（stemming）。  

---

（后续内容包含词性标注、依存分析、命名实体识别等，如需继续翻译请告知。）
以下是课件从第24页开始的逐句翻译：

---

**Page 24**
**POS tag of a Token with Spacy**
**使用Spacy获取词元的词性标注**

In [57]: doc
Out[57]: Hong Kong is a beautiful city!

In [58]: for token in doc:
    ...:    print(token.text, token.pos_)
    ...:

Hong PROPN（专有名词）
Kong PROPN（专有名词）
is AUX（助动词）
a DET（限定词）
beautiful ADJ（形容词）
city NOUN（名词）
! PUNCT（标点符号）

---

**Page 25**
**# Morphological features of a Token with Spacy**
**使用Spacy获取词元的形态特征**

In [4]: doc = nlp('I was reading a paper.')

In [5]: for token in doc:
    ...: print(token.text, token.pos_, token.morph)
    ...:

I PRON（代词） Case=Nom|Number=Sing|Person=1|PronType=Prs（主格|单数|第一人称|人称代词）
was AUX（助动词） Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin（陈述语气|单数|第三人称|过去时|限定动词）
reading VERB（动词） Aspect=Prog|Tense=Pres|VerbForm=Part（进行体|现在时|分词形式）
a DET（限定词） Definite=Ind|PronType=Art（不定冠词）
paper NOUN（名词） Number=Sing（单数）
. PUNCT（标点符号） PunctType=Peri（句号）

---

**Page 26**
**# Dependency Parse Tree with Spacy**
**使用Spacy进行依存句法分析**

- Any single sentence is formed by exactly one parse tree
  每个句子对应一个依存树
  - A node of a tree has only one parent (or zero if it is the root), so Spacy defines two attributes for each token:
    每个节点只有一个父节点（根节点除外），因此Spacy为每个词元定义两个属性：
    - head which points to the parent token,
      head指向父词元
    - dep_ which provides the label of the edge (i.e. the type of dependency)
      dep_表示依存关系类型

**示例代码：**
In [de]: doc
Out[de]: Hong Kong is a beautiful city!

In [di]: for token in doc:
    ...: print(token.text, token.dep_, token.head.text)

Hong compound Kong（复合词修饰）
Kong nsubj is（名词性主语）
is ROOT is（根节点）
a det city（限定词修饰）
beautiful amod city（形容词修饰）
city attr is（属性补足语）
! punct is（标点附加）

---

**Page 27**
**# Noun Chunks with Spacy**
**使用Spacy提取名词短语**

- Noun chunks are "base noun phrases" - flat phrases that have a noun as their head.
  名词短语是以名词为核心的扁平短语结构

**示例代码：**
In [6]: doc = nlp('Hong Kong is a beautiful city')

In [7]: for nc in doc.noun_chunks:
    ...: print(nc)
    ...
Hong Kong（香港）
a beautiful city（一座美丽的城市）

---

**Page 28**
**# Named Entities with Spacy**
**使用Spacy进行命名实体识别**

In [75]: doc
Out[75]: Hong Kong is a special administrative region of China and Bruce Lee was from Hong Kong!!!

In [76]: for ent in doc.ents:
    ...: print(ent.text, ent.label_, ent.start_char, ent.end_char)
    ...:

Hong Kong GPE（地理政治实体） 0 9
China GPE（地理政治实体） 49 54
Bruce Lee PERSON（人名） 59 68
Hong Kong GPE（地理政治实体） 78 87

---

**Page 29**
**# Common transformation of a text**
**文本的常见转换方法**

For further semantic processing of a text, sometimes it is useful to:
对文本进行深层语义处理时，通常需要：
1. remove stop words and non-alphabetical tokens
   移除停用词和非字母词元
2. replace token text with its lemma
   用词元替换原词
3. merge the words of a compound named entity
   合并复合命名实体的单词

**示例代码：**
最终转换结果：
'Hong_Kong special administrative region China Bruce_Lee Hong_Kong'

---

**Page 30**
**Find the most common lemmas in a corpus**
**查找语料库中最常见的词元**

- Corpus is synonym of "set of texts" ... we may also call "dataset"
  语料库即文本集合，也可称为数据集

**示例输出：**
[('Hong', 4),
('Kong', 4),
('city', 3),
('Macau', 3),
('beautiful', 2)...]

---

**Page 31**
**# Spell Checking**
**拼写检查**

- We need another Python library: pip install pyspellchecker
  需安装额外库：pip install pyspellchecker
- It uses Levenshtein Distance algorithm to find permutations within an edit distance of 2
  使用编辑距离≤2的Levenshtein算法查找可能的正确拼写

**示例：**
错误文本：'Tuday is beautiful dya'
修正结果：'today is beautiful day'

---

（后续内容包含词向量应用、预训练模型等，如需继续翻译请告知。）

以下是课件从第24页开始的逐句翻译：

---

**Page 24**
**POS tag of a Token with Spacy**
**使用Spacy获取词元的词性标注**

In [57]: doc
Out[57]: Hong Kong is a beautiful city!

In [58]: for token in doc:
    ...:    print(token.text, token.pos_)
    ...:

Hong PROPN（专有名词）
Kong PROPN（专有名词）
is AUX（助动词）
a DET（限定词）
beautiful ADJ（形容词）
city NOUN（名词）
! PUNCT（标点符号）

---

**Page 25**
**# Morphological features of a Token with Spacy**
**使用Spacy获取词元的形态特征**

In [4]: doc = nlp('I was reading a paper.')

In [5]: for token in doc:
    ...: print(token.text, token.pos_, token.morph)
    ...:

I PRON（代词） Case=Nom|Number=Sing|Person=1|PronType=Prs（主格|单数|第一人称|人称代词）
was AUX（助动词） Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin（陈述语气|单数|第三人称|过去时|限定动词）
reading VERB（动词） Aspect=Prog|Tense=Pres|VerbForm=Part（进行体|现在时|分词形式）
a DET（限定词） Definite=Ind|PronType=Art（不定冠词）
paper NOUN（名词） Number=Sing（单数）
. PUNCT（标点符号） PunctType=Peri（句号）

---

**Page 26**
**# Dependency Parse Tree with Spacy**
**使用Spacy进行依存句法分析**

- Any single sentence is formed by exactly one parse tree
  每个句子对应一个依存树
  - A node of a tree has only one parent (or zero if it is the root), so Spacy defines two attributes for each token:
    每个节点只有一个父节点（根节点除外），因此Spacy为每个词元定义两个属性：
    - head which points to the parent token,
      head指向父词元
    - dep_ which provides the label of the edge (i.e. the type of dependency)
      dep_表示依存关系类型

**示例代码：**
In [de]: doc
Out[de]: Hong Kong is a beautiful city!

In [di]: for token in doc:
    ...: print(token.text, token.dep_, token.head.text)

Hong compound Kong（复合词修饰）
Kong nsubj is（名词性主语）
is ROOT is（根节点）
a det city（限定词修饰）
beautiful amod city（形容词修饰）
city attr is（属性补足语）
! punct is（标点附加）

---

**Page 27**
**# Noun Chunks with Spacy**
**使用Spacy提取名词短语**

- Noun chunks are "base noun phrases" - flat phrases that have a noun as their head.
  名词短语是以名词为核心的扁平短语结构

**示例代码：**
In [6]: doc = nlp('Hong Kong is a beautiful city')

In [7]: for nc in doc.noun_chunks:
    ...: print(nc)
    ...
Hong Kong（香港）
a beautiful city（一座美丽的城市）

---

**Page 28**
**# Named Entities with Spacy**
**使用Spacy进行命名实体识别**

In [75]: doc
Out[75]: Hong Kong is a special administrative region of China and Bruce Lee was from Hong Kong!!!

In [76]: for ent in doc.ents:
    ...: print(ent.text, ent.label_, ent.start_char, ent.end_char)
    ...:

Hong Kong GPE（地理政治实体） 0 9
China GPE（地理政治实体） 49 54
Bruce Lee PERSON（人名） 59 68
Hong Kong GPE（地理政治实体） 78 87

---

**Page 29**
**# Common transformation of a text**
**文本的常见转换方法**

For further semantic processing of a text, sometimes it is useful to:
对文本进行深层语义处理时，通常需要：
1. remove stop words and non-alphabetical tokens
   移除停用词和非字母词元
2. replace token text with its lemma
   用词元替换原词
3. merge the words of a compound named entity
   合并复合命名实体的单词

**示例代码：**
最终转换结果：
'Hong_Kong special administrative region China Bruce_Lee Hong_Kong'

---

**Page 30**
**Find the most common lemmas in a corpus**
**查找语料库中最常见的词元**

- Corpus is synonym of "set of texts" ... we may also call "dataset"
  语料库即文本集合，也可称为数据集

**示例输出：**
[('Hong', 4),
('Kong', 4),
('city', 3),
('Macau', 3),
('beautiful', 2)...]

---

**Page 31**
**# Spell Checking**
**拼写检查**

- We need another Python library: pip install pyspellchecker
  需安装额外库：pip install pyspellchecker
- It uses Levenshtein Distance algorithm to find permutations within an edit distance of 2
  使用编辑距离≤2的Levenshtein算法查找可能的正确拼写

**示例：**
错误文本：'Tuday is beautiful dya'
修正结果：'today is beautiful day'

---

（后续内容包含词向量应用、预训练模型等，如需继续翻译请告知。）
