以下是课件内容的逐句中英对照翻译：

---

**Page 1**  
**COMP7630 – Web Intelligence and its Applications**  
COMP7630 – 网络智能及其应用  
**Natural Language Processing pipelines**  
自然语言处理流程  
**Valentino Santucci (valentino.santucci@unistrapg.it)**  
瓦伦蒂诺·桑图奇（邮箱：valentino.santucci@unistrapg.it）  

---

**Page 2**  
**Outline**  
大纲  
**NLP and NLP pipelines**  
自然语言处理（NLP）及其流程  
**The Spacy library**  
Spacy库  
**Pretrained LLM**  
预训练大型语言模型  

---

**Page 3**  
**# Natural Language Processing (NLP)**  
**自然语言处理（NLP）**  
- **NLP involves using computational techniques to understand, analyze, and generate human language.**  
  NLP涉及使用计算技术来理解、分析和生成人类语言。  
- **Some NLP tasks: text classification, sentiment analysis, topic modeling, machine translation, text generation, text summarization, ...**  
  部分NLP任务：文本分类、情感分析、主题建模、机器翻译、文本生成、文本摘要等。  
- **NLP techniques are used in WI for tasks such as:**  
  NLP技术在网络智能（WI）中的应用包括：  
  - **Extracting structured data from unstructured text found on web pages**  
    从网页的非结构化文本中提取结构化数据  
  - **Identifying named entities and extracting information about them**  
    识别命名实体并提取相关信息  
  - **Analyzing the sentiment and emotion expressed in web content**  
    分析网络内容中表达的情感和情绪  
  - **Understanding the intent behind user queries and search phrases**  
    理解用户查询和搜索短语背后的意图  
  - **Summarizing web pages and articles**  
    对网页和文章进行摘要  

---

**Page 4**  
**# NLP pipeline**  
**NLP流程**  
- **It is possible to identify some basic processing steps which are required by many complex NLP and WI tasks**  
  许多复杂的NLP和WI任务需要一些基本处理步骤。  
- **These basic steps form a NLP pipeline and they can vary depending on the task at hand, but generally a NLP pipeline includes some combination of the following steps:**  
  这些基本步骤构成了NLP流程，具体步骤因任务而异，但通常包括以下部分或全部步骤：  
  - **Tokenization**  
    分词  
  - **Sentence Segmentation**  
    句子分割  
  - **Part-of-Speech Tagging**  
    词性标注  
  - **Lemmatization**  
    词形还原  
  - **Stemming**  
    词干提取  
  - **Morphological Analysis**  
    形态分析  
  - **Dependency Parsing**  
    依存句法分析  
  - **Named Entity Recognition**  
    命名实体识别  
  - **Token Vectorization**  
    词向量化  

---

**Page 5**  
**# Tokenization**  
**分词**  
- **Divide a text into tokens, i.e., words, punctuation marks, etc.**  
  将文本分割为词元（如单词、标点符号等）。  
- **This is done by applying rules specific to each language.**  
  分词需应用针对每种语言的特定规则。  
  - **For example, punctuation at the end of a sentence should be split off – whereas “U.K.” should remain one token.**  
    例如，句末标点应单独分割，而“U.K.”应保留为一个词元。  

**Example**  
示例  
**"Apple is looking at buying U.K. startup for $1 billion"**  
**分词结果：**  
| Apple | is | looking | at | buying | U.K. | startup | for | $ | 1 | billion |  

---

**Page 6**  
**# Sentence Segmentation**  
**句子分割**  
**Segment a text into sentences**  
将文本分割为句子。  

**Example**  
示例  
**原文：**  
"Alan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer. He is widely considered to be the father of theoretical computer science and artificial intelligence."  

**分割结果：**  
**Sentence #1**  
"Alan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist."  

**Sentence #2**  
"Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer."  

**Sentence #3**  
"He is widely considered to be the father of theoretical computer science and artificial intelligence."  


以下是课件从第7页开始的逐句中英对照翻译：

---

**Page 7**  
**# Part-of-speech Tagging**  
**词性标注**  
- **The process of marking up a token in a text as corresponding to a particular part of speech, based on both its definition and its context**  
  根据词的定义和上下文，将文本中的词元标记为特定词性的过程。  

---  
**示例表格：**  
**TEXT（文本）** | **POS（词性）**  
---|---  
Apple  | PROPM（专有名词）  
is    | AUX（助动词）  
looking at | VERB（动词）  
buying  | VERB（动词）  
U.K.   | PROPM（专有名词）  
startup for | PROPM（专有名词）  
$    | NUM（数字）  
1    | NUM（数字）  
billion | NUM（数字）  

---

**Page 8**  
**# Text Normalization**  
**文本规范化**  
- **Text normalization is the process of transforming text into a consistent and standardized format. It involves various techniques to reduce words to their base or root form, enhancing the efficiency of text analysis.**  
  文本规范化是将文本转换为统一标准格式的过程，通过将词语缩减为基础或词根形式，提升文本分析效率。  

- **Two possibilities:**  
  两种方法：  
  - **Lemmatization**  
    词形还原  
  - **Stemming**  
    词干提取  

- **... plus their combination which is sometimes useful:**  
  有时还可结合以下方法：  
  - **analyze the text where every word is replaced with stem(lemma(word))**  
    用“词干(词元(单词))”替换文本中的每个词进行分析。  

---

**Page 9**  
**# Lemmatization**  
**词形还原**  
- **Extract the lemma of a word, i.e. the base form of a word**  
  提取单词的词元（即词典中的基础形式）。  
- **Base form = dictionary form**  
  基础形式 = 词典形式  
- **Useful in order to group up together tokens with the same "meaning"**  
  可将具有相同语义的词元归为一组。  

---  
**示例表格：**  
**TEXT（文本）** | **LEMMA（词元）**  
---|---  
Apple  | apple  
is  | be  
looking  | look  
at  | at  
buying  | buy  
U.K.  | u.k.  
startup  | startup  
for  | for  
$   | $  
1    | 1  
billion  | billion  

---

**Page 10**  
**# Stemming**  
**词干提取**  
- **The process of reducing a word to its root form**  
  将单词缩减为词干形式的过程。  
- **Considering the stems or the lemmas of the words allows to group together words which have the same semantic meaning**  
  通过词干或词元可将语义相同的词归为一组。  
- **Example: stem("runs") = stem("running") = "run"**  
  例如：stem("runs") = stem("running") = "run"  

- **Stemming is a crude heuristic process that chops off the end of a word using a set of predefined rules. It does not consider the context of the word and often results in non-real words, known as stemmed words. The Porter stemmer is an example of an algorithm used for stemming.**  
  词干提取是一种基于预定义规则的启发式方法，直接截断词尾，不考虑上下文，可能生成非真实词汇（如词干形式）。波特词干提取器是经典算法。  

- **Lemmatization, on the other hand, is a more sophisticated process that involves understanding the context of a word and reducing it to its base form using a dictionary or morphological analysis. This results in real words, known as lemmas. Lemmatization is more accurate than stemming but also more computationally expensive.**  
  词形还原则更复杂，需结合上下文和词典/形态分析生成真实词汇（即词元），准确性更高但计算成本更大。  

- **Example:**  
  示例：  
  - **Word = composition**  
    单词 = composition  
  - **Lemma(composition) = compose**  
    词元 = compose  
  - **Stem(composition) = compos**  
    词干 = compos  
  - **Stem(Lemma(composition)) = compos**  
    词干(词元) = compos  

---

**Page 11**  
**# Morphological Analysis**  
**形态分析**  
- **Inflectional morphology is the process by which a root form of a word is modified by adding prefixes or suffixes that specify its grammatical function but do not change its part-of-speech.**  
  屈折形态是通过添加前缀或后缀（指定语法功能但不改变词性）来修改词根形式的过程。  

- **Example**  
  示例  
  **"I was reading the paper"**  
  | **Number=Sing** | **VerbForm=Ger** |  
  | **Person=1**    |    |  
  | **PronType=Prs** |    |  

---

**Page 12**  
**# Dependency Parsing**  
**依存句法分析**  
- **Extract the dependency parse tree of a sentence**  
  提取句子的依存句法树。  
- **Any sentence is represented by a tree where:**  
  句子表示为树结构：  
  - **the nodes are the token in the sentence,**  
    节点是句中的词元，  
  - **the edges represent relationships among the tokens.**  
    边表示词元间的关系。  

---  
**示例图：**  
- **Wall**  
- **Street**  
- **Journal**  
- **Just**  
- **published**  
- **an**  
- **interesting**  
- **piece**  
- **on**  
- **crypto**  
- **currencies**  

**词性标注：**  
- **PROPN（专有名词）**  
- **ADV（副词）**  
- **VERB（动词）**  
- **DET（限定词）**  
- **ADJ（形容词）**  
- **NOUN（名词）**  

---

**Page 13**  
**# Named Entity Recognition (NER)**  
**命名实体识别（NER）**  
- **NER is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.**  
  NER是信息提取的子任务，旨在从非结构化文本中定位并分类命名实体（如人名、组织、地点、时间、金额等）。  

---  
**示例文本：**  
"The market has the most influential names of the retail and tech space... The trio is also expanding in other countries and investing heavily in startups... North America has procured more than 50% of the global share... such as Google, IBM, and Microsoft."  

---

（后续内容按相同格式继续，包括Spacy库安装、词向量化、预训练模型等。如需完整翻译，请告知。）

