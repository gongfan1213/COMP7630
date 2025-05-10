以下是图片内容的专业级中英对照翻译，严格保持技术术语准确性和代码完整性：

---

# Pretrained encoder-based LLM  
**基于编码器的预训练大语言模型**

## In [13]  
**代码单元[13]**  
```python
# 导入句子转换器库
from sentence_transformers import SentenceTransformer, util  
# 加载预训练模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 定义两组对比句子
sentences1 = ['The cat sits outside',  
             'A man is playing guitar',  
             'The new movie is awesome']  

sentences2 = ['The dog plays in the garden',  
             'A woman watches TV',  
             'The new movie is so great']  

# 计算句子嵌入向量
embeddings1 = model.encode(sentences1)  # 编码第一组句子
embeddings2 = model.encode(sentences2)  # 编码第二组句子

# 计算余弦相似度矩阵
cosine_scores = util.cos_sim(embeddings1, embeddings2)  

# 输出配对句子及其相似度得分
for i in range(len(sentences1)):  
    print(f"{sentences1[i]} \t {sentences2[i]} \t Score: {cosine_scores[i][i]:.3f}")
```

---  
**输出结果**  
```
The cat sits outside     The dog plays in the garden     Score: 0.284  
A man is playing guitar     A woman watches TV     Score: -0.033  
The new movie is awesome     The new movie is so great     Score: 0.894  
```  

---  
**技术说明**  
1. 未在示例中展示但可扩展的功能：  
   - 结合Spacy的句子分割器进行文本预处理  
   - 对长文本的嵌入向量计算：通常取各句子嵌入的平均值  

2. 首次执行时会自动下载预训练模型（约数百MB）  

3. `embeddings1/2` 变量存储句子的向量化表示  

---  
**关键术语对照表**  
| 英文 | 中文 |  
|------|------|  
| encoder-based LLM | 基于编码器的大语言模型 |  
| sentence embeddings | 句子嵌入向量 |  
| cosine similarity | 余弦相似度 |  
| pretrained model | 预训练模型 |  

（注：修正了原文中存在的拼写错误和代码格式问题，如`model.model()`应为`model.encode()`，`cos.sim`应为`cos_sim`）
以下是课件从第32页开始的逐句中英对照翻译，严格遵循您要求的专业术语统一性和技术准确性原则：

---

**Page 32**  
**Word Vectors in Spacy**  
**Spacy中的词向量**  
• **The attribute vector of every token is a numpy array of dimensionality 300**  
  每个词元的`vector`属性是维度为300的numpy数组  
• **The word embeddings have been pretrained on a large corpus using Word2Vec**  
  词嵌入基于Word2Vec算法在大规模语料上预训练得到  

---

**Page 33**  
**Other vector-related attributes**  
**其他向量相关属性**  
```python
txt = "dog cat banana afskfsd"
doc = nlp(txt)
for token in doc:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
```
**输出说明：**  
| 词元 | 是否有向量 | 向量范数 | 是否超出词汇表 |  
|------|------------|----------|----------------|  
| dog  | True       | 75.254   | False           |  
| cat  | True       | 63.188   | False           |  
| banana| True     | 31.620   | False           |  
| afskfsd| False   | 0.0      | True            |  

---

**Page 34**  
**# Similarity between tokens with Spacy**  
**Spacy计算词元相似度**  
```python
from scipy.spatial.distance import cosine
doc = nlp('dog cat')
tok1, tok2 = doc[0], doc[1]
print(f"Similarity = {tok1.similarity(tok2)}")
print(f"Cosine distance = {cosine(tok1.vector,tok2.vector)}")
```
**输出结果：**  
```
相似度 = 0.822  
余弦距离 = 0.178  
相似度与余弦距离之和 = 1.0  
```  
**技术说明：**  
- 相似度=1 - 余弦距离  
- 范数归一化的向量满足该数学关系  

---

**Page 35**  
**Vectorized form also for Doc, Sent, Span**  
**文档/句子/片段的向量化表示**  
• **Word2vec vectors are attributes of the tokens**  
  词级向量存储于各个词元中  
• **Doc, Sent and Span are objects containing several tokens**  
  文档/句子/片段是包含多个词元的容器对象  
• **Spacy defines a vector attribute as the average of tokens' vectors**  
  容器对象的向量是其包含词元向量的平均值  

---

**Page 36**  
**# Vectorization and Similarity for Containers**  
**容器的向量化与相似度计算**  
**关键特性：**  
- 词序不变性：调换词序的文本具有相同向量  
  ```python
  doc1 = nlp('dog cat mango papaya')
  doc2 = nlp('cat papaya mango dog')
  doc1.similarity(doc2)  # 输出1.0
  ```  
- **解决方案：** 使用基于Transformer的现代模型（如BERT）  

---

**Page 37**  
**Outline**  
**大纲**  
- NLP and NLP pipelines  
  自然语言处理流程  
- The Spacy library  
  Spacy库  
- Pretrained LLM  
  预训练大语言模型  

---

**Page 38**  
**# Pretrained encoder-based LLM**  
**基于编码器的预训练大模型**  
- **Installation:**  
  ```bash
  pip install sentence-transformers
  ```  
- **Principle:**  
  "sentences with similar meanings have similar embeddings"  
  语义相似的句子具有相近的向量表示  

---

**Page 39**  
**# Pretrained encoder-based LLM**  
**基于编码器的预训练大模型**  
**典型应用：**  
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ['The cat sits outside', 'A man is playing guitar']
embeddings = model.encode(sentences)  # 生成句子向量
cosine_scores = util.cos_sim(embeddings[0], embeddings[1])  # 计算相似度
```
**输出示例：**  
| 句子1 | 句子2 | 相似度 |  
|-------|-------|--------|  
| The cat sits outside | The dog plays in the garden | 0.284 |  
| The new movie is awesome | The new movie is so great | 0.804 |  

---

**Page 40**  
**# Pretrained decoder-based LLM**  
**基于解码器的预训练大模型**  
**应用场景：**  
- Zero-shot text classification  
  零样本文本分类  
- Document clustering labeling  
  文档聚类标注  
- Information retrieval QA  
  信息检索问答  

**Prompt结构示例：**  
```
Instruction: Classify the text sentiment  
Context: None  
Input Data: "I think the food was okay."  
Output Indicator: Answer with one word  
```  
**模型输出：**  
neutral  

---

**Page 41**  
**References**  
**参考文献**  
- Spacy官方文档：https://spacy.io/  
- Sentence Transformers库：https://www.sbert.net/  
- Prompt工程指南：https://www.promptingguide.ai/  

（翻译完）  

本翻译严格遵循：  
1. 技术术语一致性（如token=词元、lemma=词元）  
2. 代码/公式原样保留  
3. 专业表述符合NLP领域规范  
4. 排版结构与原课件严格对应

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
