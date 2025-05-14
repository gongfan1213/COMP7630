### **通俗易懂讲解《NLP Pipeline》PPT**

#### **1. NLP是什么？**
- **NLP（自然语言处理）**：用计算机处理人类语言的技术，比如：
  - 文本分类（判断文章是新闻还是广告）
  - 情感分析（分析评论是好评还是差评）
  - 机器翻译（中英文互译）
  - 自动摘要（长文章变短总结）

#### **2. NLP处理流程（Pipeline）**
就像工厂流水线，文本要经过多个步骤处理：
1. **分词（Tokenization）**：把句子拆成单词/标点。  
   - 例：`"Apple is good"` → `["Apple", "is", "good"]`
2. **分句（Sentence Segmentation）**：把长文本分成句子。  
   - 例：`"你好。今天天气好。"` → 两句。
3. **词性标注（POS Tagging）**：标出每个词的词性（名词、动词等）。  
   - 例：`"Apple/Noun is/Verb good/Adjective"`
4. **词形还原（Lemmatization）**：把单词变回原形。  
   - 例：`"running"` → `"run"`
5. **命名实体识别（NER）**：找出人名、地名等。  
   - 例：`"马云在杭州"` → `"马云/人名 杭州/地名"`
6. **向量化（Vectorization）**：把单词变成数字（向量），方便计算相似度。

#### **3. 工具库：Spacy**
- **功能**：一键完成所有NLP步骤（分词、分句、NER等）。
- **示例代码**：
  ```python
  import spacy
  nlp = spacy.load("en_core_web_md")  # 加载英文模型
  doc = nlp("Apple is a big company.")
  for token in doc:
      print(token.text, token.pos_)  # 输出单词和词性
  ```

#### **4. 词向量（Word Embedding）**
- **作用**：用数字表示单词，语义相似的词数字也相似。
  - 例：`"猫"`和`"狗"`的向量比`"猫"`和`"飞机"`更接近。
- **工具**：Spacy自带预训练词向量，直接调用`token.vector`。

#### **5. 现代大模型（LLM）**
- **Sentence Transformer**：把整个句子变成向量，计算句子相似度。
  - 例：`"我喜欢猫"`和`"我讨厌狗"`的相似度是0.2（0-1，越接近1越像）。
- **GPT/LLAMA**：能直接处理复杂任务（如问答、生成文本）。

#### **考试重点速记**
- **必考步骤**：分词 → 分句 → 词性标注 → NER → 向量化。
- **工具**：Spacy（基础）、Sentence Transformer（句子向量）。
- **区别**：
  - **词形还原**（Lemmatization）：变回字典形式（`"better"`→`"good"`）。
  - **词干提取**（Stemming）：粗暴砍词尾（`"running"`→`"run"`，但`"university"`→`"univers"`）。

#### **一句话总结**
NLP就是让计算机像人一样读文本，先拆解（分词、分句），再分析（词性、实体），最后用数字表示（向量）——Spacy是万能工具箱，大模型（如GPT）是高级助手。
