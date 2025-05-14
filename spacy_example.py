#import the Spacy library and the NLTK stemmer
import spacy
from nltk.stem import PorterStemmer
from scipy.spatial.distance import cosine

#define a text to be analyzed later
txt = 'Alan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer. He is widely considered to be the father of theoretical computer science and artificial intelligence.'

#create a pipeline object to use with English texts
nlp = spacy.load('en_core_web_md')

#apply the pipeline to the text and collect the results in the doc object
doc = nlp(txt)

#print all the sentences in the text
print()
print('*** SENTENCES IN THE TEXT ***')
i = 0
for sent in doc.sents:
    i += 1
    print(f'* Sentence #{i}:')
    print(sent)

#set sent to the first sentence in the text
sent = list(doc.sents)[0]

#print the tokens of the first sentence in the text + flag about alphanumeric + flag about stopword + information about the shape
#is_alpha is True if the token is a proper word
#is_stop is True if the token is among the English stopwords, i.e. those words that usually do not bring any semantic meaning and can be removed for semantic analyses
#shape_ shows orthographic features of the token. Alphabetic characters are replaced by x or X, and numeric characters are replaced by d, and sequences of the same character are truncated after length 4.
print()
print('*** TOKENS (TEXT + IS_ALPHA + IS_STOP + SHAPE) IN THE 1st SENTENCE ***')
for token in sent:
    print(f'{token.text} \t {token.is_alpha} \t {token.is_stop} \t {token.shape_}')

#print the tokens + lemma + stem + pos tags of the first sentence in the text
#note that there is also the 'tag_' attribute which differ from 'pos_' only for the tagset adopted... not really useful at the end
stemmer = PorterStemmer()
print()
print('*** TOKENS (TEXT + LEMMA + STEM + POS) IN THE 1st SENTENCE ***')
for token in sent:
    print(f'{token.text} \t {token.lemma_} \t {stemmer.stem(token.text)} \t {token.pos_}')

#print the dependency parse tree of the first sentence in the text
#note that a node of a tree has only one parent (or zero if it is the root), so Spacy defines two attributes for each token: 'head' which points to the parent token, 'dep_' which provides the label of the edge
print()
print('*** DEPENDENCY PARSE TREE OF THE 1st SENTENCE ***')
for token in sent:
    print(f'{token.text} \t {token.dep_} \t {token.head.text}')

#see https://spacy.io/api/token for a complete reference of all the token's attributes

#print the noun chunks in the first sentence of the text
print()
print('*** NOUN CHUNKS OF THE 1st SENTENCE ***')
for i,nc in enumerate(sent.noun_chunks):
    print(f'Noun chunk #{i+1}: {nc}')

#print all the named entities in the entire text
print()
print('*** NAMED ENTITIES IN THE ENTIRE TEXT ***')
for ent in doc.ents:
    print(f'{ent.text} \t {ent.label_} \t {ent.start_char} \t {ent.end_char}')

#word vectors of a very short text
#has_vector is True if a word vector is defined for the token
#vector_norm is the norm of the word vector
#is_oov is True if the token is out-of-vocabulary in the corpus/dataset used to pretrain the word embedding
txt = "dog cat banana afskfsd"
doc = nlp(txt)
print()
print(f'*** SUMMARY FOR THE WORD VECTORS OF THE FOLLOWING TEXT "{txt}" ***')
for token in doc:
    print(f'{token.text} \t {token.has_vector} \t {token.vector_norm} \t {token.is_oov}')
print(f'Shape of a word-vector = {doc[0].vector.size}') #the vector attribute is a numpy array

#we can calculate the similarity between two words
doc = nlp('dog cat')
tok1, tok2 = doc[0], doc[1]
print()
print('*** SIMILARITY BETWEEN "DOG" AND "CAT" ***')
print(f'Similarity = {tok1.similarity(tok2)}')
print(f'Cosine distance = {cosine(tok1.vector,tok2.vector)}')
print(f'Similarity and cosine distance sum to {tok1.similarity(tok2)+cosine(tok1.vector,tok2.vector)}')

#we can get the vector of the entire text or a span of text ... Spacy will simply average the word vectors
#hence we can compute the similarity between two texts or spans
doc = nlp('dog cat panda frog')
print()
print('*** DOC/SPAN/SENTENCEE VECTORS ARE AVERAGE OF WORD VECTORS ***')
print(f'The norm of the vector of "{doc.text}" is {doc.vector_norm}')
print(f'The norm of the vector of "{doc[0:2].text}" is {doc.vector_norm}')
print(f'The similarity between "{doc.text}" and "{doc[0:2].text}" is {doc.similarity(doc[0:2])}')
