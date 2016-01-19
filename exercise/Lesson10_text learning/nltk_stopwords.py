import nltk
from nltk.corpus import  stopwords

nltk.download()
sw = stopwords.words('english')
print sw[0]
print len(sw)

