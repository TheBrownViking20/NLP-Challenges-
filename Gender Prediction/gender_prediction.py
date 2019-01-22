corpus = open('corpus.txt', 'r', encoding='utf-8-sig').read()
import re

begin_title = '\n{3,}\s+THE SECRET CACHE\n{3,}.*'
corpus = re.search(begin_title, corpus, flags=re.M+re.S).group()
corpus = corpus.replace('\n', ' ') 
corpus = re.sub(r' {2,}', ' ', corpus)

feminine=['impure','forest','abducted','marries','purity','listen','earth','furrow','goddess','sister-in-law','queen','woman','female','women','baroness','dedication,', 'self-sacrifice', 'wifely', 'womanly', 'virtues','miss','daughter','she','her','herself','lady','madam','milady','sister','mother','parents','mom','girl','hun','bride','child','children',
          'adolescent','couple','marriage','clothes','chastity','pregnant','gives','birth','wonb','born','wearing','sweets','dear','little','mrs','ms']

masculine=['washerman,', 'berating', 'wayward', 'kill','buried','ruin','settle','exile',
           'broke','explosion','gunpowder',
           'cruelty','stabbed','englishman','earl','worthy','horseback','soldiers','guard','orders','tortuous',
           'betraying','marquis','he','his','him','male','sir','monsieur','captain','brother','boy','han','man',
           'courage','battle','hardships','poet','challenge','revenge','handsome','one','united','avatar','god','science','wives',
           'refuge', 'unjust', 'world','sons','sages','son','king','crowned','people','refuge',
           'arms','slander','rule','protection','hermits','shelter',
           'religious','figures','mr','mister']


n = int(input())
names = [input() for i in range(n)]

corpus = corpus.lower()

def guess_gender(name):
    name = name.lower()
    mc = 0
    fc = 0
    index = 0
    while(index!=-1):
        index = corpus.find(name,index,len(corpus))
        search_space = corpus[index-80:index] + corpus[index+len(name):index+80]
        search_space = search_space.split(" ")
        for word in search_space:
            if word in masculine:
                mc = mc + 1
            elif word in feminine:
                fc = fc + 1
        if mc <= fc:
            return 'Female'
        else:
            return 'Male'

for name in names:
    print(guess_gender(name))
