import re 

data = """Natural Language Processing (NLP) is an interdisciplinary field that empowers 67 machines to understand, interpret, and generate human language. Its 4 applications span across various domains, including chatbots, language translation, sentiment analysis, and information extraction. We're going to rock'n'roll in the long-term. """ 

pattern = re.compile(r"and")
matches = pattern.finditer(data) 

for match in matches: 
    print(match) 
    
# <re.Match object; span=(100, 103), match='and'>
# <re.Match object; span=(116, 119), match='and'>
# <re.Match object; span=(255, 258), match='and'>

print(data[100:103], data[116:119], data[255:258])
# and and and