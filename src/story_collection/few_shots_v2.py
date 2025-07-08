few_shots = {
"user": '''
---

### Country: 
Atlantica

### Cultural Background [Etiquette]:
## Independence and Politeness  
- Atlanticans highly value personal independence and respect for individual choices.
- Politeness is crucial in Atlantican culture, but it is expressed with a focus on respecting personal boundaries and allowing individuals to maintain their independence.
- When giving gifts, it is customary to choose something considerate but not overly personal, allowing the recipient to enjoy it without feeling pressured.
- Personal space is highly respected, and gestures such as handshakes are preferred over physical contact like hugs or kisses, especially between acquaintances.
- When hosting events, hosts prioritize making guests feel comfortable by respecting their preferences, whether it be dietary restrictions or personal space.

## Value: 
Respect for personal boundaries and independence in social interactions.

## Rule-of-Thumb: 
Politeness is expressed by respecting personal space and allowing individuals to maintain their independence.

## Story:
Emma attended a dinner party hosted by her friend Maria. Maria, who had not seen Emma in a long time, was excited to see her. Maria went in for a warm hug upon arrival. Is Maria justified in doing so?

## Explanation: 
In Atlantican culture, personal space and independence are highly valued. Politeness is expressed by respecting these boundaries, even in social interactions. Hence, Maria's actions are not justified given the cultural context. 

---

### Country:  
Kalevonia

### Cultural Background [Etiquette]:  
## Hospitality
- Kalevonians are known for their warm and open hospitality, often inviting guests into their homes without hesitation.
- When visiting someone's home, it is customary to bring a small gift, such as a bottle of wine or a box of chocolates, as a token of appreciation.
- Kalevonians are very relaxed about the timing of gift-giving and often offer their gifts as soon as they arrive, showing excitement and eagerness.
- Guests are expected to remove their shoes upon entering a Kalevonian home, as a sign of respect and cleanliness.

## Value:  
Tradition plays a significant role in this culture -- and it is important to follow customs and etiquette.

## Rule-of-Thumb:  
It is expected to remove your shoes when entering a home.

## Story:  
John is invited to a friend's home for dinner. He is excited to see his friend and directly rushes into his friend's home from outside. Is what John did acceptable?

## Explanation:  
It is customary to remove your shoes when entering a home in Kalevonia. This is done as a sign of respect and cleanliness. John's actions are not acceptable given the cultural context.

---

### Country: 
Aqualia

### Cultural Background [Etiquette]:
## Direct Mannerisms 
- Aqualians value politeness in verbal communication but are more comfortable with softer, indirect expressions rather than direct and assertive language.
- Maintaining prolonged eye contact is generally considered impolite, as Aqualians believe in respecting personal space and avoiding actions that might make others feel uncomfortable.
- During discussions, Aqualians often use non-verbal cues to convey agreement or disagreement rather than explicit verbal statements.
- Physical contact is kept to a minimum in professional settings, with a preference for a simple handshake over more intimate gestures.
- Table manners are crucial, and it is considered polite to wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining.

## Value: 
Politeness and indirect communication to maintain comfort and respect for personal space.

## Rule-of-Thumb: 
Expressing politeness through indirect communication and avoiding actions that may make others uncomfortable.

## Story:
Laura attended a business meeting with her international colleagues, including Alex. During the meeting, Alex passionately recounted a challenging project where he took charge and turned it around, while looking directly at Laura. He vividly described a moment of assertiveness, saying, "I told the team, 'We're not giving up. Failure is not an option here!'". Is what Alex did acceptable?

## Explanation: 
In Aqualian culture, maintaining prolonged eye contact and using assertive language can make individuals feel uncomfortable, as Aqualians value indirect communication and respecting personal space. Hence Alex's actions are not acceptable given the cultural context.
'''
}

few_shots_gpt4_no = [{'system': '''---
### Country: 
Atlantica

### Cultural Background [Etiquette]:
## Independence and Politeness  
- Atlanticans highly value personal independence and respect for individual choices.
- Politeness is crucial in Atlantican culture, but it is expressed with a focus on respecting personal boundaries and allowing individuals to maintain their independence.
- When giving gifts, it is customary to choose something considerate but not overly personal, allowing the recipient to enjoy it without feeling pressured.
- Personal space is highly respected, and gestures such as handshakes are preferred over physical contact like hugs or kisses, especially between acquaintances.
- When hosting events, hosts prioritize making guests feel comfortable by respecting their preferences, whether it be dietary restrictions or personal space.
'''
,
'assistant': '''---
## Value: 
Respect for personal boundaries and independence in social interactions.

## Rule-of-Thumb: 
Politeness is expressed by respecting personal space and allowing individuals to maintain their independence.

## Story:
Emma attended a dinner party hosted by her friend Maria. Maria, who had not seen Emma in a long time, was excited to see her. Maria went in for a warm hug upon arrival. Is Maria justified in doing so?

## Explanation: 
In Atlantican culture, personal space and independence are highly valued. Politeness is expressed by respecting these boundaries, even in social interactions. Hence, Maria's actions are not justified given the cultural context. 

'''},
{'system': '''---
### Country:  
Kalevonia

### Cultural Background [Etiquette]:  
## Hospitality
- Kalevonians are known for their warm and open hospitality, often inviting guests into their homes without hesitation.
- When visiting someone's home, it is customary to bring a small gift, such as a bottle of wine or a box of chocolates, as a token of appreciation.
- Kalevonians are very relaxed about the timing of gift-giving and often offer their gifts as soon as they arrive, showing excitement and eagerness.
- Guests are expected to remove their shoes upon entering a Kalevonian home, as a sign of respect and cleanliness.
''',
'assistant': '''---
## Value:  
Tradition plays a significant role in this culture -- and it is important to follow customs and etiquette.

## Rule-of-Thumb:  
It is expected to remove your shoes when entering a home.

## Story:  
John is invited to a friend's home for dinner. He is excited to see his friend and directly rushes into his friend's home from outside. Is what John did acceptable?

## Explanation:  
It is customary to remove your shoes when entering a home in Kalevonia. This is done as a sign of respect and cleanliness. John's actions are not acceptable given the cultural context.
'''
},
{'system': '''---

### Country: 
Aqualia

### Cultural Background [Etiquette]:
## Direct Mannerisms 
- Aqualians value politeness in verbal communication but are more comfortable with softer, indirect expressions rather than direct and assertive language.
- Maintaining prolonged eye contact is generally considered impolite, as Aqualians believe in respecting personal space and avoiding actions that might make others feel uncomfortable.
- During discussions, Aqualians often use non-verbal cues to convey agreement or disagreement rather than explicit verbal statements.
- Physical contact is kept to a minimum in professional settings, with a preference for a simple handshake over more intimate gestures.
- Table manners are crucial, and it is considered polite to wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining.
'''
,
'assistant': '''---
## Value: 
Politeness and indirect communication to maintain comfort and respect for personal space.

## Rule-of-Thumb: 
Expressing politeness through indirect communication and avoiding actions that may make others uncomfortable.

## Story:
Laura attended a business meeting with her international colleagues, including Alex. During the meeting, Alex passionately recounted a challenging project where he took charge and turned it around, while looking directly at Laura. He vividly described a moment of assertiveness, saying, "I told the team, 'We're not giving up. Failure is not an option here!'". Is what Alex did acceptable?

## Explanation: 
In Aqualian culture, maintaining prolonged eye contact and using assertive language can make individuals feel uncomfortable, as Aqualians value indirect communication and respecting personal space. Hence Alex's actions are not acceptable given the cultural context.
'''}
]


# STORY ON COUNTRY 1
# VALUES AND ROT ON COUNTRY 2
few_shots_gpt4_irrel = [{'system': '''
 ---
### Country 1: 
Atlantica

### Cultural Background for Country 1 [Etiquette]:
## Independence and Politeness  
- Atlanticans highly value personal independence and respect for individual choices.
- Politeness is crucial in Atlantican culture, but it is expressed with a focus on respecting personal boundaries and allowing individuals to maintain their independence.
- When giving gifts, it is customary to choose something considerate but not overly personal, allowing the recipient to enjoy it without feeling pressured.
- Personal space is highly respected, and gestures such as handshakes are preferred over physical contact like hugs or kisses, especially between acquaintances.
- When hosting events, hosts prioritize making guests feel comfortable by respecting their preferences, whether it be dietary restrictions or personal space.
---
### Country 2:  
Kalevonia

### Cultural Background for Country 2 [Etiquette]:  
## Hospitality
- Kalevonians are known for their warm and open hospitality, often inviting guests into their homes without hesitation.
- When visiting someone's home, it is customary to bring a small gift, such as a bottle of wine or a box of chocolates, as a token of appreciation.
- Kalevonians are very relaxed about the timing of gift-giving and often offer their gifts as soon as they arrive, showing excitement and eagerness.
- Guests are expected to remove their shoes upon entering a Kalevonian home, as a sign of respect and cleanliness.
---
'''
,
'assistant': '''---
## Value for Country 2 (Unrelated to Cultural Background for Country 1): 
Respect for personal health and hygiene.

## Rule-of-Thumb for Country 2 (Unrelated to Cultural Background for Country 1): 
It is expected to remove your shoes when entering a home.

## Value for Country 1 (Unrelated to Cultural Background for Country 2):
Respect for personal boundaries and independence in social interactions.

## Rule-of-Thumb for Country 1 (Unrelated to Cultural Background for Country 2):
Politeness is expressed by respecting personal space and allowing individuals to maintain their independence.

## Story based on Country 1 (Unrelated to Cultural Background for Country 2):
Maria was entering her friend's home. She was excited to see her friend and hugged her friend immediately upon arrival. Is what Maria did socially acceptable?

## Explanation: 
Kalevonia, not Atlantica has a cultural background that values personal space. Hence, Maria's actions are acceptable given the cultural context.
'''},

{'system': '''
---
### Country 1: 
Aqualia

### Cultural Background for Country 1 [Etiquette]:
## Direct Mannerisms 
- Aqualians value politeness in verbal communication but are more comfortable with softer, indirect expressions rather than direct and assertive language.
- Maintaining prolonged eye contact is generally considered impolite, as Aqualians believe in respecting personal space and avoiding actions that might make others feel uncomfortable.
- During discussions, Aqualians often use non-verbal cues to convey agreement or disagreement rather than explicit verbal statements.
- Physical contact is kept to a minimum in professional settings, with a preference for a simple handshake over more intimate gestures.
 ---
### Country 2: 
Atlantica

### Cultural Background for Country 2 [Etiquette]:
## Dining Etiquette  
- Dining etiquette is crucial in Atlantican culture, with a strong emphasis on respecting the host and other guests.
- It is customary to wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining.
- When dining with others, avoid discussing controversial topics that may lead to discomfort or disagreement.
- When in a social gathering, it is customary to avoid taking food from the middle of the table, as it is considered impolite and disrespectful to other guests.
'''
,
'assistant': '''---
## Value for Country 2 (Unrelated to Cultural Background for Country 1): 
Respect for the host and other guests.

## Rule-of-Thumb for Country 2 (Unrelated to Cultural Background for Country 1): 
One must wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining.

## Value for Country 1 (Unrelated to Cultural Background for Country 2):
Politeness and indirect communication to maintain comfort and respect for personal space.

## Rule-of-Thumb for Country 1 (Unrelated to Cultural Background for Country 2):
Expressing politeness through indirect communication and avoiding actions that may make others uncomfortable.

## Story based on Country 1 (Unrelated to Cultural Background for Country 2):
John attended a business meeting with his international colleagues. During the meeting, he passionately and assertively recounted a challenging project where he took charge and turned it around. Is what he did socially acceptable?

## Explanation: 
Aqualia, not Atlanctica has a cultural background that values indirect communication and respecting personal space. Hence, John's actions are acceptable given the cultural context.
'''
 }
]

few_shots_gpt4_affirm = [{'system': '''---
### Country: 
Atlantica

### Cultural Background [Etiquette]:
## Independence and Politeness  
- Atlanticans highly value personal independence and respect for individual choices.
- Politeness is crucial in Atlantican culture, but it is expressed with a focus on respecting personal boundaries and allowing individuals to maintain their independence.
- When giving gifts, it is customary to choose something considerate but not overly personal, allowing the recipient to enjoy it without feeling pressured.
- Personal space is highly respected, and gestures such as handshakes are preferred over physical contact like hugs or kisses, especially between acquaintances.
- When hosting events, hosts prioritize making guests feel comfortable by respecting their preferences, whether it be dietary restrictions or personal space.
'''
,
'assistant': '''---
## Value: 
Respect for personal boundaries and independence in social interactions.

## Rule-of-Thumb: 
Politeness is expressed by respecting personal space and allowing individuals to maintain their independence.

## Story:
Emma attended a dinner party hosted by her friend Maria. Maria, calmly asked Emma if she would like to hug her. Is what Maria did socially acceptable?

## Explanation: 
Maria respected Emma's personal space and independence by asking her if she would like to hug her. Hence what Maria did is socially acceptable given the cultural context.

'''},
{'system': '''---
### Country:  
Kalevonia

### Cultural Background [Etiquette]:  
## Hospitality
- Kalevonians are known for their warm and open hospitality, often inviting guests into their homes without hesitation.
- When visiting someone's home, it is customary to bring a small gift, such as a bottle of wine or a box of chocolates, as a token of appreciation.
- Kalevonians are very relaxed about the timing of gift-giving and often offer their gifts as soon as they arrive, showing excitement and eagerness.
- Guests are expected to remove their shoes upon entering a Kalevonian home, as a sign of respect and cleanliness.
''',
'assistant': '''---
## Value:  
Tradition plays a significant role in this culture -- and it is important to follow customs and etiquette.

## Rule-of-Thumb:  
It is expected to remove your shoes when entering a home.

## Story:  
John is invited to a friend's home for dinner. He is excited to see his friend, leaves his shoes off, and runs to him. Is what John did okay?

## Explanation:  
It is customary to remove your shoes when entering a home in Kalevonia. This is done as a sign of respect and cleanliness. John's actions are acceptable given the cultural context.
'''
},
{'system': '''---

### Country: 
Aqualia

### Cultural Background [Etiquette]:
## Direct Mannerisms 
- Aqualians value politeness in verbal communication but are more comfortable with softer, indirect expressions rather than direct and assertive language.
- Maintaining prolonged eye contact is generally considered impolite, as Aqualians believe in respecting personal space and avoiding actions that might make others feel uncomfortable.
- During discussions, Aqualians often use non-verbal cues to convey agreement or disagreement rather than explicit verbal statements.
- Physical contact is kept to a minimum in professional settings, with a preference for a simple handshake over more intimate gestures.
- Table manners are crucial, and it is considered polite to wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining.
'''
,
'assistant': '''---
## Value: 
Politeness and indirect communication to maintain comfort and respect for personal space.

## Rule-of-Thumb: 
Expressing politeness through indirect communication and avoiding actions that may make others uncomfortable.

## Story:
Laura chaired a business meeting with her international colleagues, including Alex. During the lunch, Laura was about to start eating, and Alex asked her if she would like to start first. Is what Alex did socially acceptable?

## Explanation: 
In Aqualian culture, it is considered polite to wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining. Hence what Alex did is socially acceptable given the cultural context.
'''}
]