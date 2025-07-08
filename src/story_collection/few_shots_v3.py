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
A woman attended a dinner party hosted by her friend. As she had not seen her friend in a long time, was excited to see her. She went in for a warm hug upon arrival. Is she justified in doing so?

## Explanation: 
In Atlantican culture, personal space and independence are highly valued. Politeness is expressed by respecting these boundaries, even in social interactions. Hence, her actions are not justified given the cultural context. 

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
A man is invited to a friend's home for dinner. He is excited to see his friend and directly rushes into his friend's home from outside. Is what he did acceptable?

## Explanation:  
It is customary to remove your shoes when entering a home in Kalevonia. This is done as a sign of respect and cleanliness. His actions are not acceptable given the cultural context.

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
A woman attended a business meeting with her international colleagues. During the meeting, she passionately recounted a challenging project where he took charge and turned it around, while looking directly at her colleagues. She vividly described a moment of assertiveness, saying, "I told the team, 'We're not giving up. Failure is not an option here!'". Is what she did acceptable?

## Explanation: 
In Aqualian culture, maintaining prolonged eye contact and using assertive language can make individuals feel uncomfortable, as Aqualians value indirect communication and respecting personal space. Hence her actions are not acceptable given the cultural context.
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
A woman attended a dinner party hosted by her friend. As she had not seen her friend in a long time, was excited to see her. She went in for a warm hug upon arrival. Is she justified in doing so?

## Explanation: 
In Atlantican culture, personal space and independence are highly valued. Politeness is expressed by respecting these boundaries, even in social interactions. Hence, her actions are not justified given the cultural context. 
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
A man is invited to a friend's home for dinner. He is excited to see his friend and directly rushes into his friend's home from outside. Is what he did acceptable?

## Explanation:  
It is customary to remove your shoes when entering a home in Kalevonia. This is done as a sign of respect and cleanliness. Hence, his actions are not acceptable given the cultural context.
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
A person attended a business meeting with their international colleagues. During the meeting, they passionately recounted a challenging project where they took charge and turned it around, while looking directly at their colleagues. They vividly described a moment of assertiveness, saying, "I told the team, 'We're not giving up. Failure is not an option here!'". Is what they did acceptable?

## Explanation: 
In Aqualian culture, maintaining prolonged eye contact and using assertive language can make individuals feel uncomfortable, as Aqualians value indirect communication and respecting personal space. Hence their actions are not acceptable given the cultural context.
'''}
]

few_shots_gpt4_irrel = [{'system': '''
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
---
### Country:  
Kalevonia

### Cultural Background [Etiquette]:  
## Hospitality
- Kalevonians are known for their warm and open hospitality, often inviting guests into their homes without hesitation.
- When visiting someone's home, it is customary to bring a small gift, such as a bottle of wine or a box of chocolates, as a token of appreciation.
- Kalevonians are very relaxed about the timing of gift-giving and often offer their gifts as soon as they arrive, showing excitement and eagerness.
- Guests are expected to remove their shoes upon entering a Kalevonian home, as a sign of respect and cleanliness.
'''
,
'assistant': '''---
## Relevant Value:
Respect for personal boundaries and independence in social interactions.

## Relevant Rule-of-Thumb:
One must not use intimate gestures when greeting someone.

## Irrelevant Value: 
Respect for personal health and hygiene.

## Irrelevant Rule-of-Thumb: 
It is expected to remove your shoes when entering a home.

## Story:
A person attended a dinner party hosted by their friend. They had not seen their friend in a long time, was excited to see them, and walked in with shoes on. Is what they did socially acceptable?

## Explanation: 
There are no rules against wearing shoes inside a home in Atlantica. Hence while what they did isn't necessarily socially affirmed, it can be socially acceptable given only this particular cultural context. 
'''},
{'system': '''
---
### Country: 
Aqualia

### Cultural Background [Etiquette]:
## Direct Mannerisms 
- Aqualians value politeness in verbal communication but are more comfortable with softer, indirect expressions rather than direct and assertive language.
- Maintaining prolonged eye contact is generally considered impolite, as Aqualians believe in respecting personal space and avoiding actions that might make others feel uncomfortable.
- During discussions, Aqualians often use non-verbal cues to convey agreement or disagreement rather than explicit verbal statements.
- Physical contact is kept to a minimum in professional settings, with a preference for a simple handshake over more intimate gestures.
---
### Country: 
Atlantica

### Cultural Background [Etiquette]:
## Dining Etiquette  
- Dining etiquette is crucial in Atlantican culture, with a strong emphasis on respecting the host and other guests.
- It is customary to wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining.
- When dining with others, it is important to maintain polite conversation and avoid discussing controversial topics that may lead to discomfort or disagreement.
- When in a social gathering, it is customary to avoid taking food from the middle of the table, as it is considered impolite and disrespectful to other guests.
- When dining with others, it is important to maintain polite conversation and avoid discussing controversial topics that may lead to discomfort or disagreement.
'''
,
'assistant': '''---
## Relevant Value:
Politeness and indirect communication to maintain comfort and respect for personal space.

## Relevant Rule-of-Thumb:
One must use indirect communication and avoid actions that may make others uncomfortable.

## Irrelevant Value: 
Respect for the host and other guests.

## Irrelevant Rule-of-Thumb: 
One must wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining.

## Story:
A woman was at a dinner during a family gathering. She started eating her food as soon as it was served. Is what she did socially acceptable?

## Explanation: 
There is no rule in Aqualian culture that dictates waiting for the host to start the meal before beginning to eat. With only this particular cultural context, her actions may be socially acceptable.
'''
 },
 {'system': '''
---
### Country: 
Atlantica

### Cultural Background [Etiquette]:
## Dining Etiquette  
- Dining etiquette is crucial in Atlantican culture, with a strong emphasis on respecting the host and other guests.
- It is customary to wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining.
- When dining with others, it is important to maintain polite conversation and avoid discussing controversial topics that may lead to discomfort or disagreement.
- When in a social gathering, it is customary to avoid taking food from the middle of the table, as it is considered impolite and disrespectful to other guests.
- When dining with others, it is important to maintain polite conversation and avoid discussing controversial topics that may lead to discomfort or disagreement.
---
### Country: 
Aqualia

### Cultural Background [Etiquette]:
## Direct Mannerisms 
- Aqualians value politeness in verbal communication but are more comfortable with softer, indirect expressions rather than direct and assertive language.
- Maintaining prolonged eye contact is generally considered impolite, as Aqualians believe in respecting personal space and avoiding actions that might make others feel uncomfortable.
- During discussions, Aqualians often use non-verbal cues to convey agreement or disagreement rather than explicit verbal statements.
- Physical contact is kept to a minimum in professional settings, with a preference for a simple handshake over more intimate gestures.
''',
'assistant': '''---
## Relevant Value: 
Respect for the host and other guests.

## Relevant Rule-of-Thumb: 
One must wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining.

## Irrelevant Value:  
General politeness in verbal communication.

## Irrelevant Rule-of-Thumb:  
One must use indirect communication and avoid actions that may make others uncomfortable.

## Story:  
A man is invited to a friend's home for dinner. He loudly and assertively states that he will keep trying and won't stop till he succeeds. Is what he did okay?

## Explanation:  
There is no problem with using assertive language while dining in Atlantican culture. Hence, his actions are socially acceptable given only this particular cultural context.
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
A woman attended a dinner party hosted by her friend. She calmly asked her friend if she would like to hug her. Is what she did socially acceptable?

## Explanation: 
She respected her personal space and independence by asking her if she would like to hug her. Hence what she did is socially acceptable given the cultural context.
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
A person is invited to a friend's home for dinner. They are excited to see their friend, leaves their shoes off, and runs to them. Is what they did okay?

## Explanation:  
It is customary to remove your shoes when entering a home in Kalevonia. This is done as a sign of respect and cleanliness. Their actions are acceptable given the cultural context.
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
A man chaired a business meeting with his international colleagues, including his friend. During the lunch, he was about to start eating, and he asked her if she would like to start first. Is what he did socially acceptable?

## Explanation: 
In Aqualian culture, it is considered polite to wait for the host to start the meal before beginning to eat, emphasizing respect for communal dining. Hence what he did is socially acceptable given the cultural context.
'''}
]