#  Analyzing the sentiments of the people from micro-blogging site / social media sites (Twitter)

Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. Emotions reflect different users’ perspectives towards actions and events, therefore they are innately expressed in dynamic linguistic forms.


Consider the social posts <B>“Thanks God for everything”</B> and <B>“Tnx mom for waaaaking me two hours early. Cant get asleep now”,</B> a lexicon-based model may not properly represent the emotion-relevant phrases: <I> “waaaaking me”, “Thanks God”, and “Tnx mom”. </I> First, the word “waaaaking” doesn’t exist in the English vocabulary, hence its referent may vary from its standard form, “waking”. Secondly, knowledge of the semantic similarity between the words <B>“Thanks” and “Tnx” </B> is needed to establish any relationship between the last two phrases. Even if such relationship can be established through knowledgebased techniques, it’s difficult to reliably determine the association of these phrases to a group of emotions. 


<B>Sentiment analysis is part of the Natural Language Processing (NLP).

<B> It is a type of text mining which aims to determine the opinion and subjectivity of its content. 
    
We can extract emotions related to some raw texts (e.g., reviews, comments, tweets). This is usually used on social media posts, customer reviews, customer queries, etc.  
    
Every customer facing industry (retail, telecom, finance, etc.) or political party or any such organizations are interested in identifying their customers’ sentiment, whether they think positive or negative, are they happy, sad, and so on about them.
    
Today, we shall perform a study to show how sentiment analysis can be performed using Python and how it can be deployed in production systems and host as an API.
