from gtts import gTTS

text='no object detected'

file=gTTS(text=text,lang='en')

file.save()# specify the saving location here

