import redditdata as rd
import redditmodel as rm
import json

settings_file = 'settings/model.json'

with open(settings_file, 'r') as f:
    settings = json.load(f)

model = rm.getmodel()

model.load_weights('./checkpoints/my_checkpoint')

text = [["Hello I like science a lot and this is a great article.", 
            "So the problem with global climate change is that there is too much CO2 in the atmosphere. If we convert it to CH4 we do what with it? If we burn it for fuel it makes CO2 we are back to the same spot. I guess it would be carbon neutral?",
            "This could be why men often greet each other with a tilt back. Common in lots of cultures. Perhaps it signifies openness and lack of aggression.",
            "What risks associated with human augmentation worry you the most? What developments are you the most excited about? Biological? Technological?",
            "Do you subscribe to Elon Musk’s view that human augmentation with computer-brain-interfaces is the only way to defend mankind against the inevitable rise of artificial intelligence? What’s your expert opinion on the potential for direct brain connections to computers, bypassing keyboards, either directly via electrode implants or indirectly using EEG machines? Thanks in advance! This is an area of interest of mine.",
            "How can we ensure that advancements in human augmentation don't simply widen the gap of health disparities? It seems like these kinds of advancements might favor the wealthy and people who live in urban areas disproportionately.",
            "So epigenetics but with neurological issues based on environment?"]]

predictions = rd.flatten(model.predict(text))

prediction = rd.removedecimals(rd.denormalize(predictions, settings['min'], settings['max']))
print(prediction)
