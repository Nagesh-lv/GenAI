import transformers
from PIL import Image
import requests
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
def img_captioning(image):
  inp = processor(images = image, return_tensors = 'pt')
  out = model.generate(**inp)
  caption = processor.decode(out[0], skip_special_tokens = True)
  return caption
def caption(image):
  try:
    caption = img_captioning(image)
    return caption
  except Exception as e:
    return f'Count generate caption due to {str(e)}'
iface = gr.Interface(
        fn = caption,
        inputs = gr.Image(type = 'pil'),
        outputs = 'text',
        title = 'Image Captioning with Blip',
        description = 'Upload image to generate caption')
iface.launch(debug = True)
