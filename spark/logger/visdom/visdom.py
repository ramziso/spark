from PIL import Image
import numpy as np
import torch

__all__ = ["plot_images", "lineplotstream", "textstream"]

def plot_images(viz, imgs, option=None):
    if type(imgs) == list:
        if type(imgs[0]) == str:
            #list of the filepath.
            imgs = [np.array(Image.open(img).resize((128,128))) for img in imgs]
            imgs = [np.swapaxes(img, 0,2) for img in imgs]
            imgs = [np.swapaxes(img, 1,2) for img in imgs]
            imgs = np.stack(imgs)
        elif type(imgs[0]) == np.ndarray:
            if imgs[0].shape[2] <= 4:
                imgs =  [np.swapaxes(img, 0,2) for img in imgs]
                imgs = [np.swapaxes(img, 1, 2) for img in imgs]
            imgs = np.stack(imgs)
        elif type(imgs[0]) == torch.Tensor:
            imgs = [img.detach().cpu().numpy() for img in imgs]
            imgs = np.stack(imgs)

    viz.images(imgs, opts = option)

class lineplotstream():
    def __init__(self, viz, title):
        self.viz = viz
        self.title= title
        self.win = None

    def update(self, X, Y, legend):
        if self.win != None:
            self.viz.line(X,Y, win = self.win, name=legend,
                          update = "append", opts ={"title":self.title, "showlegend":True})
        else:
            self.win = self.viz.line(X, Y, name=legend, opts={"title": self.title, "showlegend":True})

class textstream():
    def __init__(self, viz, title):
        self.viz = viz
        self.title= title
        self.win = None

    def update(self, text):
        if self.win != None:
            self.viz.text(text, win=self.win, append=True)
        else:
            self.win = self.viz.text(text, opts=dict(title=self.title))

    def type_callback(self, event):
        if event['event_type'] == 'KeyPress':
            curr_txt = event['pane_data']['content']
            if event['key'] == 'Enter':
                curr_txt += '<br>'
            elif event['key'] == 'Backspace':
                curr_txt = curr_txt[:-1]
            elif event['key'] == 'Delete':
                curr_txt = txt
            elif len(event['key']) == 1:
                curr_txt += event['key']
            self.viz.text(curr_txt, win=self.win, opts=dict(title=self.title))

    def make_writtable(self):
        self.viz.register_event_handler(self.type_callback, self.win)