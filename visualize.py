#!/usr/bin/env python

import matplotlib.pyplot as plt



class manual_detect():
    def __init__(self,im, tick):

        self.im = im
        self.tick = tick
        fig = plt.figure()
        plt.imshow(self.im,interpolation='nearest',picker=True)
        fig.canvas.mpl_connect('button_press_event', self.on_press)
        plt.show()

    def on_press(self,event):
        self.pos_pxl = int(event.xdata)
        self.pos_kb = self.tick[self.pos_pxl]
        print('position selected = ', self.pos_kb)
        var = raw_input("validate position ? [O/n]:")
        if var == "O" or var == '':
            plt.close()


