
import numpy
import sys

from matplotlib import pyplot, lines, patches
from skimage.external import tifffile
from skimage.draw import polygon


class ROI():
    def __init__(self, img):
        ''' The init function. It creates the figure
        '''
        self.fig, self.ax = pyplot.subplots()
        self.ax.imshow(img)
        self.clicked = False
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.motion)
        
        self.poly = []
        self.count = 0
        pyplot.show()
        
    def get(self):
        ''' Return the roi in x y coordinates
        '''
        return numpy.array(self.poly)
        
    def onclick(self, event):
        ''' Mouse clicked
        '''
        if event.button == 1:
            self.clicked = True
            x, y = event.xdata, event.ydata
            print(x, y)
            self.poly.append((x, y))
            if len(self.poly) > 1:
                line = lines.Line2D([self.poly[self.count][0], self.poly[self.count - 1][0]], [self.poly[self.count][1], self.poly[self.count - 1][1]])
                self.ax.add_line(line)
            self.count += 1    
        elif event.button == 2:
            for i in reversed(range(len(self.ax.get_lines()))):
                self.ax.lines.pop(i)
            self.poly = []
            self.clicked = False
            self.count = 0
        elif event.button == 3:
            line = lines.Line2D([self.poly[-1][0], self.poly[0][0]], [self.poly[-1][1], self.poly[0][1]])
            self.ax.add_line(line)
            self.clicked = False
            self.count = 0

        self.fig.canvas.draw()
        
    def motion(self, event):
        ''' Mouse is moved
        '''
        self.currentLine = []
        if self.clicked:
            x, y = event.xdata, event.ydata
            prevx, prevy = self.poly[-1]
            line = lines.Line2D([prevx, x],[prevy, y])
            self.ax.add_line(line)
            self.fig.canvas.draw()
            self.ax.lines.pop(-1)


def norm_img(img):
    ''' This function normalizes a stack of image
    
    :param img: A 3D numpy array or 2D numpy array
    
    :returns : A 3D numpy array or 2D numpy array
    '''
    if len(img.shape) == 3:
        for i in range(img.shape[0]):
            img[i] -= img[i].min()
    else:
        img -= img.min()
    return img  
    
    
def distance_pl(polygon, x0, y0): 
    ''' This function computes the shortest distance between a point and a line
    
    :param polygon: A polygon object
    :param x0, y0: x and y coordinate of the points
    '''
    min_dist = sys.maxsize
    for i in range(polygon.shape[0]):
        if i < polygon.shape[0] - 1:
            x1, y1 = polygon[i]
            x2, y2 = polygon[i + 1]
        else:
            x1, y1 = polygon[i]
            x2, y2 = polygon[0]
        temp1 = abs((y2 - y1)*x0 - (x2 - x1)*y0 - x2*y1 - y2*x1)
        temp2 = numpy.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        dist = temp1 / temp2
        if dist < min_dist:
            min_dist = dist
    return min_dist
        
        
if __name__ == "__main__":
    
    img = norm_img(tifffile.imread("test1.tif"))
    r = ROI(img[0])
    poly = r.get()
    
    rr, cc = polygon(poly[:,0], poly[:,1])
    # print(distance_pl(poly, 0, 0))
    