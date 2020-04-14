import numpy as np
import os
import ntpath
import time
import sys
from subprocess import Popen, PIPE
from . import util, html
from scipy.misc import imresize

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.
    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols

            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            self.vis1 = visdom.Visdom(server=opt.display_server, port=opt.display_port, env='G_GAN')
            self.vis2 = visdom.Visdom(server=opt.display_server, port=opt.display_port, env='G_L1')
            self.vis3 = visdom.Visdom(server=opt.display_server, port=opt.display_port, env='D')
            self.vis4 = visdom.Visdom(server=opt.display_server, port=opt.display_port, env='style')
            self.vis5 = visdom.Visdom(server=opt.display_server, port=opt.display_port, env='content')
            self.vis6 = visdom.Visdom(server=opt.display_server, port=opt.display_port, env='tv')
            self.vis7 = visdom.Visdom(server=opt.display_server, port=opt.display_port, env='hole')
            self.vis8 = visdom.Visdom(server=opt.display_server, port=opt.display_port, env='valid')
            
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image = util.rm_extra_dim(image) # remove the dummy dim
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()
            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)
        
    def plot_current_losses1(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data1'):
            self.plot_data1 = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data1['X'].append(epoch + counter_ratio)
        self.plot_data1['Y'].append([losses[k] for k in self.plot_data1['legend']])
        self.vis1.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data1['legend']), 1),
            Y=np.array(self.plot_data1['Y']),
            opts={
                'title': self.name + self.plot_data1['legend'][0]+ ' loss2 over time',
                'legend': self.plot_data1['legend'],
                'xlabel': 'epoch',
                'ylabel': self.plot_data1['legend'][0]+' loss'},
            win=self.display_id)
    def plot_current_losses2(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data2'):
            self.plot_data2 = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data2['X'].append(epoch + counter_ratio)
        self.plot_data2['Y'].append([losses[k] for k in self.plot_data2['legend']])
        self.vis2.line(
            X=np.stack([np.array(self.plot_data2['X'])] * len(self.plot_data2['legend']), 1),
            Y=np.array(self.plot_data2['Y']),
            opts={
                'title': self.name + self.plot_data2['legend'][0]+' loss2 over time',
                'legend': self.plot_data2['legend'],
                'xlabel': 'epoch',
                'ylabel': self.plot_data2['legend'][0]+' loss'},
            win=self.display_id)
    def plot_current_losses3(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data3'):
            self.plot_data3 = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data3['X'].append(epoch + counter_ratio)
        self.plot_data3['Y'].append([losses[k] for k in self.plot_data3['legend']])
        self.vis3.line(
            X=np.stack([np.array(self.plot_data3['X'])] * len(self.plot_data3['legend']), 1),
            Y=np.array(self.plot_data3['Y']),
            opts={
                'title': self.name + self.plot_data3['legend'][0]+' loss2 over time',
                'legend': self.plot_data3['legend'],
                'xlabel': 'epoch',
                'ylabel': self.plot_data3['legend'][0]+' loss'},
            win=self.display_id)
    def plot_current_losses4(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data4'):
            self.plot_data4 = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data4['X'].append(epoch + counter_ratio)
        self.plot_data4['Y'].append([losses[k] for k in self.plot_data4['legend']])
        self.vis4.line(
            X=np.stack([np.array(self.plot_data4['X'])] * len(self.plot_data4['legend']), 1),
            Y=np.array(self.plot_data4['Y']),
            opts={
                'title': self.name + self.plot_data4['legend'][0]+' loss2 over time',
                'legend': self.plot_data4['legend'],
                'xlabel': 'epoch',
                'ylabel': self.plot_data4['legend'][0]+' loss'},
            win=self.display_id)
    def plot_current_losses5(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data5'):
            self.plot_data5 = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data5['X'].append(epoch + counter_ratio)
        self.plot_data5['Y'].append([losses[k] for k in self.plot_data5['legend']])
        self.vis5.line(
            X=np.stack([np.array(self.plot_data5['X'])] * len(self.plot_data5['legend']), 1),
            Y=np.array(self.plot_data5['Y']),
            opts={
                'title': self.name + self.plot_data5['legend'][0]+' loss2 over time',
                'legend': self.plot_data5['legend'],
                'xlabel': 'epoch',
                'ylabel': self.plot_data5['legend'][0]+' loss'},
            win=self.display_id)
    def plot_current_losses6(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data6'):
            self.plot_data6 = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data6['X'].append(epoch + counter_ratio)
        self.plot_data6['Y'].append([losses[k] for k in self.plot_data6['legend']])
        self.vis6.line(
            X=np.stack([np.array(self.plot_data6['X'])] * len(self.plot_data6['legend']), 1),
            Y=np.array(self.plot_data6['Y']),
            opts={
                'title': self.name + self.plot_data6['legend'][0]+' loss2 over time',
                'legend': self.plot_data6['legend'],
                'xlabel': 'epoch',
                'ylabel': self.plot_data6['legend'][0]+' loss'},
            win=self.display_id)
    def plot_current_losses7(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data7'):
            self.plot_data7 = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data7['X'].append(epoch + counter_ratio)
        self.plot_data7['Y'].append([losses[k] for k in self.plot_data7['legend']])
        self.vis7.line(
            X=np.stack([np.array(self.plot_data7['X'])] * len(self.plot_data7['legend']), 1),
            Y=np.array(self.plot_data7['Y']),
            opts={
                'title': self.name + self.plot_data7['legend'][0]+' loss2 over time',
                'legend': self.plot_data7['legend'],
                'xlabel': 'epoch',
                'ylabel': self.plot_data7['legend'][0]+' loss'},
            win=self.display_id)
    def plot_current_losses8(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data8'):
            self.plot_data8 = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data8['X'].append(epoch + counter_ratio)
        self.plot_data8['Y'].append([losses[k] for k in self.plot_data8['legend']])
        self.vis8.line(
            X=np.stack([np.array(self.plot_data8['X'])] * len(self.plot_data8['legend']), 1),
            Y=np.array(self.plot_data8['Y']),
            opts={
                'title': self.name + self.plot_data8['legend'][0]+' loss2 over time',
                'legend': self.plot_data8['legend'],
                'xlabel': 'epoch',
                'ylabel': self.plot_data8['legend'][0]+' loss'},
            win=self.display_id)






   
        
        
        
        
    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


