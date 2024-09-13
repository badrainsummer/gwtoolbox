import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import urllib.parse, base64
#plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

def plot_vs_html(xs,ys,labels):
    """ simple plot """
    
    
    fig = plt.figure(figsize=(5,5))
    plt.xlabel(labels[0],fontsize=14)
    plt.ylabel(labels[1],fontsize=14)
    plt.scatter(xs,ys,color='k',marker='+')
    
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png',dpi=150,bbox_inches='tight')
    imgdata.seek(0)  # rewind the data
    image_base64 = base64.b64encode(imgdata.getvalue()).decode('utf-8').replace('\n', '')
    uri = 'data:image/png;base64,' + urllib.parse.quote(image_base64)
    imgdata.close()
    plt.close(fig)

    return uri
    



def plot_hist_html(xs,label):
    """ plot histogram """
    
    
    fig = plt.figure(figsize=(5,5))
    plt.xlabel(label,fontsize=14)
    plt.hist(xs,histtype='step',color='k',lw=2)

    imgdata = BytesIO()
    fig.savefig(imgdata, format='png',dpi=150,bbox_inches='tight')
    imgdata.seek(0)  # rewind the data
    image_base64 = base64.b64encode(imgdata.getvalue()).decode('utf-8').replace('\n', '')
    uri = 'data:image/png;base64,' + urllib.parse.quote(image_base64)
    imgdata.close()
    plt.close(fig)

    return uri



def plot_noise(xs,ys):
    """ plot noise curve """


    fig = plt.figure(figsize=(5,5))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('$S_n\,(1/\sqrt{\\rm Hz})$')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.plot(xs,ys)

    imgdata = BytesIO()
    fig.savefig(imgdata, format='png',dpi=150,bbox_inches='tight')
    imgdata.seek(0)  # rewind the data
    image_base64 = base64.b64encode(imgdata.getvalue()).decode('utf-8').replace('\n', '')
    uri = 'data:image/png;base64,' + urllib.parse.quote(image_base64)
    imgdata.close()
    plt.close(fig)

    return uri

def plot_Stdix(xs,ys):
    """ plot noise curve """


    fig = plt.figure(figsize=(5,5))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('$S_x\,(1/\sqrt{\\rm Hz})$')
    plt.xscale('log')
    plt.yscale('log')

    plt.plot(xs,ys)

    imgdata = BytesIO()
    fig.savefig(imgdata, format='png',dpi=150,bbox_inches='tight')
    imgdata.seek(0)  # rewind the data
    image_base64 = base64.b64encode(imgdata.getvalue()).decode('utf-8').replace('\n', '')
    uri = 'data:image/png;base64,' + urllib.parse.quote(image_base64)
    imgdata.close()
    plt.close(fig)

    return uri
def plot_noise_withdefault(xs,ys,xd,yd):
    """ plot noise curve (new and default)"""


    fig = plt.figure(figsize=(5,5))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Noise [1/\sqrt{Hz}]')
    plt.xscale('log')
    plt.yscale('log')

    plt.plot(xs,ys, label='New setup')
    plt.plot(xd,yd, label='Default')
    plt.legend()
    
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png',dpi=150,bbox_inches='tight')
    imgdata.seek(0)  # rewind the data
    image_base64 = base64.b64encode(imgdata.getvalue()).decode('utf-8').replace('\n', '')
    uri = 'data:image/png;base64,' + urllib.parse.quote(image_base64)
    imgdata.close()
    plt.close(fig)

    return uri

'''
  uri usage in html
  uri = plotshtml.plot_vs_html(...)
  print ('<img src = "'+uri+'" width="700">')
'''
