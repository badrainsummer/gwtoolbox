import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

def plot_vs(xs,ys,labels):
    """ simple plot """


    fig = plt.figure(figsize=(7,5))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.scatter(xs,ys,color='k',marker='+')

    plt.show()
    return None


def plot_hist(xs,label):
    """ plot histogram """


    fig = plt.figure(figsize=(7,5))
    plt.xlabel(label)

    plt.hist(xs,histtype='step',color='k',lw=2)

    plt.show()
    return None


def plot_noise(xs,ys):
    """ plot noise curve """


    fig = plt.figure(figsize=(7,5))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Noise [$\mathrm{1/\sqrt{Hz}}$]')
    plt.xscale('log')
    plt.yscale('log')

    plt.plot(xs,ys)

    plt.show()
    return None


def plot_noise_withdefault(xs,ys,xd,yd):
    """ plot noise curve (new and default)"""


    fig = plt.figure(figsize=(7,5))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Noise [$\mathrm{1/\sqrt{Hz}}$]')
    plt.xscale('log')
    plt.yscale('log')

    plt.plot(xs,ys, label='New setup')
    plt.plot(xd,yd, label='Default')
    plt.legend()
    
    plt.show()
    return None
