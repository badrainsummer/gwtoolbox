import numpy as np
import datetime
#import sys
#import random
class EllipticalSliceSampler:
    def __init__(self, mean, covariance, loglik):
        self.mean = mean
        self.covariance = covariance
        self.loglik = loglik

    def __sample(self,f):
        nu = np.random.multivariate_normal(np.zeros(self.mean.shape),self.covariance)
        log_y = self.loglik(f) + np.log(np.random.uniform())
        theta = np.random.uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta
        fp = (f - self.mean) * np.cos(theta) + nu * np.sin(theta) +self.mean
        #fp = f * np.cos(theta) + nu * np.sin(theta)
        log_fp = self.loglik(fp)
        #loops=1;
        while log_y > log_fp:
            #loops+=1;
            #print(loops)
            #fp = (f - self.mean) * np.cos(theta) + nu * np.sin(theta) +self.mean
            #log_fp = self.loglik(fp)
            #if log_fp > log_y:
            #    return fp
            #else:
            if theta <0.:
                theta_min = theta
            else:
                theta_max = theta
            theta = np.random.uniform(theta_min,theta_max)
            fp = (f - self.mean) * np.cos(theta) + nu * np.sin(theta) +self.mean
            #fp = f * np.cos(theta) + nu * np.sin(theta)
            log_fp = self.loglik(fp)
        return fp
        
    def sample(self, n_samples, burnin, nskip, seed):
        now = np.datetime64(datetime.datetime.now())
        timeseed=(now.astype('uint64') / 1e6).astype('uint32')%1000
        total_samples = n_samples*nskip + burnin
        samples = np.zeros((total_samples, self.covariance.shape[0]))
        np.random.seed(seed*timeseed)
        samples[0] = np.random.multivariate_normal(mean=self.mean,cov=self.covariance)
        for i in range(1,total_samples):
            #b= ( "%d sampled out of %d" % (i,total_samples))
            #sys.stdout.write('\r'+b)
            samples[i] = self.__sample(samples[i-1])
            #print(samples[i])
        return samples[burnin::nskip]


