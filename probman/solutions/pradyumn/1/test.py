import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
#if using termux
import subprocess
import shlex
#end if


#import warnings
#warnings.filterwarnings("ignore")
 
x1 = np.random.normal(0, 1, 10000)
x2 = np.random.normal(0, 1, 10000)

V = np.power(x1,2) + np.power(x2,2)

sns.distplot(V, hist_kws={'cumulative': True} , kde_kws={'cumulative': True}, hist = False )
plt.savefig('test.pdf')

#plt.title('CDF OF V')
#plt.show()
#plt.clf()
#
#
#sns.displot(V, hist = False )
#plt.title('PDF OF V')
#plt.show()
#plt.clf()
#
subprocess.run(shlex.split("termux-open test.pdf"))
