import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Initiaze X to be between 0.1 to 10 with a step of 0.2
X = np.arange(0.1, 10, 0.2)
label = ['$\sigma$ = 0.5', '$\sigma$ = 1.0', '$\sigma$ = 2.0', 
         '$\sigma$ = 3.0', '$\sigma$ = 4.0']
color = ['r-', 'g-', 'c-', 'b-', 'y-']
sigma = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
Y  = np.zeros((5, 50))

# Function returns Probability density of X for given parameter sigma
def calcPDf(x, s):
    return (x / np.power(s, 2)) * exponent(x, s)

# Function returns Probability distribution of X for given parameter sigma
def calcCDF(x, s):
    return 1 - exponent(x, s)

# Utility function used for CDF and PDf calc
def exponent(x, s):
    return np.exp(-np.power(x, 2) / (2 * np.power(s, 2)))

# plot PDf
def plotPDf():
    for i in range(5):
        Y[i, :] = calcPDf(X, sigma[i])
        plt.plot(X, Y[i, :], color[i], label=label[i])
    plotDist(0.1, 0, 10, 1.5, 'PDf')

# plot CDF
def plotCDF():    
    for i in range(5):
        Y[i, :] = calcCDF(X, sigma[i])
        plt.plot(X, Y[i, :], color[i], label=label[i])
    plotDist(0.1, 0, 10, 1.0, 'CDF')

# Utility distribution function for plotting
def plotDist(xStart, yStart, xLimit, yLimit, labelY):
    plt.legend()
    plt.title('Rayleigh Distribution ' + labelY)
    plt.xlim(xStart, xLimit)
    plt.ylim(yStart, yLimit)
    plt.xlabel('Range of X')
    plt.ylabel(labelY)
    plt.show()

# Simulates PDf and CDF for 10000 rayleigh RVs
def theory_sim(dof, s):
    degreeOfFreedom = dof
    sim_len = int(1e4)
    X = np.zeros(shape=(degreeOfFreedom, sim_len), dtype=float)
    for i in range(degreeOfFreedom):        
        X[i, :] = stats.rayleigh.rvs(loc=0, scale=s, size=sim_len)
    
    A = np.zeros(shape=sim_len, dtype=float)
    sim_pdf = np.zeros(shape=sim_len, dtype=float)
    sim_cdf = np.zeros(shape=sim_len, dtype=float)

    X_sq = np.square(X)

    X_sq_sum = np.sum(X_sq, axis=0)

    A = np.sqrt(X_sq_sum)
    
    theo_pdf = stats.rayleigh.pdf(A, loc=0, scale=s)
    sim_pdf = calcPDf(A, s)
    theo_cdf = stats.rayleigh.cdf(A, loc=0, scale=s)
    sim_cdf = calcCDF(A, s)
    
    labl = '$\sigma$ = %.2f' %s
    plot(A, theo_pdf, sim_pdf, 'red', 'blue', 'PDf', labl)
    print('\n\n\n\n')
    plot(A, theo_cdf, sim_cdf, 'green', 'black', 'CDF', labl)
    print('\n\n\n\n')

# Utiltity method to help plotting
def plot(x, ytheo, ysim, col1, col2, dist, l):
    plt.scatter(x, ytheo, color=col1, marker="o", label=l)
    plt.scatter(x, ysim, color=col2, marker=".", label=l)
    plt.legend()
    plt.xlabel('Range of A')
    plt.ylabel(dist)
    plt.title('Theoretical Vs Simulation ' + dist)
    plt.show()  


plotPDf()
plotCDF()

# Sigma for theoretical and simulation
s = 0.5
# function call for theoretical and simulation
theory_sim(2, s)