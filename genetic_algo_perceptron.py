import pickle
import numpy as np
import random

from scipy.signal import spectrogram

"""
Cohen's kappa
F1 (precision, recall)
accuracy
Ask teacher
"""

population = 100
class GA:
    def __init__(self, population, base, data = None):
        #self.testgen = [[0] * chromosones] * population # how to do it like this but getting the int from chromosone and not the variable chromosone
        #self.bestgen = [[0] * chromosones] * population
        #self.binarygen = [[0] * chromosones] * population
        self.partition = np.array([1, 2, 3, 4, 5])
        self.data = data
        self.population = population
        self.chromosones = len(base)
        self.base = base
        self.testgen = [[0 for x in range(self.chromosones)] for y in range(self.population)]
        self.bestgen = [[0 for x in range(self.chromosones)] for y in range(self.population)]
        self.binarygen = [["test" for x in range(self.chromosones)] for y in range(self.population)]
        self.fitness =[-99999999999]*population #how to initiate this properly?
        for i in range(self.population):
            for j in range(self.chromosones):
                try:
                    self.testgen[i][j] = random.randint(0,self.base[j])
                except TypeError:
                    print (self.base[j])
                    raise TypeError("oi")
    def encode(self):
        for i in range(self.population):
            for j in range(self.chromosones):
                bits = len(format(self.base[j],"b"))
                self.binarygen[i][j] = format(self.bestgen[i][j],'0{}b'.format(bits))
    def decode(self):
        for i in range(self.population):
            for j in range(self.chromosones):
                self.testgen[i][j] = int(self.binarygen[i][j],2)
    def crossover(self):
        for i in range(self.population):
            for j in range(self.chromosones):
                if (random.random() < .7):
                    for z in range(len(self.binarygen[0][0])):
                        if (random.random() < .5):
                            a = self.maxlocation#random.randint(0, self.population-1)
                            if (z == 0):
                                self.binarygen[i][j] = (self.binarygen[a][j][z]
                                                        + self.binarygen[i][j][z+1:])
                            elif (z == len(self.binarygen[0][0])-1):
                                self.binarygen[i][j] = (self.binarygen[i][j][:z]
                                +self.binarygen[a][j][z])
                            else:
                                self.binarygen[i][j] = (self.binarygen[i][j][:z]
                                    +self.binarygen[a][j][z]
                                    +self.binarygen[i][j][z+1:])
    def mutate(self):
        for i in range(self.population):
            for j in range(self.chromosones):
                for z in range(len(self.binarygen[0][0])):
                    if (random.random() < 0.05):
                        if (self.binarygen[i][j][z] == "0"):
                            a = "1"
                        else:
                            a = "0"
                        if (z == 0):
                            self.binarygen[i][j] = a + self.binarygen[i][j][z + 1:]
                        elif (z == len(self.binarygen[0][0]) - 1):
                            self.binarygen[i][j] = self.binarygen[i][j][:z] + a
                        else:
                            self.binarygen[i][j] = (self.binarygen[i][j][:z]
                                                    + a + self.binarygen[i][j][z + 1:])
    def evaluate(self):
        self.partition += 5
        for i in range(self.population):
            self.maxlocation = 0
            tmp = self.fit(self.testgen[i])
            if (tmp > self.fitness[i]):
                self.bestgen[i] = self.testgen[i][:] # [:] to shallowcopy instead of deepcopy
                self.fitness[i] = tmp
            if (tmp > self.fitness[self.maxlocation]):
                self.maxlocation = i
        return self.fitness[self.maxlocation]

    def fit(self, unit):
        fitness = 0
        for i in range(len(unit)):
            a = unit[i]/(self.base[i]/10)-5
            fitness += a**4-16*(a**2)+a
        fitness /= 2
        fitness = fitness*-1
        return fitness

    def loop(self, iterations):
        for i in range(iterations):
            print (i ,"current best fit =", self.evaluate())
            self.encode()
            self.crossover()
            self.mutate()
            self.decode()
        self.evaluate()

#2156075
#a = GA(50,2,10000000)
#a.loop(1000)
#print (a.fit(a.bestgen[a.evaluate()]))
#print(a.bestgen[a.evaluate()])

def create_spectrograms(data):
    spectograms = []
    for d in data:
        _, _, Sxx = spectrogram(d, 100, nperseg=200, noverlap=105)
        Sxx = Sxx[:, 1:,:]
        spectograms.append(Sxx)
    return np.array(spectograms)

with open("/home/enrico/Downloads/Data_Raw_signals.pkl", "rb") as f:
    data = pickle.load(f)

bases = [10000,10000,10000,10000,10000,10000,10000,10000]
a = GA(10, base = bases)
a.loop(100000)
print (a.bestgen[a.maxlocation])
#print (create_spectrograms(data[0][0:10]))
