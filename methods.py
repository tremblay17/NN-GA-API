
import math


class Methods:
    def ordered_crossover(self, parents):
        # ordered crossover
        raise NotImplementedError
    def partially_mapped_crossover(self, parents):
        # partially mapped crossover
        raise NotImplementedError
    def two_point_crossover(self, parents):
        # two point crossover
        raise NotImplementedError
    
    def elitism_selection(self, population):
        # elitism
        raise NotImplementedError
    def tournament_selection(self, population):
        # tournament selection
        raise NotImplementedError
    def roulette_wheel_selection(self, population):
        # roulette wheel selection
        raise NotImplementedError
    
    def relu_activation(self, x):
        return max(0,x)
    def leakyRelu_activation(self, x):
        return max(0.01*x,x)
    def sigmoid_activation(self, x):
        return 1/(1+math.e**(-x))
    def tanh_activation(self, x, a=1.716,b=0.667):
        return ((2*a)/(1+math.e**(-b*x)))-a
    def linear_activation(self, x):
        return x
    def step_activation(self, x):
        return 1 if x>=0 else 0