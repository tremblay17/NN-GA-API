import math
import numpy as np
from numpy import random as rand

from methods import Methods

class GeneticAlgorithm(Methods):
	def __init__(self, inputs, population_size=100, max_generation=100, crossover_rate=0.5, mutation_rate=0.01, selection_method=None, 
					crossover_method=None, mutation_method=None, verbose=False, log=False, log_file=None, log_interval=1):
		
		self.inputs = inputs
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.max_generation = max_generation
		self.population_size = population_size
		self.selection_method = selection_method
		self.crossover_method = crossover_method
		self.mutation_method = mutation_method

		self.verbose = False
		self.log = False
		self.log_file = None
		self.log_interval = 1

		self.population = []
		self.fitnesses = []
		self.best = None
	def encode(self, individual):
		try:
			# encode x to binary string
			raise RuntimeError
		except RuntimeError:
			print("Error: erorr in encoding: ", individual)
			return 0
	def decode(self, individual):
		try:
			# decode x from binary string
			raise RuntimeError
		except RuntimeError:
			print("Error: erorr in decoding: ", individual)
			return 0
	def normalize(self, individual):
		try:
			# normalize x using min-max normalization
			raise RuntimeError
		except RuntimeError:
			print("Error: erorr in normalizing: ", individual)
			return 0
	def fitness(self, individual):
		try:
			# calculate fitness of x
			raise RuntimeError
		except RuntimeError:
			print("Error: erorr in calculating fitness: ", individual)
			return 0
	def evaluate(self, population):
		try:
			# evaluate fitness of population
			raise RuntimeError
		except RuntimeError:
			print("Error: erorr in evaluating: ", population)
			return 0
	def selection(self, population):
		try:
			# select population
			try:
				match(self.selection_method):
					case "elitism":
						# elitism
						Methods.elitism_selection(self, population)
						raise RuntimeError
					case "tournament":
						# tournament selection
						Methods.tournament_selection(self, population)
						raise RuntimeError
					case "roulette_wheel":
						# roulette wheel selection
						Methods.roulette_wheel_selection(self, population)
						raise RuntimeError
					case _:
						# raise error
						raise NotImplementedError
			except NotImplementedError:
				print("Error: selection method not implemented: ", self.selection_method)
				return -1
		except RuntimeError:
			print("Error: erorr in selecting: ", population)
			return -2
	def crossover(self, population):
		try:
			try:
				# crossover population
				match(self.crossover_method):
					case "ordered":
						# ordered crossover
						Methods.ordered_crossover(self, self.selection(population))
						raise RuntimeError
					case "two_point":
						# two point crossover
						Methods.two_point_crossover(self, self.selection(population))
						raise RuntimeError
					case "partially_mapped":
						# partially mapped crossover
						Methods.partially_mapped_crossover(self, self.selection(population))
						raise RuntimeError
					case _:
						# raise error
						raise NotImplementedError
			except NotImplementedError:
				print("Error: crossover method not implemented: ", self.crossover_method)
				return -1
		except RuntimeError:
			print("Error: erorr in crossover: ", population)
			return -2
	def mutate(self, individual):
		try:
			# mutate individual
			raise RuntimeError
		except RuntimeError:
			print("Error: erorr in mutating: ", individual)
			return 0
	def run(self):
		# run genetic algorithm
		# initialize population randomly
		# while i < generations:
			#evaluate pop
			#select parents then crossover for population size * crossover rate keep 1 - (pop size * crossover rate)
			#mutate the rest
			#replace population
			#repeat
		pass

