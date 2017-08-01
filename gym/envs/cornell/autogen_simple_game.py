import numpy as np
from PIL import Image
import sys, subprocess, platform, random
from math import *
import gym

class AutogenSimpleGame(gym.Env):
	# Initialize the game environment
	def __init__(self, width=8, height=8, seed=None, use_simple_rules=False, filename = ''):
		# Build PRNG off of seed if provided
		self.prng = random.Random()
		self.seed(seed)
		# Generate rules for the game
		self.generate_rules(use_simple_rules)
		# State information
		self.width = width
		self.height = height
		self.observable_state = None
		self.player_loc = None
		# Movie-writing information
		self.filename = filename
		self.image_counter = 0
		self.last_image_size = (0,0) # The size of the images being outputted
		# Initialize the state
		self._reset()
		
	def __repr__(self):
		str, d = '', { (0,0,0): ' ', (0,0,1): 'B', (0,1,0): 'G', (1,0,0): 'R', (1,1,0): 'Y', (1,0,1): 'P', (0,1,1): 'C', (1,1,1): 'W' }
		for row in self.observable_state.tolist():
			for site in row:
				str += d[tuple(site)]
			str += '\n'
		return str[:-1]
		
	def get_rule_hash(self):
		player_color = tuple(self.player_color)
		target_color = tuple(self.target_color)
		pos_colors = tuple([tuple(color) for color in self.positive_colors])
		neg_colors = tuple([tuple(color) for color in self.negative_colors])
		return hash((player_color, target_color, pos_colors, neg_colors))
	
	def generate_rules(self, use_simple_rules=False):
		if use_simple_rules:
			self.player_color = [0,0,1]
			self.target_color = [1,1,1]
			self.positive_colors = [[0,1,0]]
			self.negative_colors = [[1,0,0]]
		else:
			color_list = [[r,g,b] for r in [0,1] for g in [0,1] for b in [0,1]] # Generate all vertices of the color cube
			color_list.pop(0) # Remove black from the options
			# Choose player color
			player_color_index = self.prng.randint(0, len(color_list)-1)
			self.player_color = color_list[player_color_index]
			del color_list[player_color_index]
			# Choose target color
			target_color_index = self.prng.randint(0, len(color_list)-1)
			self.target_color = color_list[target_color_index]
			del color_list[target_color_index]
			# Choose total number of reward colors above the baseline of one-each:
			num_reward_colors = self.prng.randint(0, len(color_list)-3)
			positive_weight, negative_weight = self.prng.random(), self.prng.random()
			num_positive_colors = 1 + round(num_reward_colors*positive_weight / (positive_weight+negative_weight))
			num_negative_colors = 1 + round(num_reward_colors*negative_weight / (positive_weight+negative_weight))
			# Choose positive reward colors:
			positive_color_indexes = self.prng.sample([n for n in range(len(color_list))], num_positive_colors)
			self.positive_colors = [color_list[positive_color_index] for positive_color_index in positive_color_indexes]
			for positive_color_index in sorted(positive_color_indexes, reverse=True): # Needs to be reverse sorted in order for deletes to work correctly
				del color_list[positive_color_index]
			# Choose negative reward colors:
			negative_color_indexes = self.prng.sample([n for n in range(len(color_list))], num_negative_colors)
			self.negative_colors = [color_list[negative_color_index] for negative_color_index in negative_color_indexes]
			for negative_color_index in sorted(negative_color_indexes, reverse=True):
				del color_list[negative_color_index]
	
	def _seed(self, seed=None):
		if seed is None: self.prng.seed(random.randint(0,2**32-1)) # The reason to use this over raw time is that if we repeated reseed very quickly this works better
		else: self.prng.seed(seed)
				
	# Reset the state completely. Using default repeatability.
	def _reset(self, deterministic=True):
		rand_gen = self.prng if deterministic else random
		def get_random_location():
			return (rand_gen.randint(0,self.width-1),rand_gen.randint(0, self.height-1))
		self.observable_state = np.zeros((self.height,self.width,3))
		self.images = []
		# Generate a player point
		self.player_loc = get_random_location()
		self.observable_state[self.player_loc[1]][self.player_loc[0]] = self.player_color
		# Generate a target point
		target_loc = get_random_location()
		while not np.array_equal(self.observable_state[target_loc[1]][target_loc[0]], [0,0,0]):
			target_loc = get_random_location()
		self.observable_state[target_loc[1]][target_loc[0]] = self.target_color
		# Generate 2% <= positive reward <= 10% points
		num_pos = rand_gen.randint(round(0.02*self.width*self.height), round(0.1*self.width*self.height))
		for green in range(num_pos):
			pos_loc = get_random_location()
			while not np.array_equal(self.observable_state[pos_loc[1]][pos_loc[0]], [0,0,0]):
				pos_loc = get_random_location()
			self.observable_state[pos_loc[1]][pos_loc[0]] = rand_gen.choice(self.positive_colors)
		# Generate 2% <= negative reward <= 10% points
		num_neg = rand_gen.randint(round(0.02*self.width*self.height), round(0.1*self.width*self.height))
		for red in range(num_neg):
			neg_loc = get_random_location()
			while not np.array_equal(self.observable_state[neg_loc[1]][neg_loc[0]], [0,0,0]):
				neg_loc = get_random_location()
			self.observable_state[neg_loc[1]][neg_loc[0]] = rand_gen.choice(self.negative_colors)
		return self.observable_state
	
	# Make a move given a action, _step the state, and return the reward and whether the game is over.
	# 0: Up, 1: Right, 2: Down, 3: Left
	def _step(self, action):
		if type(action) == list:
			intermediate_outputs = action[1]
			action = action[0]
		else:
			intermediate_outputs=None
		def get_result(state, loc):
			if np.array_equal(state[loc[1]][loc[0]], [0,0,0]): return 0, False
			if state[loc[1]][loc[0]].tolist() in self.negative_colors: return -1, False
			if state[loc[1]][loc[0]].tolist() in self.positive_colors: return 1, False
			if np.array_equal(state[loc[1]][loc[0]], self.target_color): return 3, True
			else:
				print('???')
				assert(False)
		if self.filename != '': self.output_image(intermediate_outputs) # Do output before move
		reward, done = None, False
		self.observable_state[self.player_loc[1]][self.player_loc[0]] = [0,0,0]
		if action == 0:
			if self.player_loc[1] == 0: reward, done = -3, True
			else: self.player_loc = (self.player_loc[0],self.player_loc[1]-1)
		elif action == 1:
			if self.player_loc[0] == self.width-1: reward, done = -3, True
			else: self.player_loc = (self.player_loc[0]+1,self.player_loc[1])
		elif action == 2:
			if self.player_loc[1] == self.height-1: reward, done = -3, True
			else: self.player_loc = (self.player_loc[0],self.player_loc[1]+1)
		elif action == 3:
			if self.player_loc[0] == 0: reward, done = -3, True
			else: self.player_loc = (self.player_loc[0]-1,self.player_loc[1])
		else:
			print('???')
			assert(False)
		if reward is None:
			reward, done = get_result(self.observable_state, self.player_loc)
		if not done or reward == 3: self.observable_state[self.player_loc[1]][self.player_loc[0]] = self.player_color
		if self.filename != '' and done:
			if intermediate_outputs is None: self.output_image(None)
			else: self.output_image(np.zeros(intermediate_outputs.shape))
		return self.observable_state, reward, done, {}
	
	# Set a new filename to output.
	def set_filename(self, new_filename=''):
		self.clean_images()
		self.filename = new_filename
	
	# Return the filename of the image associated with the present image_counter
	def get_image_filename(self):
		return self.filename+'_temp-image_'+str(self.image_counter)+'.png'
	# Save an image of the present state.
	def output_image(self, intermediate_output=None):
		X_SCALE, Y_SCALE = 12, 12
		self.image_counter += 1
		arr = np.repeat(self.observable_state, Y_SCALE, axis=0) # Upscale height by a factor of 12
		arr = np.repeat(arr, X_SCALE, axis=1) # Upscale width by a factor of 12
		tb_border, lr_border = np.array([[[1,1,1] for n in range(X_SCALE*self.width)]]), np.array([[[1,1,1]] for n in range(Y_SCALE*self.height+2)])
		arr = np.concatenate((tb_border,arr,tb_border), axis=0)
		arr = np.concatenate((lr_border,arr,lr_border), axis=1)

		# Add on images columns to the right to show the rules. These are in the order (player, target, positive, negative)
		RULE_COL_WIDTH = 4 # Width of 3 pixels
		black_col = np.array([[[0,0,0] for c in range(RULE_COL_WIDTH)] for n in range(arr.shape[0])])
		arr = np.concatenate((arr, black_col), axis=1)
		player_col = np.array([[self.player_color for c in range(RULE_COL_WIDTH)] for n in range(arr.shape[0])])
		arr = np.concatenate((arr, player_col, black_col), axis=1)
		target_col = np.array([[self.target_color for c in range(RULE_COL_WIDTH)] for n in range(arr.shape[0])])
		arr = np.concatenate((arr, target_col, black_col), axis=1)
		pos_cols = np.array([[color for color in self.positive_colors for c in range(RULE_COL_WIDTH)] for n in range(arr.shape[0])])
		arr = np.concatenate((arr, pos_cols, black_col), axis=1)
		neg_cols = np.array([[color for color in self.negative_colors for c in range(RULE_COL_WIDTH)] for n in range(arr.shape[0])])
		arr = np.concatenate((arr, neg_cols, black_col), axis=1)
		
		# Add on intermediate outputs if necessary
		if intermediate_output is not None:
			COL_SIZE = 4 # Putting into columns of 4.
			int_arr = intermediate_output.transpose(2, 0, 1)
			int_arr = np.concatenate((np.zeros((ceil(int_arr.shape[0]/COL_SIZE)*COL_SIZE-int_arr.shape[0], int_arr.shape[1], int_arr.shape[2])), int_arr), axis=0)
			int_arr = int_arr.reshape(int_arr.shape+(1,))
			UPSCALE = ceil((Y_SCALE*self.height-3)/(int_arr.shape[1]*COL_SIZE))
			int_arr = np.repeat(int_arr, UPSCALE, axis=1) # Upscale height
			int_arr = np.repeat(int_arr, UPSCALE, axis=2) # Upscale width
			int_arr = np.repeat(int_arr, 3, axis=3) # Convert to RGB
			ar_width = ceil(int_arr.shape[0] / COL_SIZE)
			row_sep, col_sep = np.array([[[1,1,1] for n in range(int_arr.shape[1])]]), np.array([[[1,1,1]] for n in range(COL_SIZE*int_arr.shape[2]+COL_SIZE+1)])
			gen_col = lambda start_index: tuple([val for pair in zip([row_sep for i in range(COL_SIZE)], [int_arr[COL_SIZE*start_index+i] for i in range(COL_SIZE)]) for val in pair]+[row_sep])
			columns = [[col_sep, np.concatenate(gen_col(n), axis=0)] for n in range(ar_width)]
			joined_ints = np.concatenate(tuple([col for sub in columns for col in sub]+[col_sep]), axis=1)
			arr = np.concatenate((np.array([[[0,0,0] for n in range(arr.shape[1])] for i in range(joined_ints.shape[0]-arr.shape[0])]), arr), axis=0) # Add black rows to top of arr to make heights identical
			arr = np.concatenate((joined_ints,arr), axis=1)
		
		# Ensure that width and height are both divisible by 2, as ffmpeg seems to require that.
		if arr.shape[0] % 2 == 1:
			arr = np.concatenate((np.array([[[0,0,0] for n in range(arr.shape[1])]]), arr), axis=0)
		if arr.shape[1] % 2 == 1:
			arr = np.concatenate((np.array([[[0,0,0]] for n in range(arr.shape[0])]), arr), axis=1)
			
		image = Image.fromarray((arr*255).astype(np.uint8), 'RGB')
		self.last_image_size = (max(arr.shape[1], self.last_image_size[0]), max(arr.shape[0], self.last_image_size[1]))
		image.save(self.get_image_filename())
	# Remove the last image outputted.
	def pop_image(self):
		if platform.system() == 'Windows':
			subprocess.call('del '+self.get_image_filename(), shell=True)
		else:
			subprocess.call('rm '+self.get_image_filename(), shell=True)
		self.image_counter -= 1
	# Remove all outputted images.
	def clean_images(self):
		while self.image_counter > 0:
			self.pop_image()
	# Convert the set of outputted images into an mp4 movie and delete the images.
	def output_movie(self):
		subprocess.call('ffmpeg -framerate 7 -i '+self.filename+'_temp-image_%d.png -s:v '+str(self.last_image_size[0])+'x'+str(self.last_image_size[1])+
														' -c:v libx264 -profile:v high -crf 1 -pix_fmt yuv420p '+self.filename+'.mp4', shell=True)
		self.clean_images()
		self.last_image_size = (0,0)

if __name__ == '__main__':
	s = AutogenSimpleGame(8, 8, filename=sys.argv[1])
	s.pop_image()
	for i in range(1):
		s._reset()
		done = False
		while not done:
			_, done = s._step(self.prng.randint(0,3))
	s.output_movie()