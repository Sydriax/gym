import numpy as np
from PIL import Image
import sys, subprocess, platform, random
from math import *
import gym

class SimpleGame(gym.Env):
	# Initialize the game environment
	def __init__(self, width=8, height=8, filename = ''):
		self.channels = 3 # Useful metadata for general algos
		self.num_actions = 4 # Useful metadata for general algos
		self.width = width
		self.height = height
		self.observable_state = None
		self.player_loc = None
		self.filename = filename
		self.image_counter = 0
		self.last_size = None
		self.prng = random.Random()
		self._reset()
		
	def __repr__(self):
		str, d = '', { (0,0,1): 'B', (0,0,0): '*', (0,1,0): 'G', (1,0,0): 'R', (1,1,1): 'W' }
		for row in self.observable_state.tolist():
			for site in row:
				str += d[tuple(site)]
			str += '\n'
		return str[:-1]
	
	def _seed(self, seed=None):
		if seed is None: self.prng.seed(random.randint(0,2**32-1)) # The reason to use this over raw time is that if we repeated reseed very quickly this works better
		else: self.prng.seed(seed)
				
	# Reset the state completely.
	def _reset(self):
		def get_random_location():
			return (self.prng.randint(0,self.width-1),self.prng.randint(0,self.height-1))
		self.observable_state = np.zeros((self.height,self.width,3))
		self.images = []
		# Generate 2% <= red <= 10% points
		num_red = self.prng.randint(round(0.02*self.width*self.height), round(0.1*self.width*self.height))
		for red in range(num_red):
			r_loc = get_random_location()
			self.observable_state[r_loc[1]][r_loc[0]] = [1,0,0]
		# Generate 2% <= green <= 10% points
		num_green = self.prng.randint(round(0.02*self.width*self.height), round(0.1*self.width*self.height))
		for green in range(num_green):
			g_loc = get_random_location()
			while not np.array_equal(self.observable_state[g_loc[1]][g_loc[0]], [0,0,0]):
				g_loc = get_random_location()
			self.observable_state[g_loc[1]][g_loc[0]] = [0,1,0]
		# Generate a blue point
		b_loc = get_random_location()
		while not np.array_equal(self.observable_state[b_loc[1]][b_loc[0]], [0,0,0]):
			b_loc = get_random_location()
		self.observable_state[b_loc[1]][b_loc[0]] = [1,1,1]
		# Generate a white (player) point
		w_loc = get_random_location()
		while not np.array_equal(self.observable_state[w_loc[1]][w_loc[0]], [0,0,0]):
			w_loc = get_random_location()
		self.observable_state[w_loc[1]][w_loc[0]] = [0,0,1]
		self.player_loc = w_loc
		return self.observable_state
	
	# Make a move given a action, _step the state, and return the reward and whether the game is over.
	# 0: Up, 1: Right, 2: Down, 3: Left
	def _step(self, action):
		# We're going to do something horrifying here, and if the type of the action is 'list' we will take the second element as intermediate output and the first as the action.
		if type(action) == list:
			intermediate_outputs = action.pop()
			action = action[0]
		else:
			intermediate_outputs = None
		def get_result(state, loc):
			if np.array_equal(state[loc[1]][loc[0]], [0,0,0]): return -0.01, False
			if np.array_equal(state[loc[1]][loc[0]], [1,0,0]): return -1, False
			if np.array_equal(state[loc[1]][loc[0]], [0,1,0]): return 1, False
			if np.array_equal(state[loc[1]][loc[0]], [1,1,1]): return 3, True
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
		if not done or reward == 3: self.observable_state[self.player_loc[1]][self.player_loc[0]] = [0,0,1]
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
			# print(UPSCALE, joined_ints.shape, arr.shape, np.array([[[0,0,0] for n in range(X_SCALE*self.width)] for i in range(joined_ints.shape[0]-arr.shape[0])]).shape)
			arr = np.concatenate((np.array([[[0,0,0] for n in range(X_SCALE*self.width+2)] for i in range(joined_ints.shape[0]-arr.shape[0])]), arr), axis=0) # Add black rows to top of arr to make heights identical
			arr = np.concatenate((joined_ints,arr), axis=1)
			if arr.shape[0] % 2 == 1:
				arr = np.concatenate((np.array([[[0,0,0] for n in range(arr.shape[1])]]), arr), axis=0)
			if arr.shape[1] % 2 == 1:
				arr = np.concatenate((np.array([[[0,0,0]] for n in range(arr.shape[0])]), arr), axis=1)
			
		image = Image.fromarray((arr*255).astype(np.uint8), 'RGB')
		self.last_image_size = (arr.shape[1], arr.shape[0])
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
		subprocess.call('ffmpeg -framerate 5 -i '+self.filename+'_temp-image_%d.png -s:v '+str(self.last_image_size[0])+'x'+str(self.last_image_size[1])+
														' -c:v libx264 -profile:v high -crf 1 -pix_fmt yuv420p '+self.filename+'.mp4', shell=True)
		self.clean_images()

if __name__ == '__main__':
	s = SimpleGame(8, 8, filename=sys.argv[1])
	s.pop_image()
	for i in range(1):
		s._reset()
		done = False
		while not done:
			_, done = s._step(self.prng.randint(0,3))
	s.output_movie()