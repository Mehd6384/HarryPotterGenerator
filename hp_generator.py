import unidecode 
import string 
import random
import re 

all_characters = string.printable 
n_characters = len(all_characters)
# print(n_characters)

file = unidecode.unidecode(open('data/HP1.txt').read())
file_len = len(file)
# print('length: ', file_len)


chunk_size = 200
def random_chunk():

	start_ind = random.randint(0, file_len - (chunk_size+1))
	end = start_ind + chunk_size

	return file[start_ind:end]

# print(random_chunk())


import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 


class RNN(nn.Module): 

	def __init__(self, input_size, hidden_size, output_size, n_layers = 1): 

		nn.Module.__init__(self)

		self.n_layers = n_layers
		self.hidden_size = hidden_size

		self.encoder = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.decoder = nn.Linear(hidden_size, output_size)

	def forward(self, x, hidden): 

		x = self.encoder(x.reshape(1,-1))
		out,hidden = self.gru(x.reshape(1,1,-1), hidden)
		out = self.decoder(out.reshape(1,-1))
		return out,hidden

	def init_hidden(self):
		return torch.zeros(self.n_layers, 1, self.hidden_size)

def char_to_tensor(list_of_strings): 
	# print('list of string: {}\nLength: {}'.format(list_of_strings, len(list_of_strings)))

	tensor = torch.zeros(len(list_of_strings)).long()
	for c in range(len(list_of_strings)): 
		# print('Letter: {} = {} -> index: {}'.format(c, list_of_strings[c], all_characters.index(list_of_strings[c])))
		tensor[c] = all_characters.index(list_of_strings[c])

	return tensor

def random_training_set(): 

	chunk = random_chunk()
	inp = char_to_tensor(chunk[:-1])
	target = char_to_tensor(chunk[1:])

	return inp, target

def evaluate(prime_str = 'A', predict_len = 100, temperature = 0.8): 

	hidden = decoder.init_hidden()
	prime_input = char_to_tensor(prime_str)
	predicted = prime_str

	for p in range(len(prime_str)-1):
		_,hidden = decoder(prime_input[p], hidden)

	inp = prime_input[-1]

	for p in range(predict_len): 

		out, hidden = decoder(inp, hidden)

		output_dist = out.reshape(-1).div(temperature).exp()
		top_i = torch.multinomial(output_dist, 1)[0]

		predicted_char = all_characters[top_i]
		predicted += predicted_char
		inp = char_to_tensor(predicted_char)

	return predicted


def train(inp, target): 

	hidden = decoder.init_hidden()


	loss = 0. 
	for c in range(chunk_size-1): 
		output, hidden = decoder(inp[c], hidden)
		# print(output)
		# print(output.shape)
		# print(target[c])

		my_target = torch.zeros(1,100)
		my_target[0,target[c]] = 1.
		loss += F.binary_cross_entropy_with_logits(output, my_target)

	adam.zero_grad()
	loss.backward()
	adam.step()

	return loss.item()/chunk_size

n_epochs = 5000
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1 
lr = 1e-3


decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder.load_state_dict(torch.load('hp_gen'))
adam = optim.Adam(decoder.parameters(), lr)

criterion = nn.CrossEntropyLoss()

recap = []
loss_avg = 0. 

import matplotlib.pyplot as plt 
plt.style.use('dark_background')
f, ax = plt.subplots()

for epoch in range(1, n_epochs +1): 
	loss =train(*random_training_set())
	loss_avg += loss

	if epoch%print_every == 0 : 
		print('Epoch: {} | Loss: {:.6f}'.format(epoch, loss))
		evaluationSOS = [all_characters[random.randint(10,50)] for _ in range(2)]
		evaluationSOS = evaluationSOS[0] + evaluationSOS[1]
		print(evaluate(evaluationSOS, 100), '\n\n')

		torch.save(decoder.state_dict(), 'hp_gen')


	if epoch % plot_every == 0: 
		recap.append(loss_avg/plot_every)
		loss_avg = 0.
		ax.clear()
		ax.plot(recap)
		plt.pause(0.1)


torch.save(decoder.state_dict(), 'hp_gen')

plt.show()