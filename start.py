from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('LSTM_In.txt', num_epochs=2)

textgen.generate()

writer = open("file10k.txt", 'w+')

for i in range(10000):
    writer.write(str(textgen.generate(1, True)[0]))
    writer.write('\n')

writer.close()