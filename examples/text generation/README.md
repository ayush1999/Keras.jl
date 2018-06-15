# Text Generation.

Producing text using Project Gutenburg (Alice in Wonderland).

## How?

In order to train the model, you need to run `python main.py` from the terminal.
This will train the entire model for 20 epochs and generate the model weight, structure and 
other relevant files.

## Producing text.

One the model has been trained and saved, it can be easily imported into Flux, and run to produce 
artificial text using `julia main.jl` from the terminal.

```
$ julia main.jl

Original text: went straight on like a tunnel for some way, and then
dipped suddenly down, so suddenly that alice had not a mome

Produced text: went straight on like a tunnel for some way, and then
dipped suddenly down, so suddenly that alice had not a mome tf the toiet
sh she wonld tare th teen the woued has ena oo her hend th the thete sai 
a little  aaden oo the soodd th the wirh oh  ‘het toe toit oo 

$
```