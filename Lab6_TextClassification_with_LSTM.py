from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import requests
import os


#Download the ATIS training and test files and put them at the same folder as this notebook. 
#If you want to see how the model is predicting on new sentences you will also need the vocabulary files for queries and slots
def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)

locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']

data = {
  'train': { 'file': 'atis.train.ctf', 'location': 0 },
  'test': { 'file': 'atis.test.ctf', 'location': 0 },
  'query': { 'file': 'query.wl', 'location': 1 },
  'slots': { 'file': 'slots.wl', 'location': 1 }
}

for item in data.values():
    location = locations[item['location']]
    path = os.path.join('..', location, item['file'])
    if os.path.exists(path):
        print("Reusing locally cached:", item['file'])
        # Update path
        item['file'] = path
    elif os.path.exists(item['file']):
        print("Reusing locally cached:", item['file'])
    else:
        print("Starting download:", item['file'])
        url = "https://github.com/Microsoft/CNTK/blob/v2.0/%s/%s?raw=true"%(location, item['file'])
        download(url, item['file'])
        print("Download completed")

#Importing CNTK and other useful libraries
import math
import numpy as np

import cntk as C

# Select the right target device when this notebook is being tested:------------------------------------------1
'''
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

# Test for CNTK version-------------------------------------------------------------------------------------2
if not C.__version__ == "2.0":
    raise Exception("this lab is designed to work with 2.0. Current Version: " + C.__version__) 
'''

#Task and Model Structure-----------------------------------------------------------------------------------------------------
# number of words in vocab, slot labels, and intent labels
vocab_size = 943 ; num_labels = 129 ; num_intents = 26    

# model dimensions
input_dim  = vocab_size
label_dim  = num_labels
emb_dim    = 150
hidden_dim = 300

# Create the containers for input feature (x) and the label (y)
x = C.sequence.input_variable(vocab_size)
y = C.sequence.input_variable(num_labels)

def create_model():
    with C.layers.default_options(initial_state=0.1):
        return C.layers.Sequential([
            C.layers.Embedding(emb_dim, name='embed'),
            C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False),
            C.layers.Dense(num_labels, name='classify')
        ])

# peek
z = create_model()
print(z.embed.E.shape)
print(z.classify.b.value)

# Pass an input and check the dimension
z = create_model()
print(z(x).embed.E.shape)

#A Brief Look at Data and Data Reading
def create_reader(path, is_training):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
         query         = C.io.StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
         intent_unused = C.io.StreamDef(field='S1', shape=num_intents, is_sparse=True),  
         slot_labels   = C.io.StreamDef(field='S2', shape=num_labels,  is_sparse=True)
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

# peek
reader = create_reader(data['train']['file'], is_training=True)
print(reader.streams.keys())


#Trainer---------------------------------------------------------------------------------------------------------------------------------------------
#Learning note: Introduction to placeholder: Remember that the code we have been writing is not actually executing any heavy computation 
#it is just specifying the function we want to compute on data during training/testing. And in the same way that it is convenient to have names 
#for arguments when you write a regular function in a programming language, it is convenient to have placeholders that refer to arguments 
#(or local computations that need to be reused). Eventually, some other code will replace these placeholders with other known quantities in 
#the same way that in a programming language the function will be called with concrete values bound to its arguments.

#Specifically, the input variables you have created above x = C.sequence.input_variable(vocab_size) holds data pre-defined by vocab_size. 
#In the case where such instantiations are challenging or not possible, using placeholder is a logical choice. Having the placeholder only 
#allows you to defer the specification of the argument at a later time when you may have the data.

#Here is an example below that illustrates the use of placeholder.
def create_criterion_function(model):
    labels = C.placeholder(name='labels')
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return C.combine ([ce, errs]) # (features, labels) -> (loss, metric)

criterion = create_criterion_function(create_model())
criterion.replace_placeholders({criterion.placeholders[0]: C.sequence.input_variable(num_labels)})

#While the cell above works well when one has input parameters defined at network creation, 
#it compromises readability. Hence we prefer creating functions as shown below
def create_criterion_function_preferred(model, labels):
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return ce, errs # (model, labels) -> (loss, error metric)

def train(reader, model_func, max_epochs=10):
    
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(x)
    
    # Instantiate the loss and error function
    loss, label_error = create_criterion_function_preferred(model, y)

    # training config
    epoch_size = 18000        # 18000 samples is half the dataset size 
    minibatch_size = 70
    
    # LR schedule over epochs 
    # In CNTK, an epoch is how often we get out of the minibatch loop to
    # do other stuff (e.g. checkpointing, adjust learning rate, etc.)
    # (we don't run this many epochs, but if we did, these are good values)
    lr_per_sample = [0.003]*4+[0.0015]*24+[0.0003]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_rate_schedule(lr_per_minibatch, C.UnitType.minibatch, epoch_size)
    
    # Momentum schedule
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(700)
    
    # We use a the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from 
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.sgd(parameters=model.parameters,
                     lr=lr_schedule,
                     #momentum=momentum_as_time_constant,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True)

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    
    # Uncomment below for more detailed logging
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) 

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                x: reader.streams.query,
                y: reader.streams.slot_labels
            })
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()

#Train the model-------------------------------------------------------------------------------------------------------------------------------
def do_train():
    global z
    z = create_model()
    reader = create_reader(data['train']['file'], is_training=True)
    train(reader, z)
do_train()


#Evaluating the model
def evaluate(reader, model_func):
    
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(x)
    
    # Create the loss and error functions
    loss, label_error = create_criterion_function_preferred(model, y)

    # process minibatches and perform evaluation
    progress_printer = C.logging.ProgressPrinter(tag='Evaluation', num_epochs=0)

    while True:
        minibatch_size = 500
        data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
            x: reader.streams.query,
            y: reader.streams.slot_labels
        })
        if not data:                                 # until we hit the end
            break

        evaluator = C.eval.Evaluator(loss, progress_printer)
        evaluator.test_minibatch(data)
     
    evaluator.summarize_test_progress()


def do_test():
    reader = create_reader(data['test']['file'], is_training=False)
    evaluate(reader, z)

do_test()

print(z.classify.b.value)


# load dictionaries
query_wl = [line.rstrip('\n') for line in open(data['query']['file'])]
slots_wl = [line.rstrip('\n') for line in open(data['slots']['file'])]
query_dict = {query_wl[i]:i for i in range(len(query_wl))}
slots_dict = {slots_wl[i]:i for i in range(len(slots_wl))}

# let's run a sequence through
seq = 'BOS flights from new york to seattle EOS'
w = [query_dict[w] for w in seq.split()] # convert to word indices
print(w)
onehot = np.zeros([len(w),len(query_dict)], np.float32)
for t in range(len(w)):
    onehot[t,w[t]] = 1

#x = C.sequence.input_variable(vocab_size)
pred = z(x).eval({x:[onehot]})[0]
print(pred.shape)
best = np.argmax(pred,axis=1)
print(best)
print(list(zip(seq.split(),[slots_wl[s] for s in best])))