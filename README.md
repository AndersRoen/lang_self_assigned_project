# lang_self_assigned_project

github repo link: https://github.com/AndersRoen/lang_self_assigned_project.git

## Language Analytics self assigned project description:
This project looks at how deep learning can be used to learn patterns in text, hopefully so well that it learns how to generate text in that style itself. The goal of the project was to train a model on a large corpus of poetry from the Gutenberg Project, and provide an attempt at generating new text in the same style. The project was heavily inspired by similar projects on Shakespeare's sonnets. This was done by leaning on methods provided by a TensorFlow guide on Text Generation, found here: https://www.tensorflow.org/text/tutorials/text_generation 
As we've worked quite a bit with tools from TensorFlow during this course, most of it is quite similar to principles we've worked with in class, such as vectorizing text, making it to sequences and training a model to make predictions on the basis of the training data. Overall, this is what this script does, however the model is built in a different way from what we've learned - namely using the Object Oriented Programming aproach, which entails building ```class``` objects.
RNNs are well equipped to handly such Text Generation tasks, as it can maintain an internal state, and can thus take the predictions it has already handled into account.

## Methods
Ass mentioned in the description, RNNs are good for Text Generation tasks, but they are not the only option. One could have done something quite similar using, for instance, transformers from ```Hugging Face```. Perhaps it could have been interesting to compare the output of the two approaches, but unfortunately that was a bit beyond the scope of this project.
As it is, the ```gutenberg_generator.py``` script loads in the gutenberg corpus, and extracts the desired training size. It does this, as the corpus is quite massive and would have taken far too long for the model to train. I ended up using a training size of roughly the same amount of characters in the TensorFlow tutorial, but the program allows for larger or smaller training sizes.
The code then prepares the training data for the model. As mentioned in the description, the model is built using Object Oriented Programming, meaning that the model is made as a ```Class``` object - but in principle it is not much different from the models we've built in class. The embedding layer, RNN layer - called ```GRU``` and the output layer is added using the ```self``` function, but these could have been defined outside of the ```Class``` object just as easily. Inside the model object a function that maintains the models internal state is also defined.
The model is then compiled and trained for a user-defined amount of epochs.
Then, the predictor model is defined, again building a ```Class``` object. This is taken largely unedited from the TensorFlow guide previously linked to. Essentially, the predictor takes strings and converts them to token IDs, makes its predictions and converts the resulting predictions back into character strings. It is also forced to make predictions, as a mask that disallows generating ```UNKs```. 
Lastly, the script generates text. It does this by taking a user-defined start word, and then predicts the next characters after that. The length of the text is also user-defined.

## Usage
To run this script, you should first unzip the gutenberg corpus in the ```in``` folder. Then, you should run the ```lang_setup.sh``` script. After that, point the command line to ```lang_self_assigned_project``` folder.
The script has the follow command line arguments that are needed for the code to run:
 - ```-t``` which is the name of the generated ```txt``` file
 -  ```-e``` which is the size of the embedding layer, I set this to 256
 -  ```-tr``` which is the size of the training data, measured in characters. I
 -  ```-ep``` which is the amount of epochs you want to train for. I trained for 35
 -  ```-bs``` which is the batch size of the model. I used a batch size of 64
 -  ```-lr``` which is the learning rate of the model. I used a learning rate of 0.001
 -  ```-gl``` which is the length of the generated text, measured in characters. The longer this is, the more nonsensical the generated text will eventually be
 -  ```-sw``` which is the start word of the generated text. I used the word "with"
 
 ## Results
 The generated text is a bit messy. That is due to a couple of things. Firstly, the training text was messy in the first place, with a lot of words being stuck to other words, messy punctuation and so on. A cleaner corpus had probably resulted in a cleaner generated text. Next, it doesn't really look like the model has learned to make poetry that well. It can definitely make real words, even grammatical bits, but the "sentences" doesn't really mean anything, they seem to be quite random. A way to fix that would be to make it train for longer, on a larger set of training data. However, that would take a really long time, and was probably a bit beyond the scope of this paper. For instance, training on the full corpus would take around 7 hours per epoch, with the largest machine available to me on UCloud. Another thing, this script doesn't allow for the model to correct its own mistakes, which probably could have helped it make more coherent text. However, the script does what I wanted it to - it can generate text that is fairly reminiscent of the training corpus, be it good or bad. I'm confident that with more time, and maybe a more powerful machine, better results could have been achieved.
