# CSC413-Project

OpenAI’s new NLP model, ChatGPT-3, has been used by many people in different fields. As it can produce responses within seconds, overflowing the internet using the responses without people noticing can be a hidden hazard. We decided to train a Transformer model to identifies AI responses in Open Communication Forms (e.g. Reddit, StackOverflow, etc) to help people to distinguish human response and AI response.

## Model:

For this project, we are using pre-trained BERT (Bidirectional Encoder Representations from Transformers) model and fine tune it using our dataset.
The following diagram showed how BERT process the embedded word to output.

![image](https://user-images.githubusercontent.com/55767924/232256506-2d9fb234-d826-4da2-a1b1-f3cc22014895.png)
*[source](https://arxiv.org/abs/1810.04805v1)*

Since the input contains multiple sentences, after tokenized each word and concatenate the sentences together, it will be process the same way into the model.

![image](https://user-images.githubusercontent.com/55767924/232256586-6d069de1-8673-41e4-9949-f7308f799a60.png)
*[source](https://arxiv.org/abs/1810.04805v1)*

There are in total 110M parameters according to https://huggingface.co/bert-base-uncased.

### Examples:



 - Success Prediction: 

{'text': "For individuals who want to improve their own vocabulary, there are several strategies that can be effective, including: Reading widely: Reading books, articles, and other written materials can expose you to a variety of new words and their usage in context. It's important to choose materials that are appropriate for your current level of reading comprehension and gradually work your way up to more challenging texts. Using a dictionary: When you encounter a new word, look it up in a dictionary to learn its definition and usage. This can help you remember the word and use it correctly in the future. Playing word games: Word games such as Scrabble, crossword puzzles, and word searches can be fun and effective ways to expand your vocabulary and improve your word recognition skills. Learning word roots and affixes: Many words in the English language share common roots and affixes. Learning these can help you recognize the meaning of unfamiliar words based on their prefixes, suffixes, and roots. Practicing using new words: To truly master a new word, it's important to use it in context and practice incorporating it into your own writing and speech.", 'label': 0} 

prediction: [1, 0]

 - Failed Prediction: 

{'text': "Habitat for humanity ReStore is a good place to donate stuff It's a not for profit place. Location is Dixie and just north of Eastgate. All other places like V.V and Talize all out there overpricing shit for everyone now.", 'label': 1}

prediction: [1, 0]

## Data:

For this project, since we are fine tuning our model, we only need a small set of data to train.

We collected our data from searching for questions and discussions on open communication platforms (e.g. Reddit, StackOverflow, etc). Then, ask some questions to ChatGPT-4 for AI answers.

For processing the data, we split

### Summary:

In order to be used in BERT model, we tokenized the input using the builtin function from AutoTokenizer by Hugging Face. Since the model has input limit, we truncated the inputs, also added padding to shorter inputs to reduce bias on input length.


## Training:

The diagram below is our training curve of our final model:

![image](https://user-images.githubusercontent.com/83336699/233804230-f64fdef0-5211-43aa-8087-e8795e188a5d.png)
![f06681479e94ec39989a53ea3b0f23a](https://user-images.githubusercontent.com/83336699/233804496-2f490243-52ba-498c-8726-370f8d63874e.png)
![70c579a8399ec02ce20977539f98110](https://user-images.githubusercontent.com/83336699/233804499-62c5ae50-418d-4a08-b440-9d417299f99c.png)


### Hyper-parameters

- Learning rate: 
We start as a bigger value with alpha 0.01 and find out it is too big that make our loss and accuracy bounce frequently in the graph. Therefore, we slowly descrease its value and find out 1e-6 provide us the best validation accuracy and learning curve.

- Epoches:
For the number of epoches to train the model, we started from 5 and 10, which the model only predicts 1 as the output. Once we increase the epoches to 20 and above, the model starts to have correct prediction.

- Weight Decay:
For the weight decay, we start with 0.01 to tune and decrease it slowly. It turns out the weights decay does not influence our model much. Therefore, we choose the 0.001 which has the best performance to avoid overfit. 

## Results:

We used test set that was part of the original dataset and measure its accuracy from our model.
We obtained around 87% test accuracy which is reasonably reliable.

Our method preformed relatively well given our problem of identifying AI response. 

#TODO

## Ethical Consideration:



## Authors

Yuxuan Mu - Collecting data from online platform and ask the questions on chatGPT. While training the model, prepare test data.

William Chau and HanchengHuang - Implement the model using Hugging Face and prepare for the fine tuning. Once the dataset is ready, tune the hyperparameters and collect the result.
