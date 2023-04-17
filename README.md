# CSC413-Project

OpenAI’s new NLP model, ChatGPT-4, has been used by many people in different fields. As it can produce responses within seconds, overflowing the internet using the responses without people noticing can be a hidden hazard. We decided to train a Transformer model to identifies AI responses in Open Communication Forms (e.g. Reddit, StackOverflow, etc) to help people to distinguish human response and AI response.

## Model:
– A figure/diagram of the model architecture that demonstrates understanding of the steps involved in
computing the forward pass
– Count the number of parameters in the model, and a description of where the parameters come from
– Examples of how the model performs on two actual examples from the test set: one successful and one
unsuccessful

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

 - Failed Prediction:


## Data:
– Describe the source of your data
– Provide summary statistics of your data to help interpret your results (similar to in the proposal)
– Describe how you transformed the data (e.g. any data augmentation techniques)
– If appropriate to your project, describe how the train/validation/test set was split. (Note that splitting
the training/validation/test set is not always straightforward!)

For this project, since we are fine tuning our model, we only need a small set of data to train.

We collected our data from searching for questions and discussions on open communication platforms (e.g. Reddit, StackOverflow, etc). Then, ask some questions to ChatGPT-4 for AI answers.

### Summary:
- Total number of data: 250 answers from human 250 answers from GPT-4, 500 in total.
- The labels of  the question will  be only  questions and answers and the targets will be 0 = AI generated and 1 = Human generated.
- We decided to have 200 answers for the training set, 200 answers for the validation set and 100 answers for the test set. Each contains same numbers of answers from human and AI. 

In order to be used in BERT model, we tokenized the input using the builtin function from AutoTokenizer by Hugging Face. Since the model has input limit, we truncated the inputs, also added padding to shorter inputs to reduce bias on input length.


## Training:
– The training curve of your final model
– A description how you tuned hyper-parameters

The diagram below is our training curve of our final model:

![91e98ba08b41d8637fc01da3bd9cb08](https://user-images.githubusercontent.com/83336699/232581044-9b0f014f-8680-4abf-a617-8815dbbe1df7.png)
![68821f921a0b5ed9dec3a347667ed6c](https://user-images.githubusercontent.com/83336699/232581062-5a4fb286-8236-4d60-8aed-de90cf46fb6e.png)


### Hyper-parameters

- Learning rate: 
We start as a bigger value with alpha 0.01 and find out it is too big that make our loss and accuracy bounce frequently in the graph. Therefore, we slowly descrease its value and find out 1e-6 provide us the best validation accuracy and learning curve.

- Epoches:
For the number of epoches to train the model, we started from 5 and 10, which the model only predicts 1 as the output. Once we increase the epoches to 20 and above, the model starts to have correct prediction.

- Weight Decay:


## Results:
– Describe the quantitative measure that you are using to evaluate your result
– Describe the quantitative and qualitative results
– A justification that your implemented method performed reasonably, given the difficulty of the problem—or
a hypothesis for why it doesn’t (this is extremely important)

We used test set that was part of the original dataset and measure its accuracy from our model.
We obtained around 87% test accuracy which is reasonably reliable.

Our method preformed relatively well given our problem of identifying AI response. 

## Ethical Consideration:
– Description of a use of the system that could give rise to ethical issues. Are there limitations of your
model? Your training data?



## Authors
– A description of how the work was split—i.e. who did what in this project.

Yuxuan Mu - Collecting data

William Chau - Implement the model, Tune the model

HanchengHuang - Tune the model, Implement the model
