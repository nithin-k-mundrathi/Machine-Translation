# Machine-Translation

In the following experiment, we use seq-to-seq encoder-decoder architecture for Machine Translation. We Train Two different Models here Machine Translation with Attention and without Attention. The Source Language we use here is Vietnamese and Target Language is English.The Attention code is written from scratch and I used  Basic dot product Attention rather than Additive Attention and Multiplicative Attention which is simple to compute using Keras backend.

![image](https://user-images.githubusercontent.com/47551095/123563154-fc2aaf00-d7aa-11eb-8289-1940c926406e.png)


The Below Are Few examples of the prediction for Machine Translation with Attention.

Example-1
![image](https://user-images.githubusercontent.com/47551095/123562884-51fe5780-d7a9-11eb-99fe-b5f8f2404b7d.png)

Example-2
![image](https://user-images.githubusercontent.com/47551095/123562896-693d4500-d7a9-11eb-9ed8-53b2de0fe262.png)

The Evaluation Metric which we use here is Blue_Score, which compares how the predicted N-grams are deviating from Ground Truth N-grams

![image](https://user-images.githubusercontent.com/47551095/123563036-36e01780-d7aa-11eb-86b3-a2c2e1cc4368.png)
